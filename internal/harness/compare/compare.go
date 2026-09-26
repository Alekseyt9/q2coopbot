// Package compare summarizes recorded suites. It does not replay physics or
// claim causal regression from noisy timing measurements.
package compare

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

type File struct {
	Path string `json:"path"`
	Hash string `json:"sha256"`
}
type Manifest struct {
	Version            int    `json:"version"`
	SourceUnchanged    bool   `json:"source_unchanged"`
	ArtifactsUnchanged bool   `json:"artifacts_unchanged"`
	Artifacts          []File `json:"artifact_files"`
	Sources            []File `json:"source_files"`
}
type Run struct {
	Fixture  string         `json:"fixture_fingerprint"`
	Accepted *bool          `json:"accepted"`
	Metrics  map[string]any `json:"metrics"`
	Speed    map[string]any `json:"speed"`
}
type Suite struct {
	Runs         int      `json:"runs"`
	Provenance   bool     `json:"provenance_valid"`
	ManifestPath string   `json:"manifest"`
	Results      []Run    `json:"results"`
	Manifest     Manifest `json:"-"`
}

func Load(path string) (Suite, error) {
	var s Suite
	data, err := os.ReadFile(path)
	if err != nil {
		return s, err
	}
	if err = json.Unmarshal(data, &s); err != nil {
		return s, err
	}
	p := s.ManifestPath
	if !filepath.IsAbs(p) {
		p = filepath.Join(filepath.Dir(path), p)
	}
	data, err = os.ReadFile(p)
	if err != nil {
		return s, err
	}
	if err = json.Unmarshal(data, &s.Manifest); err != nil {
		return s, err
	}
	if s.Runs == 0 || s.Runs != len(s.Results) {
		return s, fmt.Errorf("suite run count is incomplete")
	}
	for _, r := range s.Results {
		if r.Fixture == "" || r.Accepted == nil {
			return s, fmt.Errorf("run lacks fixture fingerprint or verdict")
		}
	}
	return s, nil
}

type Stats struct {
	Count int     `json:"count"`
	Mean  float64 `json:"mean"`
	Min   float64 `json:"min"`
	Max   float64 `json:"max"`
}
type Metric struct {
	Before *Stats   `json:"before"`
	After  *Stats   `json:"after"`
	Delta  *float64 `json:"mean_delta,omitempty"`
}
type Group struct {
	Fixture        string            `json:"fixture_fingerprint"`
	Status         string            `json:"status"`
	BeforeRuns     int               `json:"before_runs"`
	AfterRuns      int               `json:"after_runs"`
	BeforeAccepted int               `json:"before_accepted"`
	AfterAccepted  int               `json:"after_accepted"`
	Metrics        map[string]Metric `json:"metrics,omitempty"`
}
type Report struct {
	Compatible    bool     `json:"compatible"`
	Reasons       []string `json:"reasons"`
	SourceChanges []string `json:"source_changes"`
	ClientChanged bool     `json:"client_changed"`
	Groups        []Group  `json:"groups"`
}

func hash(m Manifest, name string) string {
	for _, f := range m.Artifacts {
		if f.Path == name {
			return f.Hash
		}
	}
	return ""
}
func changes(a, b []File) []string {
	left, right := map[string]string{}, map[string]string{}
	keys := map[string]bool{}
	for _, f := range a {
		left[f.Path] = f.Hash
		keys[f.Path] = true
	}
	for _, f := range b {
		right[f.Path] = f.Hash
		keys[f.Path] = true
	}
	out := []string{}
	for k := range keys {
		if left[k] != right[k] {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}
func scalarValues(prefix string, m map[string]any, out map[string]float64) {
	for k, v := range m {
		key := prefix + "." + k
		switch v := v.(type) {
		case float64:
			out[key] = v
		case map[string]any:
			scalarValues(key, v, out)
		}
	}
}
func summarize(runs []Run) map[string]*Stats {
	out := map[string]*Stats{}
	for _, r := range runs {
		values := map[string]float64{}
		scalarValues("metrics", r.Metrics, values)
		scalarValues("speed", r.Speed, values)
		for k, v := range values {
			s := out[k]
			if s == nil {
				s = &Stats{Min: v, Max: v}
				out[k] = s
			}
			s.Count++
			s.Mean += v
			s.Min = min(s.Min, v)
			s.Max = max(s.Max, v)
		}
	}
	for _, s := range out {
		s.Mean /= float64(s.Count)
	}
	return out
}
func Compare(a, b Suite) Report {
	r := Report{Compatible: true, Reasons: []string{}, SourceChanges: changes(a.Manifest.Sources, b.Manifest.Sources), Groups: []Group{}}
	if !a.Provenance || !b.Provenance || !a.Manifest.SourceUnchanged || !b.Manifest.SourceUnchanged || !a.Manifest.ArtifactsUnchanged || !b.Manifest.ArtifactsUnchanged || a.Manifest.Version != 1 || b.Manifest.Version != 1 {
		r.Reasons = append(r.Reasons, "unverified_provenance")
	}
	ah, bh := hash(a.Manifest, "q2scenario-report.exe"), hash(b.Manifest, "q2scenario-report.exe")
	if ah == "" || bh == "" || ah != bh {
		r.Reasons = append(r.Reasons, "analyzer_changed_or_unknown: regenerate both reports with the same analyzer")
	}
	ac, bc := hash(a.Manifest, "q2coopbot.exe"), hash(b.Manifest, "q2coopbot.exe")
	if ac == "" || bc == "" {
		r.Reasons = append(r.Reasons, "client_hash_missing")
	}
	r.ClientChanged = ac != bc
	left, right := map[string][]Run{}, map[string][]Run{}
	keys := map[string]bool{}
	for _, v := range a.Results {
		left[v.Fixture] = append(left[v.Fixture], v)
		keys[v.Fixture] = true
	}
	for _, v := range b.Results {
		right[v.Fixture] = append(right[v.Fixture], v)
		keys[v.Fixture] = true
	}
	names := []string{}
	for k := range keys {
		names = append(names, k)
	}
	sort.Strings(names)
	baseOK := len(r.Reasons) == 0
	for _, key := range names {
		x, y := left[key], right[key]
		g := Group{Fixture: key, Status: "comparable", BeforeRuns: len(x), AfterRuns: len(y)}
		for _, v := range x {
			if v.Accepted != nil && *v.Accepted {
				g.BeforeAccepted++
			}
		}
		for _, v := range y {
			if v.Accepted != nil && *v.Accepted {
				g.AfterAccepted++
			}
		}
		if len(x) == 0 || len(y) == 0 {
			g.Status = "unmatched_fixture"
			r.Reasons = append(r.Reasons, "unmatched_fixture: "+key)
		} else if !baseOK {
			g.Status = "incompatible_provenance"
		} else {
			before, after := summarize(x), summarize(y)
			g.Metrics = map[string]Metric{}
			metricKeys := map[string]bool{}
			for k := range before {
				metricKeys[k] = true
			}
			for k := range after {
				metricKeys[k] = true
			}
			for k := range metricKeys {
				m := Metric{Before: before[k], After: after[k]}
				if m.Before != nil && m.After != nil {
					d := m.After.Mean - m.Before.Mean
					m.Delta = &d
				}
				g.Metrics[k] = m
			}
		}
		r.Groups = append(r.Groups, g)
	}
	r.Compatible = len(r.Reasons) == 0
	return r
}
