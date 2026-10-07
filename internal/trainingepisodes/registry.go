// Package trainingepisodes describes model-independent experiments, not gameplay.
package trainingepisodes

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"

	"q2coopbot/internal/policy"
)

type Manifest struct {
	Version int      `json:"version"`
	Files   []string `json:"files"`
}
type Seeds struct {
	Start int `json:"start"`
	Count int `json:"count"`
}
type Recipe struct {
	Runner       string `json:"runner"`
	Loadout      string `json:"loadout,omitempty"`
	Mixed        bool   `json:"mixed,omitempty"`
	RewardConfig string `json:"reward_config,omitempty"`
}
type Episode struct {
	Generator    *Generator       `json:"generator,omitempty"`
	Version      int              `json:"version"`
	ID           string           `json:"id"`
	Revision     int              `json:"revision"`
	Title        string           `json:"title"`
	Status       string           `json:"status"`
	Map          string           `json:"map"`
	Scope        string           `json:"scope"`
	Goal         string           `json:"goal"`
	Monsters     []string         `json:"monsters"`
	Geometry     string           `json:"geometry"`
	InitialState string           `json:"initial_state"`
	Skill        int              `json:"skill"`
	Timescale    int              `json:"timescale"`
	GameFrames   int              `json:"game_frames"`
	Modes        []string         `json:"modes"`
	PPOTrainable bool             `json:"ppo_trainable"`
	Splits       map[string]Seeds `json:"splits"`
	Recipe       Recipe           `json:"recipe"`
	Missing      []string         `json:"missing"`
}
type Registry struct {
	Path     string            `json:"path"`
	SHA256   string            `json:"sha256"`
	Episodes []Episode         `json:"episodes"`
	Hashes   map[string]string `json:"episode_hashes"`
}

func Hash(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:]), nil
}
func read(path string, v any) error {
	f, e := os.Open(path)
	if e != nil {
		return e
	}
	defer f.Close()
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	if e = d.Decode(v); e != nil {
		return e
	}
	var extra any
	if e = d.Decode(&extra); e != io.EOF {
		return fmt.Errorf("trailing JSON in %s", path)
	}
	return nil
}

var name = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)
var splitNames = []string{"train", "validation", "test", "confirmation"}

func Load(path string) (*Registry, error) {
	abs, e := filepath.Abs(path)
	if e != nil {
		return nil, e
	}
	var m Manifest
	if e = read(abs, &m); e != nil {
		return nil, e
	}
	if m.Version != 1 || len(m.Files) == 0 {
		return nil, fmt.Errorf("unsupported or empty training registry")
	}
	r := &Registry{Path: abs, Hashes: map[string]string{}}
	r.SHA256, e = Hash(abs)
	if e != nil {
		return nil, e
	}
	seen := map[string]bool{}
	files := map[string]bool{}
	type interval struct {
		lo, hi int
		key    string
	}
	var ranges []interval
	for _, file := range m.Files {
		if filepath.Base(file) != file || filepath.Ext(file) != ".json" || !name.MatchString(file[:len(file)-5]) || files[file] {
			return nil, fmt.Errorf("invalid or duplicate registry file %q", file)
		}
		files[file] = true
		var ep Episode
		p := filepath.Join(filepath.Dir(abs), file)
		if e = read(p, &ep); e != nil {
			return nil, fmt.Errorf("%s: %w", file, e)
		}
		if e = ep.validate(); e != nil {
			return nil, fmt.Errorf("%s: %w", file, e)
		}
		if seen[ep.ID] {
			return nil, fmt.Errorf("duplicate episode ID %s", ep.ID)
		}
		seen[ep.ID] = true
		for split, s := range ep.Splits {
			ranges = append(ranges, interval{s.Start, s.Start + s.Count - 1, ep.ID + ":" + split})
		}
		h, e := Hash(p)
		if e != nil {
			return nil, e
		}
		r.Hashes[ep.ID] = h
		r.Episodes = append(r.Episodes, ep)
	}
	entries, e := os.ReadDir(filepath.Dir(abs))
	if e != nil {
		return nil, e
	}
	for _, f := range entries {
		if !f.IsDir() && filepath.Ext(f.Name()) == ".json" && f.Name() != filepath.Base(abs) && !files[f.Name()] {
			return nil, fmt.Errorf("unregistered episode file %s", f.Name())
		}
	}
	sort.Slice(ranges, func(i, j int) bool { return ranges[i].lo < ranges[j].lo })
	for i := 1; i < len(ranges); i++ {
		if ranges[i].lo <= ranges[i-1].hi {
			return nil, fmt.Errorf("seed leakage between %s and %s", ranges[i-1].key, ranges[i].key)
		}
	}
	return r, nil
}
func (e Episode) validate() error {
	if e.Version != 1 || e.Revision < 1 || !name.MatchString(e.ID) || e.Title == "" || e.Scope == "" || e.Goal == "" || e.Geometry == "" || e.InitialState == "" || len(e.Monsters) == 0 {
		return fmt.Errorf("incomplete episode description")
	}
	if e.Status != "runnable" && e.Status != "planned" {
		return fmt.Errorf("unknown episode status")
	}
	if e.Map != "base1" && e.Map != "base2" || e.Skill < 0 || e.Skill > 3 || e.Timescale != 2 || e.GameFrames < 20 || e.GameFrames > 10000 {
		return fmt.Errorf("invalid runtime conditions")
	}
	if len(e.Splits) != len(splitNames) {
		return fmt.Errorf("four disjoint splits required")
	}
	for _, split := range splitNames {
		s, ok := e.Splits[split]
		if !ok || s.Start < 0 || s.Count < 1 || int64(s.Start)+int64(s.Count) > 2147483648 {
			return fmt.Errorf("invalid %s seeds", split)
		}
	}
	seen := map[string]bool{}
	for _, mode := range e.Modes {
		if mode != "rules" && mode != "learned" || seen[mode] {
			return fmt.Errorf("invalid controller modes")
		}
		seen[mode] = true
	}
	if len(seen) == 0 {
		return fmt.Errorf("controller modes required")
	}
	if e.Status == "planned" {
		if len(e.Missing) == 0 || e.Recipe.Runner != "" || e.PPOTrainable {
			return fmt.Errorf("planned episode must describe missing support without a runnable recipe")
		}
		return nil
	}
	if len(e.Missing) != 0 {
		return fmt.Errorf("runnable episode still has missing support")
	}
	if e.Generator != nil {
		if err := e.Generator.validate(e); err != nil {
			return err
		}
	}
	switch e.Recipe.Runner {
	case "combat-baseline":
		if e.Map != "base1" || e.GameFrames < 150 || e.GameFrames > 500 || !e.PPOTrainable {
			return fmt.Errorf("unsupported synchronous recipe")
		}
		if e.Recipe.RewardConfig != "scripts/scenarios/combat-reward-recoil-v5.json" {
			return fmt.Errorf("unsupported reward recipe")
		}
		switch e.Recipe.Loadout {
		case "blaster":
		case "machinegun", "weapons", "weapons-scarce":
			if seen["rules"] {
				return fmt.Errorf("fixed MG/multiweapon teacher is unsupported by current harness")
			}
		default:
			return fmt.Errorf("unsupported loadout")
		}
	case "campaign-comparison":
		if e.PPOTrainable || e.Recipe.Loadout != "" || e.Recipe.Mixed || e.Recipe.RewardConfig != "" {
			return fmt.Errorf("campaign captures are evaluation only until on-policy export is implemented")
		}
	default:
		return fmt.Errorf("unknown harness binding")
	}
	return nil
}

type Task struct {
	Instances     []Instance `json:"instances,omitempty"`
	Episode       Episode    `json:"episode"`
	EpisodeSHA256 string     `json:"episode_sha256"`
	Split         string     `json:"split"`
	Seeds         []int      `json:"seeds"`
	Modes         []string   `json:"modes"`
	RunnerPath    string     `json:"runner_path"`
	RunnerSHA256  string     `json:"runner_sha256"`
	RewardSHA256  string     `json:"reward_sha256,omitempty"`
}
type Plan struct {
	Version        int    `json:"version"`
	RegistryPath   string `json:"registry_path"`
	RegistrySHA256 string `json:"registry_sha256"`
	ModelPath      string `json:"model_path,omitempty"`
	ModelSHA256    string `json:"model_sha256,omitempty"`
	Tasks          []Task `json:"tasks"`
	Workers        int    `json:"workers"`
	OutputRoot     string `json:"output_root"`
}

func Build(r *Registry, root string, ids []string, split, mode, model, out string, count, offset int) (*Plan, error) {
	if count < 4 || count > 80 || count%4 != 0 || offset < 0 {
		return nil, fmt.Errorf("count must be a multiple of four in 4..80; nonnegative offset")
	}
	p := &Plan{Version: 1, RegistryPath: r.Path, RegistrySHA256: r.SHA256, Workers: 4}
	var e error
	p.OutputRoot, e = filepath.Abs(out)
	if e != nil {
		return nil, e
	}
	wanted := map[string]bool{}
	for _, id := range ids {
		if wanted[id] {
			return nil, fmt.Errorf("duplicate episode selection")
		}
		wanted[id] = true
	}
	if len(wanted) == 0 {
		return nil, fmt.Errorf("select episodes explicitly")
	}
	needsModel := false
	for _, ep := range r.Episodes {
		if !wanted[ep.ID] {
			continue
		}
		delete(wanted, ep.ID)
		if ep.Status != "runnable" {
			return nil, fmt.Errorf("episode %s is planned, not runnable: %v", ep.ID, ep.Missing)
		}
		s, ok := ep.Splits[split]
		if !ok || count > s.Count || offset > s.Count-count {
			return nil, fmt.Errorf("selection exceeds %s cohort of %s", split, ep.ID)
		}
		if split == "train" && !ep.PPOTrainable {
			return nil, fmt.Errorf("%s cannot produce on-policy training data yet", ep.ID)
		}
		modes := []string{mode}
		if mode == "both" {
			modes = []string{"rules", "learned"}
		}
		for _, m := range modes {
			found := false
			for _, allowed := range ep.Modes {
				if allowed == m {
					found = true
				}
			}
			if !found {
				return nil, fmt.Errorf("%s does not support %s", ep.ID, m)
			}
			needsModel = needsModel || m == "learned"
		}
		t := Task{Episode: ep, EpisodeSHA256: r.Hashes[ep.ID], Split: split, Modes: modes}
		for i := 0; i < count; i++ {
			t.Seeds = append(t.Seeds, s.Start+offset+i)
		}
		if ep.Generator != nil {
			world, err := loadGenerationWorld(root)
			if err != nil {
				return nil, err
			}
			for _, seed := range t.Seeds {
				v, err := generate(ep, split, seed, &world)
				if err != nil {
					return nil, err
				}
				t.Instances = append(t.Instances, v)
			}
		}
		script := "run_learned_combat_baseline.ps1"
		if ep.Recipe.Runner == "campaign-comparison" {
			script = "run_learned_campaign_comparison.ps1"
		}
		t.RunnerPath = filepath.Join(root, "scripts", script)
		t.RunnerSHA256, e = Hash(t.RunnerPath)
		if e != nil {
			return nil, e
		}
		if ep.Recipe.RewardConfig != "" {
			t.RewardSHA256, e = Hash(filepath.Join(root, filepath.FromSlash(ep.Recipe.RewardConfig)))
			if e != nil {
				return nil, e
			}
		}
		p.Tasks = append(p.Tasks, t)
	}
	if len(wanted) > 0 {
		return nil, fmt.Errorf("unknown episode selection: %v", wanted)
	}
	if needsModel {
		if model == "" {
			return nil, fmt.Errorf("learned runs require a model")
		}
		p.ModelPath, e = filepath.Abs(model)
		if e != nil {
			return nil, e
		}
		p.ModelSHA256, e = Hash(p.ModelPath)
		if e != nil {
			return nil, e
		}
		var header struct {
			Kind          string `json:"kind"`
			WeaponHead    string `json:"weapon_head"`
			Deterministic bool   `json:"deterministic"`
		}
		b, e := os.ReadFile(p.ModelPath)
		if e != nil {
			return nil, e
		}
		if e = json.Unmarshal(b, &header); e != nil {
			return nil, e
		}
		if header.Kind != "combat_ppo_v1" {
			return nil, fmt.Errorf("current registered learned adapters require PPO weights")
		}
		if split == "train" && header.Deterministic {
			return nil, fmt.Errorf("on-policy training requires stochastic weights")
		}
		if _, e = policy.LoadPPO(p.ModelPath); e != nil {
			return nil, fmt.Errorf("model cannot run through the current Go inference adapter: %w", e)
		}
		for _, t := range p.Tasks {
			if (t.Episode.Recipe.Loadout == "weapons" || t.Episode.Recipe.Loadout == "weapons-scarce") && header.WeaponHead != "combat_masked_weapon_v1" {
				return nil, fmt.Errorf("%s requires the masked weapon head", t.Episode.ID)
			}
		}
	} else if model != "" {
		return nil, fmt.Errorf("rules-only plan does not accept a model")
	}
	return p, nil
}
