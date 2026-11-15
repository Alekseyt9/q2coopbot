// Export client-only features and actually applied primitives for world modelling.
package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"strings"
)

type row struct {
	Source   string    `json:"source_sha256,omitempty"`
	Episode  string    `json:"episode"`
	Frame    int       `json:"frame"`
	Features []float64 `json:"features"`
	Next     []float64 `json:"next_features"`
	Action   []float64 `json:"action"`
	Utility  float64   `json:"utility"`
}

// Observable training surrogate only: health and visible-enemy count. This is
// not a kill, victory or damage-attribution label; occlusion can change it.
func utility(o policy.Observation) float64 { return float64(o.Health)/100 - .1*float64(len(o.Enemies)) }
func run() error {
	root := flag.String("root", "", "capture root containing native dataset/steps.jsonl files")
	out := flag.String("out", "", "fresh JSONL output")
	guardedWorld := flag.Bool("guarded-world", false, "offline world data: retain guarded actually executed commands, never expert BC labels")
	terminalWorld := flag.Bool("terminal-world", false, "offline world data: retain verified same-life terminal next observations")
	flag.Parse()
	if *root == "" || *out == "" {
		return fmt.Errorf("--root and --out required")
	}
	var paths []string
	err := filepath.WalkDir(*root, func(p string, d os.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if !d.IsDir() && d.Name() == "steps.jsonl" {
			paths = append(paths, p)
		}
		return nil
	})
	if err != nil {
		return err
	}
	if len(paths) == 0 {
		return fmt.Errorf("no native steps")
	}
	f, e := os.OpenFile(*out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if e != nil {
		return e
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	sources := map[string]string{}
	count := 0
	for _, p := range paths {
		var report struct {
			Proof struct {
				Accepted bool `json:"accepted"`
			} `json:"command_proof"`
			Sync  bool `json:"synchronous_confirmed"`
			Reset bool `json:"observed_reset_confirmed"`
		}
		b, e := os.ReadFile(filepath.Join(filepath.Dir(p), "report.json"))
		if e != nil {
			return e
		}
		if e = json.Unmarshal(b, &report); e != nil {
			return e
		}
		if !report.Proof.Accepted || !report.Sync || !report.Reset {
			return fmt.Errorf("unverified native dataset: %s", p)
		}
		b, e = os.ReadFile(p)
		if e != nil {
			return e
		}
		h := sha256.Sum256(b)
		sources[p] = hex.EncodeToString(h[:])
		s := bufio.NewScanner(strings.NewReader(string(b)))
		s.Buffer(make([]byte, 65536), 4*1024*1024)
		for s.Scan() {
			var step learningenv.Step
			if e = json.Unmarshal(s.Bytes(), &step); e != nil {
				return e
			}
			o := step.Observation
			n := step.Next
			if step.Version != learningenv.StepVersion || step.Terminal && !*terminalWorld || step.Truncated || n == nil || len(step.Interventions) > 0 && !*guardedWorld || o.Health <= 0 || n.Health <= 0 && !*terminalWorld || o.AgeMS < 0 || o.AgeMS > 300 || n.AgeMS < 0 || n.AgeMS > 300 || !policy.SameLife(o.Identity, n.Identity) || n.Identity.Frame != o.Identity.Frame+1 || step.Owner != "provider" && step.Owner != "rules" {
				continue
			}
			if (*guardedWorld || *terminalWorld) && (step.Native == nil || step.Execution == nil || !step.Execution.Matched || !step.Execution.WindowExclusive) {
				continue
			}
			a := step.AppliedAction
			if a.Weapon != "" {
				continue
			}
			if _, e = policy.Command(o, a, [3]int16{}); e != nil {
				return e
			}
			x, e := policy.FeaturesForVersion(o, policy.WeaponFeatureVersion)
			if e != nil {
				return e
			}
			nx, e := worldNextFeatures(*n, *terminalWorld)
			if e != nil {
				return e
			}
			av := []float64{a.Forward, a.Side, a.YawDelta / 180, a.PitchDelta / 180, 0, 0, 0, 0}
			if a.Attack {
				av[4] = 1
			}
			switch a.Vertical {
			case "release":
				av[5] = 1
			case "jump":
				av[6] = 1
			case "crouch":
				av[7] = 1
			}
			id := o.Identity
			id.Frame = 0
			identity, _ := json.Marshal(id)
			r := row{Source: sources[p], Episode: step.Episode + ":" + string(identity), Frame: o.Identity.Frame, Features: x, Next: nx, Action: av, Utility: utility(o)}
			if e = enc.Encode(r); e != nil {
				return e
			}
			count++
		}
		if e = s.Err(); e != nil {
			return e
		}
	}
	if count == 0 {
		return fmt.Errorf("no eligible transitions")
	}
	if e = f.Sync(); e != nil {
		return e
	}
	meta := struct {
		Version       string            `json:"version"`
		Features      string            `json:"feature_version"`
		Rows          int               `json:"rows"`
		Sources       map[string]string `json:"sources_sha256"`
		Scope         string            `json:"scope"`
		GuardedWorld  bool              `json:"guarded_world"`
		TerminalWorld bool              `json:"terminal_world"`
	}{"hjepa_data_v1", policy.WeaponFeatureVersion, count, sources, "Native verified applied actions, contiguous same-life frames. Opt-in guarded/terminal rows are offline world dynamics, not expert labels or PPO. Postmortem inventory is masked unavailable. Visibility utility is a legacy proxy until native relabelling.", *guardedWorld, *terminalWorld}
	b, _ := json.MarshalIndent(meta, "", "  ")
	if e = os.WriteFile(*out+".meta.json", b, 0600); e != nil {
		return e
	}
	fmt.Printf("Exported %d transitions from %d datasets\n", count, len(paths))
	return nil
}
func worldNextFeatures(o policy.Observation, terminal bool) ([]float64, error) {
	if terminal && o.Health <= 0 {
		// Stock death cleanup may expose invalid dropped inventory. Only the world
		// target masks it as unavailable; the native observation stays untouched.
		o.Inventory = nil
		o.InventoryAgeFrames = nil
	}
	return policy.FeaturesForVersion(o, policy.WeaponFeatureVersion)
}
func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
