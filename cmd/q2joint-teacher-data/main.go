// Export observed-only decision BC features with native dispatch proof.
// This command never loads or executes a neural network.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"q2coopbot/internal/harness"
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

type source struct {
	Root    string `json:"root"`
	Seed    int    `json:"seed"`
	Split   string `json:"split"`
	Release int    `json:"network_release_frame"`
}
type request struct {
	Version string   `json:"version"`
	Cases   []source `json:"cases"`
}
type trace struct {
	Capture  *policy.Capture `json:"combat_policy"`
	Command  quake.UserCmd   `json:"sent_command"`
	Sequence uint32          `json:"client_sequence"`
}
type sample struct {
	Seed            int                  `json:"seed"`
	Identity        policy.Identity      `json:"identity"`
	Features        []float64            `json:"features"`
	Target          int                  `json:"target"`
	Weapon          int                  `json:"weapon"`
	Attack          bool                 `json:"attack"`
	TargetMask      bool                 `json:"target_mask"`
	WeaponMask      bool                 `json:"weapon_mask"`
	AttackMask      bool                 `json:"attack_mask"`
	NativeMatched   bool                 `json:"native_matched"`
	NativeExclusive bool                 `json:"native_exclusive"`
	PreviousTarget  *policy.TargetIntent `json:"adapted_previous_target,omitempty"`
}

func load(path string, v any) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	return json.Unmarshal(b, v)
}
func pastIntent(p *policy.TargetIntent, o policy.Observation) *policy.TargetIntent {
	if p != nil && policy.SameLife(p.Identity, o.Identity) && p.Identity.Frame+1 == o.Identity.Frame {
		return p
	}
	return nil
}

// Learn the teacher's executed fire release, including its explicitly recorded
// burst/risk decision. Never copy a guard-modified turn or movement as fire-only.
func fireCandidate(c policy.Capture) bool {
	if !c.Changed && (c.LimitReason == "" || c.LimitReason == "no_movement_goal") && len(c.Selection.Interventions) == 0 {
		return true
	}
	if c.Applied.Attack || !c.Proposed.Attack || len(c.Selection.Interventions) != 0 {
		return false
	}
	switch c.LimitReason {
	case "machinegun_burst_pause", "barrel_blast_risk", "hand_grenade_requires_safe_throw":
		command := c.ProposedCommand
		command.Buttons &^= 1
		// Client send time stamps duration/light after planner guards. Neither
		// is a learned action; dispatch proof checks the full sent command.
		command.Msec, command.Light = c.AppliedCommand.Msec, c.AppliedCommand.Light
		return command == c.AppliedCommand
	}
	return false
}
func teacherTarget(c policy.Capture) (int, *policy.TargetIntent, error) {
	if c.TeacherAimSource != "enemy" || c.TeacherAimEntity <= 0 {
		return 0, nil, nil
	}
	available, e := policy.TargetAvailability(c.Observation)
	if e != nil {
		return 0, nil, e
	}
	for i, enemy := range policy.TargetEnemies(c.Observation) {
		if enemy.ID == c.TeacherAimEntity && available[i+1] && enemy.Track != nil {
			return i + 1, &policy.TargetIntent{Identity: c.Observation.Identity, Entity: enemy.ID, Track: *enemy.Track}, nil
		}
	}
	return 0, nil, nil
}

func exportCase(s source, enc *json.Encoder) (map[string]int, error) {
	counts := map[string]int{}
	var capture struct {
		Complete bool `json:"capture_complete"`
		Valid    bool `json:"provenance_valid"`
		Results  []struct {
			Root      string `json:"root"`
			Seed      int    `json:"seed"`
			Valid     bool   `json:"capture_valid"`
			Dispatch  bool   `json:"dispatch_valid"`
			Confirmed bool   `json:"seed_confirmed"`
		} `json:"results"`
	}
	if e := load(filepath.Join(s.Root, "report.json"), &capture); e != nil {
		return nil, e
	}
	if !capture.Complete || !capture.Valid || len(capture.Results) != 1 {
		return nil, fmt.Errorf("invalid capture")
	}
	result := capture.Results[0]
	if result.Seed != s.Seed || !result.Valid || !result.Dispatch || !result.Confirmed {
		return nil, fmt.Errorf("invalid seed/dispatch")
	}
	root := result.Root
	log, e := os.Open(filepath.Join(root, "server.log"))
	if e != nil {
		return nil, e
	}
	events, e := learningenv.ReadDamageEvents(log)
	log.Close()
	if e != nil {
		return nil, e
	}
	log, e = os.Open(filepath.Join(root, "server.log"))
	if e != nil {
		return nil, e
	}
	native, e := learningenv.ReadNativeSteps(log, events)
	log.Close()
	if e != nil {
		return nil, e
	}
	if native.Release.Seed != s.Seed || native.Release.Frame != s.Release {
		return nil, fmt.Errorf("native release mismatch")
	}
	commands, e := harness.ReadAppliedCommandsForClient(filepath.Join(root, "server.log"), "SoloRetreatBot")
	if e != nil {
		return nil, e
	}
	execution := learningenv.NewExecutionIndex(commands)
	file, e := os.Open(filepath.Join(root, "bot.jsonl"))
	if e != nil {
		return nil, e
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 4096), 8*1024*1024)
	rows := []trace{}
	for scanner.Scan() {
		var r trace
		if e = json.Unmarshal(scanner.Bytes(), &r); e != nil {
			return nil, e
		}
		rows = append(rows, r)
	}
	if e = scanner.Err(); e != nil {
		return nil, e
	}
	var previous *policy.TargetIntent
	last := policy.Identity{}
	for i, r := range rows {
		if r.Capture == nil {
			return nil, fmt.Errorf("missing raw capture")
		}
		c := *r.Capture
		o := c.Observation
		id := o.Identity
		if id.Spawncount != native.Release.Spawncount || id.Frame < s.Release || id.Life != 1 || o.Health <= 0 {
			previous = nil
			continue
		}
		if policy.SameLife(last, id) && id.Frame <= last.Frame {
			return nil, fmt.Errorf("duplicate/nonmonotonic identity")
		}
		last = id
		if c.Selection == nil || c.Selection.Mode != "rules" || c.Selection.Owner != "rules" || c.Provider != "rules" || c.Selection.Fallback != "" {
			return nil, fmt.Errorf("not exclusive rules teacher")
		}
		if c.Applied.Identity != id || c.AppliedCommand != r.Command || c.ClientSequence != r.Sequence {
			return nil, fmt.Errorf("raw applied mismatch")
		}
		if o.PreviousTarget != nil {
			return nil, fmt.Errorf("unexpected raw previous target")
		}
		step := learningenv.Step{Observation: o, ClientSequence: r.Sequence, Command: r.Command, AppliedAction: c.Applied}
		if i+1 < len(rows) && rows[i+1].Capture != nil {
			next := rows[i+1].Capture.Observation
			if policy.SameLife(id, next.Identity) && next.Identity.Frame == id.Frame+1 {
				step.Next = &next
			}
		}
		proof := execution.Match(&step)
		n, e := native.Match(&step)
		if e != nil {
			return nil, e
		}
		if !proof.Matched || n.BeginFrame != id.Frame {
			return nil, fmt.Errorf("unmatched native command")
		}
		o.PreviousTarget = pastIntent(previous, o)
		target, intent, e := teacherTarget(c)
		if e != nil {
			return nil, e
		}
		previous = intent
		features, e := policy.FeaturesForVersion(o, "combat_features_v10")
		if e != nil {
			return nil, e
		}
		if len(features) != 1147 {
			return nil, fmt.Errorf("wrong feature width")
		}
		out := sample{Seed: s.Seed, Identity: id, Features: features, Target: target, Attack: c.Applied.Attack, NativeMatched: proof.Matched, NativeExclusive: proof.WindowExclusive, PreviousTarget: o.PreviousTarget}
		eligible := proof.WindowExclusive && step.Next != nil && o.AgeMS >= 0 && o.AgeMS <= 300
		out.TargetMask = eligible && target > 0
		// Executed stationary no-movement-goal commands can still teach aim/fire.
		out.AttackMask = out.TargetMask && fireCandidate(c)
		mask, e := policy.WeaponAvailability(o)
		if e != nil {
			return nil, e
		}
		weaponKnown := false
		for j, name := range policy.WeaponChoices() {
			if name == c.Applied.Weapon {
				out.Weapon = j
				weaponKnown = true
				break
			}
		}
		if !weaponKnown {
			return nil, fmt.Errorf("unknown teacher weapon request")
		}
		out.WeaponMask = eligible && mask[out.Weapon] && (target > 0 || out.Weapon > 0)
		if e = enc.Encode(out); e != nil {
			return nil, e
		}
		counts["rows"]++
		if out.TargetMask {
			counts["target"]++
		}
		if out.AttackMask {
			counts["attack"]++
			if out.Attack {
				counts["attack_positive"]++
			} else {
				counts["attack_negative"]++
			}
		}
		if out.WeaponMask {
			counts["weapon"]++
			if out.Weapon > 0 {
				counts["weapon_request"]++
			}
		}
		if o.PreviousTarget != nil {
			counts["adapted_previous_target"]++
		}
		if !eligible {
			counts["context_only_boundary"]++
		}
	}
	return counts, nil
}

func run() error {
	reqPath := flag.String("request", "", "verified closed corpus request")
	out := flag.String("out", "", "fresh output directory")
	flag.Parse()
	var req request
	if e := load(*reqPath, &req); e != nil {
		return e
	}
	if req.Version != "registered_joint_teacher_export_request_v1" || len(req.Cases) == 0 {
		return fmt.Errorf("bad request")
	}
	if e := os.Mkdir(*out, 0755); e != nil {
		return e
	}
	files := map[string]*os.File{}
	enc := map[string]*json.Encoder{}
	totals := map[string]map[string]int{}
	for _, split := range []string{"train", "development"} {
		f, e := os.Create(filepath.Join(*out, split+".jsonl"))
		if e != nil {
			return e
		}
		defer f.Close()
		files[split] = f
		enc[split] = json.NewEncoder(f)
		totals[split] = map[string]int{}
	}
	seen := map[int]bool{}
	for _, s := range req.Cases {
		if seen[s.Seed] || enc[s.Split] == nil {
			return fmt.Errorf("duplicate seed/split")
		}
		seen[s.Seed] = true
		counts, e := exportCase(s, enc[s.Split])
		if e != nil {
			return fmt.Errorf("seed%d: %w", s.Seed, e)
		}
		for k, v := range counts {
			totals[s.Split][k] += v
		}
	}
	for _, split := range []string{"train", "development"} {
		if totals[split]["target"] == 0 || totals[split]["weapon_request"] == 0 || totals[split]["attack_positive"] == 0 || totals[split]["attack_negative"] == 0 {
			return fmt.Errorf("missing decision support in %s: %v", split, totals[split])
		}
		if e := files[split].Sync(); e != nil {
			return e
		}
	}
	f, e := os.Create(filepath.Join(*out, "report.json"))
	if e != nil {
		return e
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(map[string]any{"state": "complete", "version": "joint_teacher_features_v1", "feature_version": "combat_features_v10", "feature_width": 1147, "cases": len(req.Cases), "splits": totals, "neural_execution": false, "training_performed": false, "movement_supervision": false, "vertical_supervision": false, "scope": "Past-only adapted teacher target history; observed Go features; exact native dispatch and exclusive next-frame windows for labels. Boundary observations retained as context. Target/fire/weapon candidates only, no optimality or full BC acceptance."})
}
func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
