// Verify native save/load continuations before admitting fragment data.
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
	"reflect"
	"regexp"
	"strconv"
	"strings"

	"q2coopbot/internal/harness/checkpoint"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

type row struct {
	Frame      int             `json:"frame"`
	Generation int             `json:"spawncount"`
	Sequence   int             `json:"client_sequence"`
	Actor      int             `json:"self_entity"`
	Command    quake.UserCmd   `json:"sent_command"`
	Combat     *policy.Capture `json:"combat_policy"`
}

func lines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 65536), 4<<20)
	var result []string
	for s.Scan() {
		result = append(result, s.Text())
	}
	return result, s.Err()
}
func rows(path string) (map[int]row, error) {
	data, err := lines(path)
	if err != nil {
		return nil, err
	}
	result := map[int]row{}
	for _, line := range data {
		var r row
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			return nil, err
		}
		if _, ok := result[r.Frame]; ok {
			return nil, fmt.Errorf("duplicate trace frame %d", r.Frame)
		}
		result[r.Frame] = r
	}
	return result, nil
}
func read(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}
func digest(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:]), nil
}

var kv = regexp.MustCompile(`([a-z]+)=(-?[0-9]+)`)

func values(line string) map[string]int {
	r := map[string]int{}
	for _, m := range kv.FindAllStringSubmatch(line, -1) {
		r[m[1]], _ = strconv.Atoi(m[2])
	}
	return r
}
func applied(log []string, r row) error {
	count, begin, end := 0, 0, 0
	for _, line := range log {
		if !strings.HasPrefix(line, "sv_test_applied_cmd ") && !strings.HasPrefix(line, "sv_test_step ") {
			continue
		}
		v := values(line)
		if v["spawncount"] != r.Generation {
			continue
		}
		if strings.HasPrefix(line, "sv_test_applied_cmd ") && v["frame"] == r.Frame {
			count++
			for _, key := range []string{"spawncount", "frame", "seq", "pitch", "yaw", "roll", "forward", "side", "up", "buttons", "impulse", "msec", "light"} {
				if _, ok := v[key]; !ok {
					return fmt.Errorf("native command field missing: %s", key)
				}
			}
			cmd := quake.UserCmd{Pitch: int16(v["pitch"]), Yaw: int16(v["yaw"]), Roll: int16(v["roll"]), Forward: int16(v["forward"]), Side: int16(v["side"]), Up: int16(v["up"]), Buttons: byte(v["buttons"]), Impulse: byte(v["impulse"]), Msec: byte(v["msec"]), Light: byte(v["light"])}
			if v["seq"] != r.Sequence || !strings.Contains(line, " kind=new ") || cmd != r.Command {
				return fmt.Errorf("native command differs at %d", r.Frame)
			}
		}
		if strings.HasPrefix(line, "sv_test_step ") && v["seq"] == r.Sequence && v["actor"] == r.Actor {
			if strings.Contains(line, " phase=begin ") && v["frame"] == r.Frame {
				begin++
			}
			if strings.Contains(line, " phase=end ") && v["frame"] == r.Frame+1 {
				end++
			}
		}
	}
	if count != 1 || begin != 1 || end != 1 {
		return fmt.Errorf("native single-step proof missing/duplicated at %d (%d,%d,%d)", r.Frame, count, begin, end)
	}
	return nil
}
func effects(log []string, generation, start, n int) []string {
	var result []string
	current := -1
	for _, line := range log {
		if strings.HasPrefix(line, "sv_test_step ") && strings.Contains(line, " phase=begin ") {
			v := values(line)
			if v["spawncount"] == generation {
				current = v["frame"]
			} else {
				current = -1
			}
		}
		if current < start || current >= start+n {
			continue
		}
		for _, kind := range []string{"damage", "hitscan", "sg", "projectile", "enemy_bolt", "aim_pose"} {
			at := strings.Index(line, "g_test_"+kind+" version=")
			if at >= 0 {
				result = append(result, fmt.Sprintf("%d:%s", current-start, line[at:]))
				break
			}
		}
	}
	return result
}
func verify(source, branch string, n int) (map[string]any, error) {
	var manifest checkpoint.Manifest
	if err := read(filepath.Join(source, "combat-checkpoint", "manifest.json"), &manifest); err != nil {
		return nil, err
	}
	if manifest.Barrier == nil || manifest.Barrier.Boundary != checkpoint.CombatPreDecisionBoundary || len(manifest.Barrier.Participants) != 1 {
		return nil, fmt.Errorf("single combat source checkpoint required")
	}
	name := strings.TrimSuffix(manifest.Barrier.Participants[0].Name, ".json")
	capture, err := checkpoint.ReadParticipant(filepath.Join(source, "combat-checkpoint"), name)
	if err != nil {
		return nil, err
	}
	var restore checkpoint.RestoreReceipt
	if err := read(filepath.Join(branch, "control.json.restored.json"), &restore); err != nil {
		return nil, err
	}
	var load, release checkpoint.Result
	if err := read(filepath.Join(branch, "load-result.json"), &load); err != nil {
		return nil, err
	}
	if err := read(filepath.Join(branch, "release-result.json"), &release); err != nil {
		return nil, err
	}
	if !load.RNGRestored || !load.CombatRestored || !load.WorldUnadvanced || load.LoadAnchor == nil || load.State != "load_held" || release.State != "release_verified" || !release.BarrierVerified || restore.Combat == nil ||
		restore.SourceID != capture.ID || restore.SourceFrame != capture.Frame || restore.Mode != "combat" || restore.Frame != 1 || restore.Generation == capture.Generation || restore.Combat.Root.Frame != 1 || restore.Combat.PrefixFrame != 0 {
		return nil, fmt.Errorf("native load/release/controller binding missing")
	}
	var state struct {
		Root     policy.Identity                `json:"decision_root"`
		Sampling policy.HJEPASamplingCheckpoint `json:"sampling"`
	}
	if err := json.Unmarshal(capture.Combat, &state); err != nil {
		return nil, err
	}
	rootActionEquivalent := (restore.Combat.SamplingScope == policy.HJEPASamplingRootFireScope || restore.Combat.SamplingScope == policy.HJEPASamplingRootJointScope) && state.Sampling.Scope == "" && state.Sampling.Stream == 0
	if (restore.Combat.SamplingStream != state.Sampling.Stream || restore.Combat.SamplingScope != state.Sampling.Scope) && !rootActionEquivalent {
		return nil, fmt.Errorf("different draw streams require fragment evaluation, not exact continuation proof")
	}
	if load.LoadAnchor.ID != capture.ID || load.LoadAnchor.Boundary != capture.Boundary || load.LoadAnchor.Map != capture.Map ||
		load.LoadAnchor.Frame != 1 || load.LoadAnchor.Generation != restore.Generation || restore.Participant != capture.Participant || restore.SelfEntity != capture.SelfEntity ||
		restore.Combat.Source != state.Root || restore.Combat.Model != state.Sampling.Model || restore.Combat.Root.Spawncount != restore.Generation || restore.Combat.Root.Actor != capture.SelfEntity ||
		restore.Combat.Root.Map != capture.Map || restore.Combat.Root.Life != state.Root.Life {
		return nil, fmt.Errorf("source package, loaded clock and controller lineage differ")
	}
	a, err := rows(filepath.Join(source, "bot.jsonl"))
	if err != nil {
		return nil, err
	}
	b, err := rows(filepath.Join(branch, "bot.jsonl"))
	if err != nil {
		return nil, err
	}
	la, err := lines(filepath.Join(source, "server.log"))
	if err != nil {
		return nil, err
	}
	lb, err := lines(filepath.Join(branch, "server.log"))
	if err != nil {
		return nil, err
	}
	width := 0
	for i := 0; i < n; i++ {
		x, ok := a[capture.Frame+i]
		if !ok {
			return nil, fmt.Errorf("source row missing %d", i)
		}
		y, ok := b[1+i]
		if !ok {
			return nil, fmt.Errorf("branch row missing %d", i)
		}
		if x.Generation != capture.Generation || y.Generation != restore.Generation || x.Actor != capture.SelfEntity || y.Actor != x.Actor || x.Combat == nil || y.Combat == nil {
			return nil, fmt.Errorf("trace native binding mismatch %d", i)
		}
		wantSource, wantNative := state.Root, restore.Combat.Root
		wantSource.Frame += i
		wantNative.Frame += i
		if x.Combat.Observation.Identity != wantSource || y.Combat.Observation.Identity != wantNative {
			return nil, fmt.Errorf("observation identity differs from native root at step%d", i)
		}
		sx, sy := x.Combat.Selection, y.Combat.Selection
		if sx == nil || sy == nil || sx.Owner != "provider" || sy.Owner != "provider" || sx.Candidate == nil || sy.Candidate == nil || sx.Sample == nil || sy.Sample == nil || sx.ProviderVersion != restore.Combat.Model || sy.ProviderVersion != sx.ProviderVersion {
			return nil, fmt.Errorf("sampled same-model provider ownership missing %d", i)
		}
		fx, err := policy.HJEPAFeatures(x.Combat.Observation, policy.HJEPAContextFeatureVersion)
		if err != nil {
			return nil, err
		}
		fy, err := policy.HJEPAFeatures(y.Combat.Observation, policy.HJEPAContextFeatureVersion)
		if err != nil {
			return nil, err
		}
		width = len(fx)
		if !reflect.DeepEqual(fx, fy) {
			for j := range fx {
				if fx[j] != fy[j] {
					return nil, fmt.Errorf("feature differs step%d index%d source%.17g branch%.17g", i, j, fx[j], fy[j])
				}
			}
			return nil, fmt.Errorf("feature shape differs")
		}
		want, got := *sx.Candidate, *sy.Candidate
		want.Identity = got.Identity
		if want != got || !reflect.DeepEqual(sx.Sample, sy.Sample) || !reflect.DeepEqual(sx.Interventions, sy.Interventions) || x.Command != y.Command {
			return nil, fmt.Errorf("proposal/sample/guarded command differs step%d", i)
		}
		if err := applied(la, x); err != nil {
			return nil, err
		}
		if err := applied(lb, y); err != nil {
			return nil, err
		}
	}
	ea, eb := effects(la, capture.Generation, capture.Frame, n), effects(lb, restore.Generation, 1, n)
	if !reflect.DeepEqual(ea, eb) {
		return nil, fmt.Errorf("native effects differ (%d source, %d loaded)", len(ea), len(eb))
	}
	hashes := map[string]string{}
	for _, p := range []string{filepath.Join(source, "combat-checkpoint", "manifest.json"), filepath.Join(source, "combat-checkpoint", "sidecar", name+".json"), filepath.Join(source, "bot.jsonl"), filepath.Join(source, "server.log"), filepath.Join(branch, "bot.jsonl"), filepath.Join(branch, "server.log"), filepath.Join(branch, "load-result.json"), filepath.Join(branch, "release-result.json"), filepath.Join(branch, "control.json.restored.json")} {
		h, err := digest(p)
		if err != nil {
			return nil, err
		}
		hashes[p] = h
	}
	return map[string]any{"accepted": true, "proof": "native_hjepa_predecision_continuation_v1", "decisions_compared": n, "feature_width": width, "features_compared": width * n, "native_single_steps_verified": 2 * n, "native_effect_records_compared": len(ea), "model": restore.Combat.Model, "source_frame": capture.Frame, "loaded_frame": 1, "input_sha256": hashes, "fragment_training": false, "promotion": false, "scope": "Matching H-JEPA features, proposals, samples, guarded/native commands and logged gameplay effects for this native continuation only"}, nil
}
func main() {
	source := flag.String("source", "", "completed source directory")
	branch := flag.String("branch", "", "completed loaded branch directory")
	n := flag.Int("frames", 30, "decisions to compare")
	other := flag.String("other-branch", "", "compare two explicitly configured fragment sampling streams")
	same := flag.Bool("same-stream", false, "require a repeat of the same fragment stream")
	rootOnly := flag.Bool("root-only", false, "compare original source and loaded root features for any sampling stream")
	rootStreams := flag.Bool("root-fire-streams", false, "select two sampled root fire actions; action-stratified, not IID training data")
	export := flag.String("export-fragment", "", "fresh typed fragment JSON; diagnostic unless a verified TRAIN plan is supplied")
	plan := flag.String("plan", "", "condition plan required for TRAIN fragment admission")
	repo := flag.String("repo", ".", "repository root for condition plan verification")
	creditPrefix := flag.String("credit-prefix", "", "typed TRAIN prefix for native terminal return targets")
	creditTail := flag.String("credit-tail", "", "longer same-stream TRAIN continuation")
	creditOut := flag.String("credit-out", "", "fresh target JSON")
	flag.Parse()
	if *rootStreams {
		r, err := selectRootFireStreams(*source)
		if err != nil {
			_ = json.NewEncoder(os.Stdout).Encode(map[string]any{"accepted": false, "reason": err.Error()})
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		_ = json.NewEncoder(os.Stdout).Encode(r)
		return
	}
	if *creditPrefix != "" || *creditTail != "" || *creditOut != "" {
		if *creditPrefix == "" || *creditTail == "" || *creditOut == "" {
			fmt.Fprintln(os.Stderr, "credit-prefix,credit-tail,credit-out required")
			os.Exit(2)
		}
		r, err := fragmentCredit(*creditPrefix, *creditTail, *creditOut)
		if err != nil {
			_ = json.NewEncoder(os.Stdout).Encode(map[string]any{"accepted": false, "reason": err.Error()})
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		_ = json.NewEncoder(os.Stdout).Encode(r)
		return
	}
	minimumFrames := 2
	if *export != "" || *rootOnly {
		minimumFrames = 1
	}
	if *source == "" || *branch == "" || *n < minimumFrames || *n > 500 {
		fmt.Fprintf(os.Stderr, "source,branch and %d..500 frames required\n", minimumFrames)
		os.Exit(2)
	}
	var r map[string]any
	var err error
	if *rootOnly {
		r, err = verifySourceRoot(*source, *branch)
	} else if *export != "" {
		r, err = exportFragment(*source, *branch, *export, *plan, *repo, *n)
	} else if *other != "" {
		r, err = verifyForkPair(*source, *branch, *other, *n, *same)
	} else {
		r, err = verify(*source, *branch, *n)
	}
	if err != nil {
		_ = json.NewEncoder(os.Stdout).Encode(map[string]any{"accepted": false, "reason": err.Error()})
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	_ = json.NewEncoder(os.Stdout).Encode(r)
}
