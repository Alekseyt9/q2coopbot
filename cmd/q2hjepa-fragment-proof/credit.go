package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"q2coopbot/internal/harness/checkpoint"
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
)

// Establish terminal returns with a longer continuation of the SAME sampled
// stream. A control transfer or horizon alone never supplies a zero tail value.
func fragmentCredit(prefixPath, tailPath, out string) (map[string]any, error) {
	var prefix, tail learningenv.Fragment
	if err := read(prefixPath, &prefix); err != nil {
		return nil, err
	}
	if err := prefix.Validate(); err != nil {
		return nil, err
	}
	if err := read(tailPath, &tail); err != nil {
		return nil, err
	}
	if err := tail.Validate(); err != nil {
		return nil, err
	}
	if prefix.Split != "train" || tail.Split != "train" || prefix.Family != tail.Family || prefix.Model != tail.Model || prefix.Sampling != tail.Sampling || len(prefix.Rows) > len(tail.Rows) {
		return nil, fmt.Errorf("paired unchanged TRAIN prefix/tail required")
	}
	inputs := map[string]string{}
	for _, f := range []learningenv.Fragment{prefix, tail} {
		for path, want := range f.SourceSHA256 {
			got, err := digest(path)
			if err != nil || got != want {
				return nil, fmt.Errorf("changed fragment source %s", path)
			}
			inputs[path] = want
		}
	}
	for i, a := range prefix.Rows {
		b := tail.Rows[i]
		if len(a.Events) != len(b.Events) {
			return nil, fmt.Errorf("longer tail native effects differ at%d", i)
		}
		a.Identity = b.Identity
		a.NextIdentity = b.NextIdentity
		a.Action.Identity = b.Action.Identity
		a.Native.Spawncount = b.Native.Spawncount
		// Packet sequences and ledger indexes are process-local receipts,
		// already verified against each capture's native log by the exporter.
		a.Native.Sequence = b.Native.Sequence
		a.Native.DamageIndexes = b.Native.DamageIndexes
		// End markers describe capture length; the longer stream has the same
		// transition without the shorter capture's horizon marker.
		a.Terminal, a.Truncated, a.BootstrapRequired, a.Boundary = b.Terminal, b.Truncated, b.BootstrapRequired, b.Boundary
		for j := range a.Events {
			a.Events[j].Spawncount = b.Events[j].Spawncount
		}
		if !reflect.DeepEqual(a, b) {
			return nil, fmt.Errorf("longer tail does not reproduce prefix transition%d", i)
		}
	}
	var source struct {
		Start struct {
			Confirmed bool `json:"confirmed"`
			Entities  []struct {
				ID     int    `json:"id"`
				Class  string `json:"class"`
				Health int    `json:"health"`
			} `json:"entities"`
		} `json:"generated_start"`
	}
	reportPath := filepath.Join(prefix.SourceDirectory, "report.json")
	if err := read(reportPath, &source); err != nil {
		return nil, err
	}
	if !source.Start.Confirmed {
		return nil, fmt.Errorf("native source roster missing")
	}
	alive := map[int]bool{}
	totalHP := 0
	rosterClosureSafe := true
	for _, e := range source.Start.Entities {
		if e.Class == "player" {
			continue
		}
		switch e.Class {
		case "monster_soldier", "monster_soldier_light", "monster_soldier_ss", "monster_infantry", "monster_parasite", "monster_gunner", "monster_berserk", "monster_brain":
		default:
			// A medic or an unsupported lifecycle prevents a roster-clear goal
			// proof, but does not invalidate a native actor-death terminal.
			rosterClosureSafe = false
		}
		if e.ID <= 0 || e.Health <= 0 || alive[e.ID] {
			return nil, fmt.Errorf("invalid native source roster")
		}
		alive[e.ID] = true
		totalHP += e.Health
	}
	if len(alive) == 0 {
		return nil, fmt.Errorf("empty native goal roster")
	}
	var manifest checkpoint.Manifest
	if err := read(filepath.Join(prefix.SourceDirectory, "combat-checkpoint", "manifest.json"), &manifest); err != nil {
		return nil, err
	}
	if manifest.Barrier == nil {
		return nil, fmt.Errorf("source checkpoint root missing")
	}
	logPath := filepath.Join(prefix.SourceDirectory, "server.log")
	log, err := os.ReadFile(logPath)
	if err != nil {
		return nil, err
	}
	events, err := learningenv.ReadDamageEvents(strings.NewReader(string(log)))
	if err != nil {
		return nil, err
	}
	native, err := learningenv.ReadNativeSteps(strings.NewReader(string(log)), events)
	if err != nil {
		return nil, err
	}
	for _, s := range native.Steps {
		if s.Spawncount == manifest.Barrier.Generation && s.EndFrame <= manifest.Barrier.Frame {
			for _, index := range s.DamageIndexes {
				e := events[index]
				if e.HealthAfter <= 0 {
					delete(alive, e.Target)
				}
			}
		}
	}
	tailLogPath := filepath.Join(tail.BranchDirectory, "server.log")
	if _, bound := tail.SourceSHA256[tailLogPath]; !bound {
		return nil, fmt.Errorf("native tail log is not source-bound")
	}
	tailLog, err := os.ReadFile(tailLogPath)
	if err != nil {
		return nil, err
	}
	objectives, err := fragmentObjectiveStates(string(log), string(tailLog), policy.Identity{Map: manifest.Map, Spawncount: manifest.Barrier.Generation, Frame: manifest.Barrier.Frame}, tail.Rows)
	if err != nil {
		return nil, err
	}
	if len(alive) == 0 && objectives == nil && rosterClosureSafe {
		return nil, fmt.Errorf("goal already complete before fragment root")
	}
	rewards, terminal, reason, err := nativeTerminalRewardsWithObjectives(tail.Rows, alive, totalHP, rosterClosureSafe, objectives)
	if err != nil {
		return nil, err
	}
	if terminal < len(prefix.Rows)-1 {
		return nil, fmt.Errorf("prefix crosses native terminal; recut before optimizer admission")
	}
	suffix := make([]float64, terminal+1)
	running := 0.
	for i := terminal; i >= 0; i-- {
		running += rewards[i]
		suffix[i] = running
	}
	labels := make([]map[string]any, len(prefix.Rows))
	for i := range labels {
		labels[i] = map[string]any{"index": i, "return_to_go": suffix[i]}
	}
	for _, path := range []string{prefixPath, tailPath, reportPath, logPath} {
		h, err := digest(path)
		if err != nil {
			return nil, err
		}
		inputs[path] = h
	}
	result := map[string]any{"version": "hjepa_fragment_mc_targets_v1", "optimizer_ready": true, "checkpoint_family": prefix.Family, "model_version": prefix.Model, "sampling_stream": prefix.Sampling.Stream, "prefix": prefixPath, "tail": tailPath, "reward": "100goal-100death+10capped_progress-.1received-self-.5team_v1", "terminal_reason": reason, "terminal_step": terminal, "terminal_next_identity": tail.Rows[terminal].NextIdentity, "source_initial_monster_hp": totalHP, "bootstrap_value": 0, "bootstrap_proof": "native_terminal_in_same_stream_continuation", "targets": labels, "sources_sha256": inputs, "scope": "Conditional native Monte Carlo targets for the short prefix; no optimizer update or quality claim"}
	if reason == "native_objective_clear" {
		result["terminal_native_objective"] = objectives[terminal]
	}
	if prefix.Sampling.Scope != "" {
		result["version"] = "hjepa_fragment_mc_targets_v2"
		if prefix.Sampling.Scope == policy.HJEPASamplingRootFireScope {
			result["version"] = "hjepa_fragment_mc_targets_v3"
		} else if prefix.Sampling.Scope == policy.HJEPASamplingRootJointScope {
			result["version"] = "hjepa_fragment_mc_targets_v4"
		}
		result["sampling_scope"] = prefix.Sampling.Scope
	}
	for path, want := range inputs {
		got, err := digest(path)
		if err != nil || got != want {
			return nil, fmt.Errorf("credit source changed before publication: %s", path)
		}
	}
	file, err := os.OpenFile(out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return nil, err
	}
	err = json.NewEncoder(file).Encode(result)
	closed := file.Close()
	if err != nil {
		return nil, err
	}
	if closed != nil {
		return nil, closed
	}
	return map[string]any{"accepted": true, "version": result["version"], "optimizer_ready": true, "rows": len(labels), "terminal_reason": reason, "terminal_step": terminal, "prefix_root_return": suffix[0], "output": out}, nil
}

// nativeTerminalRewards never treats a horizon or controller transfer as a terminal.
func nativeTerminalRewards(rows []learningenv.FragmentTransition, initialAlive map[int]bool, totalHP int, rosterClosureSafe bool) ([]float64, int, string, error) {
	return nativeTerminalRewardsWithObjectives(rows, initialAlive, totalHP, rosterClosureSafe, nil)
}

func nativeTerminalRewardsWithObjectives(rows []learningenv.FragmentTransition, initialAlive map[int]bool, totalHP int, rosterClosureSafe bool, objectives map[int]learningenv.CombatObjective) ([]float64, int, string, error) {
	if objectives != nil {
		for i, r := range rows {
			s, ok := objectives[i]
			if !ok || s.Map != r.NextIdentity.Map || s.Spawncount != r.NextIdentity.Spawncount || s.Frame != r.NextIdentity.Frame || s.Alive < 0 || s.Pending < 0 {
				return nil, -1, "", fmt.Errorf("objective is not joined to next identity at %d", i)
			}
		}
	}
	alive := make(map[int]bool, len(initialAlive))
	for id, living := range initialAlive {
		alive[id] = living
	}
	rewards := make([]float64, len(rows))
	terminal := -1
	reason := ""
	damage := 0.
	for i, r := range rows {
		dead := false
		for _, e := range r.Events {
			loss := float64(max(0, min(max(0, e.HealthBefore), e.Take)))
			if e.Target == r.Identity.Actor {
				rewards[i] -= .1 * loss
				if e.Attacker == r.Identity.Actor {
					rewards[i] -= loss
				}
				if e.HealthBefore > 0 && e.HealthAfter <= 0 {
					dead = true
				}
			}
			if e.Attacker == r.Identity.Actor && e.Target != r.Identity.Actor {
				if strings.HasPrefix(e.TargetClass, "monster_") {
					old := min(1., damage/float64(totalHP))
					damage += loss
					rewards[i] += 10 * (min(1., damage/float64(totalHP)) - old)
				} else if e.TargetClass == "player" {
					rewards[i] -= .5 * loss
				}
			}
			if e.HealthAfter <= 0 {
				delete(alive, e.Target)
			}
		}
		if dead {
			if r.NextHealth > 0 {
				return nil, -1, "", fmt.Errorf("native death lacks next dead observation")
			}
			terminal, reason = i, "native_death"
			rewards[i] -= 100
			break
		}
		if objectives != nil && objectives[i].Alive == 0 && objectives[i].Pending == 0 {
			if r.NextHealth <= 0 {
				return nil, -1, "", fmt.Errorf("native objective goal lacks living next observation")
			}
			terminal, reason = i, "native_objective_clear"
			rewards[i] += 100
			break
		}
		if objectives == nil && rosterClosureSafe && len(alive) == 0 {
			terminal, reason = i, "native_source_roster_clear"
			rewards[i] += 100
			break
		}
	}
	if terminal < 0 {
		return nil, -1, "", fmt.Errorf("unresolved continuation: no native terminal, bootstrap still required")
	}
	return rewards, terminal, reason, nil
}
