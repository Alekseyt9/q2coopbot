package main

import (
	"fmt"
	"strings"

	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
)

// Join supervisor truth to the exact post-command observation. If telemetry is
// enabled, incomplete or differently clocked telemetry cannot fall back to the
// weaker initial-roster test. Legacy captures remain separately admissible.
func fragmentObjectiveStates(sourceLog, tailLog string, root policy.Identity, rows []learningenv.FragmentTransition) (map[int]learningenv.CombatObjective, error) {
	sourceEnabled := strings.Contains(sourceLog, "g_test_objective")
	tailEnabled := strings.Contains(tailLog, "g_test_objective")
	if !sourceEnabled && !tailEnabled {
		return nil, nil
	}
	if !sourceEnabled || !tailEnabled {
		return nil, fmt.Errorf("source/tail objective telemetry mode differs")
	}
	source, err := learningenv.ReadCombatObjectives(strings.NewReader(sourceLog))
	if err != nil {
		return nil, err
	}
	tail, err := learningenv.ReadCombatObjectives(strings.NewReader(tailLog))
	if err != nil {
		return nil, err
	}
	var anchor *learningenv.CombatObjective
	for i := range source {
		s := &source[i]
		if s.Map == root.Map && s.Spawncount == root.Spawncount && s.Frame == root.Frame {
			anchor = s
		}
	}
	if anchor == nil || anchor.Alive+anchor.Pending == 0 {
		return nil, fmt.Errorf("living/pending native objective at source root not established")
	}
	byIdentity := map[policy.Identity]learningenv.CombatObjective{}
	for _, s := range tail {
		byIdentity[policy.Identity{Map: s.Map, Spawncount: s.Spawncount, Frame: s.Frame}] = s
	}
	states := make(map[int]learningenv.CombatObjective, len(rows))
	for i, r := range rows {
		next := r.NextIdentity
		s, ok := byIdentity[policy.Identity{Map: next.Map, Spawncount: next.Spawncount, Frame: next.Frame}]
		if !ok || s.GameFrame != anchor.GameFrame+i+1 {
			return nil, fmt.Errorf("native objective missing or world clock differs at transition %d", i)
		}
		states[i] = s
	}
	return states, nil
}
