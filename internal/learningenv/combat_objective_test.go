package learningenv

import (
	"encoding/json"
	"q2coopbot/internal/policy"
	"strings"
	"testing"
)

func TestNativeObjectiveReceiptNeedsOriginalZeroState(t *testing.T) {
	var g GoalStop
	if err := json.Unmarshal([]byte(`{"version":"combat_goal_stop_v2","reason":"combat_goal_complete","spawncount":42,"actor":1,"classes":["monster_jorg"],"kill_frame":20,"observed_frame":21,"health":80,"kills":[{"frame":20,"target":50,"target_class":"monster_jorg"}],"native_objective":{"spawncount":42,"frame":21,"map":"boss2","game_frame":23,"alive":0,"pending":0}}`), &g); err != nil {
		t.Fatal(err)
	}
	release := &CombatRelease{Spawncount: 42, Frame: 10}
	o := policy.Observation{Identity: policy.Identity{Map: "boss2", Spawncount: 42, Actor: 1, Connection: 1, Life: 1, Frame: 21}, Health: 80}
	events := []DamageEvent{{Map: "boss2", Spawncount: 42, Frame: 20, Attacker: 1, AttackerClass: "player", Target: 50, TargetClass: "monster_jorg", Mod: 4, HealthBefore: 5, HealthAfter: -1}}
	state := *g.NativeObjective
	verify := func(states []CombatObjective) error {
		return VerifyGoalStopWithObjectiveForClasses(g, release, events, o, []string{"monster_jorg"}, "boss2", states)
	}
	if err := verify([]CombatObjective{state}); err != nil {
		t.Fatal(err)
	}
	if VerifyGoalStopForClasses(g, release, events, o, []string{"monster_jorg"}, "boss2") == nil {
		t.Fatal("version2 accepted without independent state")
	}
	for _, mutate := range []func(*CombatObjective){func(s *CombatObjective) { s.Alive = 1 }, func(s *CombatObjective) { s.Pending = 1 }, func(s *CombatObjective) { s.Frame-- }, func(s *CombatObjective) { s.Frame++ }, func(s *CombatObjective) { s.Spawncount++ }, func(s *CombatObjective) { s.Map = "base1" }} {
		x := state
		mutate(&x)
		if verify([]CombatObjective{x}) == nil {
			t.Fatal("forged receipt/source mismatch accepted", x)
		}
		g.NativeObjective = &x
		if verify([]CombatObjective{x}) == nil {
			t.Fatal("living/pending/stale objective accepted", x)
		}
		g.NativeObjective = &state
	}
	if verify(nil) == nil || verify([]CombatObjective{state, state}) == nil {
		t.Fatal("missing or duplicate proof accepted")
	}
}

func TestReadNativeObjectiveRejectsUnboundAndDuplicateState(t *testing.T) {
	line := "sv_test_objective spawncount=42 server_frame=21 g_test_objective version=1 map=boss2 game_frame=23 alive=0 pending=1\n"
	ready := "g_test_objective ready version=1\n"
	states, err := ReadCombatObjectives(strings.NewReader(ready + line))
	if err != nil || len(states) != 1 || states[0].Pending != 1 {
		t.Fatal(states, err)
	}
	for _, bad := range []string{line, ready + line + line, ready + "g_test_objective version=1 map=boss2 game_frame=23 alive=0 pending=0\n", ready + strings.ReplaceAll(line, "alive=0", "alive=-1")} {
		if _, err := ReadCombatObjectives(strings.NewReader(bad)); err == nil {
			t.Fatal("bad objective log accepted", bad)
		}
	}
}
