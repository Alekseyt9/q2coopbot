package learningenv

import (
	"encoding/json"
	"testing"

	"q2coopbot/internal/policy"
)

func TestNativeGoalStopEvidence(t *testing.T) {
	for _, scenario := range []string{"win", "solo", "release_frame", "missing", "corpse", "prop", "death", "setup", "wrong_world", "unclosed", "forged_target", "second_life"} {
		t.Run(scenario, func(t *testing.T) {
			var g GoalStop
			if err := json.Unmarshal([]byte(`{"version":"combat_goal_stop_v1","reason":"combat_goal_complete","spawncount":42,"actor":1,"classes":["monster_parasite","monster_gunner"],"kill_frame":20,"observed_frame":21,"health":56,"kills":[{"frame":15,"target":352,"target_class":"monster_parasite"},{"frame":20,"target":343,"target_class":"monster_gunner"}]}`), &g); err != nil {
				t.Fatal(err)
			}
			o := policy.Observation{Identity: policy.Identity{Map: "base1", Spawncount: 42, Actor: 1, Connection: 1, Life: 1, Frame: 21}, Health: 56}
			release := &CombatRelease{Spawncount: 42, Frame: 10}
			events := []DamageEvent{{Map: "base1", Spawncount: 42, Frame: 15, Attacker: 1, AttackerClass: "player", Target: 352, TargetClass: "monster_parasite", Mod: 4, HealthBefore: 7, HealthAfter: -1}, {Map: "base1", Spawncount: 42, Frame: 20, Attacker: 1, AttackerClass: "player", Target: 343, TargetClass: "monster_gunner", Mod: 4, HealthBefore: 7, HealthAfter: -1}}
			mixed := true
			switch scenario {
			case "solo":
				mixed = false
				g.Classes = g.Classes[:1]
				g.Kills = g.Kills[:1]
				g.KillFrame = 15
			case "release_frame":
				mixed = false
				g.Classes = g.Classes[:1]
				g.Kills = g.Kills[:1]
				g.KillFrame = 15
				release.Frame = 15
			case "missing":
				events = events[:1]
			case "corpse":
				events[1].HealthBefore = -1
			case "prop":
				events[1].TargetClass = "func_explosive"
			case "death":
				events = append(events, DamageEvent{Map: "base1", Spawncount: 42, Frame: 18, Target: 1, TargetClass: "player", HealthBefore: 7, HealthAfter: -1})
			case "setup":
				release.Frame = 19
			case "wrong_world":
				events[1].Spawncount = 41
			case "unclosed":
				g.ObservedFrame = 20
				o.Identity.Frame = 20
			case "forged_target":
				g.Kills[1].Target = 999
			case "second_life":
				o.Identity.Life = 2
			}
			err := VerifyGoalStop(g, release, events, o, mixed)
			if (err == nil) != (scenario == "win" || scenario == "solo" || scenario == "release_frame") {
				t.Fatalf("%s: %v", scenario, err)
			}
		})
	}
}

func TestGoalStopUsesFrozenFixtureClass(t *testing.T) {
	for _, class := range []string{"monster_soldier", "monster_infantry"} {
		var goal GoalStop
		text := `{"version":"combat_goal_stop_v1","reason":"combat_goal_complete","spawncount":42,"actor":1,"classes":["` + class + `"],"kill_frame":15,"observed_frame":16,"health":80,"kills":[{"frame":15,"target":352,"target_class":"` + class + `"}]}`
		if err := json.Unmarshal([]byte(text), &goal); err != nil {
			t.Fatal(err)
		}
		observed := policy.Observation{Identity: policy.Identity{Map: "base1", Spawncount: 42, Actor: 1, Connection: 1, Life: 1, Frame: 16}, Health: 80}
		release := &CombatRelease{Spawncount: 42, Frame: 10}
		events := []DamageEvent{{Map: "base1", Spawncount: 42, Frame: 15, Attacker: 1, AttackerClass: "player", Target: 352, TargetClass: class, Mod: 1, HealthBefore: 7, HealthAfter: -3}}
		if err := VerifyGoalStopForClasses(goal, release, events, observed, []string{class}); err != nil {
			t.Fatal(err)
		}
		if err := VerifyGoalStopForClasses(goal, release, events, observed, []string{"monster_parasite"}); err == nil {
			t.Fatal("receipt changed the expected fixture class")
		}
		observed.Identity.Map = "base2"
		events[0].Map = "base2"
		if err := VerifyGoalStopForClasses(goal, release, events, observed, []string{class}, "base2"); err != nil {
			t.Fatal(err)
		}
		if err := VerifyGoalStopForClasses(goal, release, events, observed, []string{class}, "base1"); err == nil {
			t.Fatal("receipt changed frozen map")
		}
		events[0].Mod = 21
		if err := VerifyGoalStopForClasses(goal, release, events, observed, []string{class}); err == nil {
			t.Fatal("setup telefrag counted as learned kill")
		}
	}
}

func TestGoalStopRequiresEveryGroupMember(t *testing.T) {
	classes := []string{"monster_parasite", "monster_soldier", "monster_infantry", "monster_gunner"}
	for _, count := range []int{3, 4} {
		var goal GoalStop
		if err := json.Unmarshal([]byte(`{"version":"combat_goal_stop_v1","reason":"combat_goal_complete","spawncount":42,"actor":1,"kill_frame":20,"observed_frame":21,"health":80}`), &goal); err != nil {
			t.Fatal(err)
		}
		goal.Classes = classes[:count]
		observed := policy.Observation{Identity: policy.Identity{Map: "city1", Spawncount: 42, Actor: 1, Connection: 1, Life: 1, Frame: 21}, Health: 80}
		release := &CombatRelease{Spawncount: 42, Frame: 10}
		var events []DamageEvent
		for i, class := range goal.Classes {
			events = append(events, DamageEvent{Map: "city1", Spawncount: 42, Frame: 20, Attacker: 1, AttackerClass: "player", Target: 350 + i, TargetClass: class, Mod: 4, HealthBefore: 7, HealthAfter: -1})
			goal.Kills = append(goal.Kills, struct {
				Frame  int    `json:"frame"`
				Target int    `json:"target"`
				Class  string `json:"target_class"`
			}{20, 350 + i, class})
		}
		if err := VerifyGoalStopForClasses(goal, release, events, observed, classes[:count], "city1"); err != nil {
			t.Fatal(err)
		}
		if VerifyGoalStopForClasses(goal, release, events[:count-1], observed, classes[:count], "city1") == nil {
			t.Fatal("missing native kill accepted")
		}
		goal.Classes = goal.Classes[:count-1]
		goal.Kills = goal.Kills[:count-1]
		if VerifyGoalStopForClasses(goal, release, events, observed, classes[:count], "city1") == nil {
			t.Fatal("receipt omitted required group member")
		}
	}
}
