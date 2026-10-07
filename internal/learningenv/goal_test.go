package learningenv

import (
	"encoding/json"
	"testing"

	"q2coopbot/internal/policy"
)

func TestNativeGoalStopEvidence(t *testing.T) {
	for _, scenario := range []string{"win", "solo", "missing", "corpse", "prop", "death", "setup", "wrong_world", "unclosed", "forged_target", "second_life"} {
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
			if (err == nil) != (scenario == "win" || scenario == "solo") {
				t.Fatalf("%s: %v", scenario, err)
			}
		})
	}
}
