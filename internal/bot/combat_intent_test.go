package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"slices"
	"testing"
	"time"
)

func TestCampaignCombatIntentAndStaleRouteChoice(t *testing.T) {
	clear, hidden := true, false
	w := World{Map: "base1", Goal: "reach_level_exit", Campaign: &CampaignDecision{}, Snapshot: quake.Snapshot{
		Map: "base1", Frame: 100, Health: 100, Weapon: "Blaster", OnGround: true,
		Self: quake.Vec3{32, -224, 24.125}, Enemies: []quake.Object{{ID: 7, Origin: quake.Vec3{200, -224, 24}, ClearShot: &clear}},
	}}
	tactician := NewTactician("test")
	for _, tc := range []struct {
		name, action string
		change       func(*World)
	}{
		{"visible", "engage", func(w *World) {}},
		{"health_route", "recover", func(w *World) { w.Goal = "recover_health"; w.Snapshot.Health = 30 }},
		{"hidden", "route", func(w *World) { w.Snapshot.Enemies[0].ClearShot = &hidden }},
		{"distant", "route", func(w *World) { w.Snapshot.Enemies[0].Origin[0] = 900 }},
		{"no_ammo", "route", func(w *World) { w.Snapshot.Weapon = "Machinegun" }},
		{"button", "route", func(w *World) { w.Goal = "touch_button" }},
		{"partner", "", func(w *World) { at := quake.Vec3{}; w.Snapshot.Teammate = &at }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			changed := w
			changed.Snapshot.Enemies = slices.Clone(w.Snapshot.Enemies)
			tc.change(&changed)
			intent := combatIntent(changed)
			if tc.action == "" {
				if intent != nil {
					t.Fatal(intent)
				}
				return
			}
			if intent == nil || intent.Action != tc.action {
				t.Fatal(intent)
			}
			if slices.Contains(tactician.options(changed), "follow") == (tc.action == "engage") {
				t.Fatal("incorrect route option", tactician.options(changed))
			}
		})
	}
	// A delayed follow response must be rejected when a new visible threat
	// now interrupts the campaign route, even within its ordinary frame TTL.
	tactician.pending <- tacticalResult{decision: TacticalDecision{Action: "follow", Map: "base1", Frame: 99}}
	if _, ok := tactician.poll(w); ok {
		t.Fatal("stale route choice accepted")
	}
}

func TestBase1CampaignCombatCommandGapAndRecovery(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	clear := true
	w := World{Map: "base1", Updated: now, Goal: "reach_level_exit", Campaign: &CampaignDecision{}, Geometry: &g, Snapshot: quake.Snapshot{
		Map: "base1", Frame: 100, Health: 100, Weapon: "Blaster", OnGround: true, Self: quake.Vec3{32, -224, 24.125},
		Enemies: []quake.Object{{ID: 7, Class: "monster_infantry", Origin: quake.Vec3{200, -224, 24}, ClearShot: &clear}},
	}}
	for _, action := range []string{"", "follow", "recover"} {
		p := Planner{Campaign: true, World: w}
		if action != "" {
			p.World.Tactic = &TacticalDecision{Action: action}
		}
		cmd := p.commandAt(quake.UserCmd{}, now)
		if cmd.Buttons&1 == 0 || cmd.Forward != 0 || cmd.Side != 0 || p.World.Command.MoveLimitReason != "campaign_combat_hold" || p.World.Command.AimSource != "enemy" {
			t.Fatal("pending/stale route choice left combat", action, cmd, p.World.Command)
		}
	}
	// A health route must keep moving even if the old model chose attack;
	// it may shoot the observed enemy without giving up the resource goal.
	p := Planner{Campaign: true, World: w, hasGoal: true, goalPoint: quake.Vec3{32, -184, 24.125}}
	p.World.Goal, p.World.Navigation = "recover_health", "direct_clear"
	p.World.Tactic = &TacticalDecision{Action: "attack"}
	cmd := p.commandAt(quake.UserCmd{}, now)
	if cmd.Buttons&1 == 0 || p.World.Command.MoveSource != "route" || cmd.Forward == 0 && cmd.Side == 0 || p.World.Command.CombatIntent.Action != "recover" {
		t.Fatal("resource route lost to old attack", cmd, p.World.Command)
	}
}
