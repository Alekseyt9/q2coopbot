package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestSoloTacticRejectsStaleTargetAndFrame(t *testing.T) {
	clear := true
	w := World{Map: "base1", Goal: "reach_level_exit", Campaign: &CampaignDecision{}, Snapshot: quake.Snapshot{Map: "base1", Frame: 100, Health: 40, Weapon: "Blaster", OnGround: true, Enemies: []quake.Object{{ID: 7, Class: "monster_soldier", Origin: quake.Vec3{40, 0, 0}, ClearShot: &clear}}}}
	tactic := NewTactician("test")
	found := false
	for _, a := range tactic.options(w) {
		if a == "retreat" {
			found = true
		}
	}
	if !found {
		t.Fatal("solo retreat absent")
	}
	d := TacticalDecision{Action: "retreat", Map: "base1", Frame: 100, Target: 7, At: time.Now(), Source: "live"}
	for _, tc := range []struct {
		name   string
		change func(*World)
	}{
		{"frame", func(w *World) { w.Snapshot.Frame = 121 }},
		{"map", func(w *World) { w.Snapshot.Map = "base2" }},
		{"reset", func(w *World) { w.Snapshot.Frame = 3 }},
		{"death", func(w *World) { w.Snapshot.Health = 0 }},
		{"target", func(w *World) { w.Snapshot.Enemies = nil }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			changed := w
			tc.change(&changed)
			tactic.pending <- tacticalResult{decision: d}
			if _, ok := tactic.poll(changed); ok {
				t.Fatal("stale decision applied")
			}
		})
	}
	tactic.pending <- tacticalResult{decision: d}
	if got, ok := tactic.poll(w); !ok || got.Source != "live" {
		t.Fatal("fresh solo decision refused", got)
	}
}

func TestSoloRetreatKeepsFireAndAim(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	clear := true
	s := quake.Snapshot{Self: quake.Vec3{32, -224, 24}, OnGround: true, Health: 40, Weapon: "Blaster", Enemies: []quake.Object{{ID: 1, Class: "monster_soldier", Origin: quake.Vec3{96, -200, 24}, ClearShot: &clear}}}
	p := &Planner{Campaign: true, World: World{Geometry: &g, Snapshot: s, Command: CommandDecision{AimSource: "enemy"}}}
	cmd := p.combatRetreat(quake.UserCmd{Buttons: 1, Pitch: 123}, combatSpacing(s))
	if cmd.Forward >= 0 || cmd.Buttons != 1 || cmd.Pitch != 123 || p.World.Command.MoveSource != "combat_retreat" {
		t.Fatal("solo retreat did not preserve firing", cmd, p.World.Command)
	}
	p.Campaign = false
	p.World.Command = CommandDecision{}
	if got := p.combatRetreat(quake.UserCmd{}, combatSpacing(s)); got.Forward != 0 || got.Side != 0 {
		t.Fatal("retreat outside solo campaign")
	}
}
