package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestParasiteRecordedFiringStall(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP assets required")
	}
	g, e := quake.LoadMap(root, "base1")
	if e != nil {
		t.Fatal(e)
	}
	clear := true
	friend := quake.Vec3{32, -352, 24.125}
	s := quake.Snapshot{Self: quake.Vec3{-44.625, -424.625, 24.125}, OnGround: true, Teammate: &friend, Weapon: "Blaster", Enemies: []quake.Object{{ID: 57, Class: "monster_parasite", Origin: quake.Vec3{200, -224, 24}, Solid: 7266, ClearShot: &clear}}}
	p := &Planner{World: World{Geometry: &g, Snapshot: s, Command: CommandDecision{AimEntity: 57, LimitReason: "friendly_line_of_fire"}}}
	got := p.combatRetreat(quake.UserCmd{}, combatSpacing(s))
	if p.World.Command.MoveSource != "combat_firing_position" || got.Forward == 0 && got.Side == 0 || got.Buttons != 0 {
		t.Fatalf("recorded stall: %+v %+v", got, p.World.Command)
	}
	// Refinement must not buy a firing lane by approaching a second threat.
	s.Enemies = append(s.Enemies, quake.Object{ID: 58, Origin: quake.Vec3{0, -424, 24}})
	p.World.Snapshot = s
	p.World.Command = CommandDecision{AimEntity: 57, LimitReason: "friendly_line_of_fire"}
	p.combatRetreat(quake.UserCmd{}, combatSpacing(s))
	if p.World.Command.MoveSource == "combat_firing_position" {
		t.Fatal("refined search approached a second threat")
	}
}
