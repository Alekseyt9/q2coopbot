package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBase1CoverCycleAndGroupGuard(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	clear := true
	s := quake.Snapshot{Map: "base1", Frame: 100, Health: 100, OnGround: true, Movers: []quake.Mover{{Model: 9}, {Model: 10}, {Model: 20}}, Weapon: "Blaster", Self: quake.Vec3{239.875, -416, 24.125}, Enemies: []quake.Object{{ID: 1, Class: "monster_infantry", Solid: 8290, Origin: quake.Vec3{200, -224, 24}, ClearShot: &clear}}}
	w := World{Map: s.Map, Goal: "reach_level_exit", Campaign: &CampaignDecision{}, Geometry: &g, Snapshot: s}
	c := plannedCover(w)
	if c == nil {
		t.Fatal("no short wall cover")
	}
	if !coverHidden(&g, c.hide, s.Enemies) || !coverWalkClear(w, c.peek, c.hide) {
		t.Fatal("invalid cover corridor")
	}
	p := &Planner{Campaign: true, World: w, cover: c}
	cmd, ok := p.combatCoverCommand(s, quake.UserCmd{Buttons: 1}, "cover")
	if !ok || cmd.Buttons != 0 {
		t.Fatal("withdraw fired", cmd)
	}
	s.Self = c.hide
	p.World.Snapshot = s
	p.combatCoverCommand(s, quake.UserCmd{}, "")
	s.Frame += 2
	p.World.Snapshot = s
	p.combatCoverCommand(s, quake.UserCmd{}, "")
	if c.stage != "peek" {
		t.Fatal(c.stage)
	}
	s.Self = c.peek
	s.Frame++
	p.World.Snapshot = s
	p.combatCoverCommand(s, quake.UserCmd{}, "")
	if c.stage != "fire" {
		t.Fatal(c.stage)
	}
	s.Frame++
	p.World.Snapshot = s
	cmd, ok = p.combatCoverCommand(s, quake.UserCmd{Buttons: 1}, "")
	if !ok || cmd.Buttons != 1 || cmd.Forward != 0 || cmd.Side != 0 {
		t.Fatal("firing exposure invalid", cmd)
	}
	s.Frame += 3
	p.World.Snapshot = s
	p.combatCoverCommand(s, quake.UserCmd{}, "")
	if c.stage != "return" {
		t.Fatal(c.stage)
	}
	s.Self = c.hide
	p.World.Snapshot = s
	p.combatCoverCommand(s, quake.UserCmd{}, "")
	if p.cover != nil {
		t.Fatal("cycle did not end behind wall")
	}
	w.Snapshot.Enemies = append(w.Snapshot.Enemies, quake.Object{ID: 2, Class: "monster_infantry", Solid: 8290, Origin: quake.Vec3{240, -448, 24}, ClearShot: &clear})
	if coverHidden(&g, c.hide, w.Snapshot.Enemies) {
		t.Fatal("wall falsely shields other threat")
	}
	w.Snapshot = s
	w.Snapshot.OnGround = false
	if plannedCover(w) != nil {
		t.Fatal("airborne cover offered")
	}
}
