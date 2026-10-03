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
	if !g.ClearShot(s.EyePoint(), s.Enemies[0].AimPoint()) || coverBlasterClear(w, s.Self, s.Enemies[0]) {
		t.Fatal("fixture must expose eye-visible but muzzle-blocked target")
	}
	// ClearShot shrinks brushes for visibility. A grazing bolt still collides
	// in MASK_SHOT and this original target cannot have a flat local peek.
	if plannedCover(w) != nil {
		t.Fatal("grazing wall trajectory offered as cover")
	}
	s.Enemies[0].Origin[0] = 240
	w.Snapshot = s
	c := plannedCover(w)
	if c == nil {
		t.Fatal("no short wall cover")
	}
	if !coverHidden(&g, c.hide, s.Enemies) || !coverWalkClear(w, c.peek, c.hide) {
		t.Fatal("invalid cover corridor")
	}
	if !coverBlasterClear(w, c.peek, s.Enemies[0]) {
		t.Fatal("peek did not repair blocked blaster trajectory")
	}
	blocked := *c
	blocked.stage = "fire"
	blockedWorld := w
	blockedWorld.Snapshot.Enemies = append([]quake.Object(nil), s.Enemies...)
	blockedWorld.Snapshot.Enemies[0].Origin[0] = 200
	guard := &Planner{World: blockedWorld, cover: &blocked}
	guarded, handled := guard.combatCoverCommand(blockedWorld.Snapshot, quake.UserCmd{Buttons: 1}, "")
	if !handled || guarded.Buttons != 0 || blocked.stage != "return" {
		t.Fatal("blocked muzzle was allowed to fire")
	}
	mixed := w
	mixed.Snapshot.Enemies = append(append([]quake.Object(nil), s.Enemies...), quake.Object{ID: 2, Class: "monster_tank", Origin: quake.Vec3{200, -224, 24}})
	mixedCover := *c
	mixedPlanner := &Planner{World: mixed, cover: &mixedCover}
	if _, handled := mixedPlanner.combatCoverCommand(mixed.Snapshot, quake.UserCmd{}, ""); handled || mixedPlanner.cover != nil {
		t.Fatal("new explosive threat after target did not cancel cover")
	}
	p := &Planner{Campaign: true, World: w, cover: c}
	cmd, ok := p.combatCoverCommand(s, quake.UserCmd{Buttons: 1}, "cover")
	if !ok || cmd.Buttons != 0 {
		t.Fatal("withdraw fired", cmd)
	}
	s.Self = c.hide
	p.World.Snapshot = s
	p.combatCoverCommand(s, quake.UserCmd{}, "")
	s.Frame += 4
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
	s.Frame += 8
	p.World.Snapshot = s
	p.combatCoverCommand(s, quake.UserCmd{}, "")
	if c.stage != "return" {
		t.Fatal(c.stage)
	}
	s.Self = c.hide
	p.World.Snapshot = s
	p.combatCoverCommand(s, quake.UserCmd{}, "")
	if p.cover == nil || c.stage != "wait" || c.rounds != 1 {
		t.Fatal("cycle did not schedule another protected peek")
	}
	clear = false // Target remains observed but its line is occluded.
	s.Frame += 4
	p.World.Snapshot = s
	p.combatCoverCommand(s, quake.UserCmd{}, "")
	if c.stage != "peek" {
		t.Fatal("hidden observed target did not permit repeat")
	}
	s.Enemies = nil
	s.Self = c.hide
	p.World.Snapshot = s
	p.combatCoverCommand(s, quake.UserCmd{Buttons: 1}, "")
	if p.cover != nil {
		t.Fatal("lost target kept maneuver alive")
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
