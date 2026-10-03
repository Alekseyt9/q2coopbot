package bot

import (
	"math"
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBase1CircleCorridorAndGroup(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	clear := true
	s := quake.Snapshot{Map: "base1", Frame: 100, Health: 25, OnGround: true, Weapon: "Blaster", Self: quake.Vec3{32, -224, 24.125}, Movers: []quake.Mover{{Model: 9}, {Model: 10}, {Model: 20}}, Enemies: []quake.Object{{ID: 1, Class: "monster_infantry", Solid: 8290, Origin: quake.Vec3{200, -224, 24}, ClearShot: &clear}}}
	w := World{Goal: "reach_level_exit", Campaign: &CampaignDecision{}, Geometry: &g, Snapshot: s}
	next, sign, ok := circleStep(w, 0)
	if !ok || math.Abs(quake.Horizontal(next, s.Enemies[0].Origin)-quake.Horizontal(s.Self, s.Enemies[0].Origin)) > .01 || math.Abs(next[1]-s.Self[1]) < 10 {
		t.Fatal("no safe arc at maintained radius", next, ok)
	}
	p := &Planner{World: w}
	p.World.Command.AimSource = "enemy"
	cmd := p.combatCircle(quake.UserCmd{Buttons: 1})
	if cmd.Buttons != 1 || cmd.Forward == 0 && cmd.Side == 0 || p.circleDirection != sign {
		t.Fatal("circle lost firing or direction", cmd)
	}
	closeWorld := w
	closeWorld.Snapshot.Enemies = append([]quake.Object(nil), s.Enemies...)
	closeWorld.Snapshot.Enemies[0].Origin = quake.Vec3{96, -200, 24}
	closePlanner := &Planner{Campaign: true, World: closeWorld}
	closePlanner.World.Command.AimSource = "enemy"
	closeCmd := closePlanner.combatCircle(quake.UserCmd{Buttons: 1})
	if closePlanner.World.Command.MoveSource != "combat_retreat" || closeCmd.Forward == 0 && closeCmd.Side == 0 || closeCmd.Buttons != 1 {
		t.Fatal("circle did not fall back to firing retreat for close enemy", closeCmd)
	}
	// Retreat must also respect observed flank enemies behind occlusion.
	noLOS := false
	guardedWorld := closeWorld
	guardedWorld.Snapshot.Enemies = append(append([]quake.Object(nil), closeWorld.Snapshot.Enemies...), quake.Object{ID: 2, Class: "monster_infantry", Origin: s.Self, ClearShot: &noLOS})
	guardedWorld.Snapshot.Enemies[1].Origin[0] -= 40
	guarded := &Planner{Campaign: true, World: guardedWorld}
	guarded.World.Command.AimSource = "enemy"
	guardedCmd := guarded.combatCircle(quake.UserCmd{Buttons: 1})
	end := s.Self // This fixture has zero yaw/pitch/delta angles.
	end[0] += float64(guardedCmd.Forward) * .2
	end[1] -= float64(guardedCmd.Side) * .2
	flank := guardedWorld.Snapshot.Enemies[1].Origin
	if quake.Horizontal(end, flank) < quake.Horizontal(s.Self, flank)-2 {
		t.Fatal("retreat approached occluded flank enemy", guardedCmd)
	}
	// Both directions approach another observed threat, even without LOS.
	hidden := false
	w.Snapshot.Enemies = append(append([]quake.Object(nil), s.Enemies...), quake.Object{ID: 2, Class: "monster_infantry", Origin: quake.Vec3{32, -288, 24}, ClearShot: &hidden}, quake.Object{ID: 3, Class: "monster_infantry", Origin: quake.Vec3{32, -160, 24}, ClearShot: &hidden})
	if _, _, ok := circleStep(w, sign); ok {
		t.Fatal("arc approached a flank threat")
	}
	w.Snapshot = s
	w.Snapshot.OnGround = false
	if _, _, ok := circleStep(w, sign); ok {
		t.Fatal("airborne arc accepted")
	}
	w.Snapshot = s
	w.Snapshot.Enemies = nil
	if _, _, ok := circleStep(w, sign); ok {
		t.Fatal("lost target accepted")
	}
}
