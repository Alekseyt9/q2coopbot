package bot

import (
	"math"
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBase1GroupRetreatFixtureAndFlank(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	self, front, flank := quake.Vec3{128, -304, 24.125}, quake.Vec3{192, -304, 24.125}, quake.Vec3{32, -352, 24.125}
	for _, at := range []quake.Vec3{self, front, flank} {
		if !g.PlayerMoveClear(at, at) {
			t.Fatal("fixture hull blocked", at)
		}
		if drop, ok := g.GroundDrop(at, 4); !ok || drop > 1 {
			t.Fatal("fixture unsupported", at, drop, ok)
		}
	}
	clear := true
	s := quake.Snapshot{Health: 100, Weapon: "Blaster", OnGround: true, Self: self, Enemies: []quake.Object{
		{ID: 1, Class: "monster_infantry", Origin: front, ClearShot: &clear},
		{ID: 2, Class: "monster_infantry", Origin: flank, ClearShot: &clear},
	}}
	for _, e := range s.Enemies {
		if !g.ClearShot(s.EyePoint(), e.AimPoint()) {
			t.Fatal("fixture target occluded", e.ID)
		}
	}
	// Straight west approaches the southwest flank; the selected retreat
	// must choose an oblique supported step that separates from the primary.
	straight := self
	straight[0] -= 16
	if quake.Distance(straight, flank) >= quake.Distance(self, flank)-2 {
		t.Fatal("fixture lacks flank constraint")
	}
	p := Planner{Campaign: true, World: World{Geometry: &g, Snapshot: s, Command: CommandDecision{AimSource: "enemy"}}}
	cmd := p.combatRetreat(quake.UserCmd{Buttons: 1}, combatSpacing(s))
	if p.World.Command.MoveSource != "combat_retreat" || cmd.Buttons&1 == 0 {
		t.Fatal("retreat absent", cmd, p.World.Command)
	}
	// With view yaw/pitch zero the native command basis is x=forward,y=-side.
	next := self
	next[0] += float64(cmd.Forward) * .2
	next[1] -= float64(cmd.Side) * .2
	if quake.Distance(next, front) <= quake.Distance(self, front)+0.5 || quake.Distance(next, flank) < quake.Distance(self, flank)-2 || math.Abs(float64(cmd.Side)) < 1 {
		t.Fatal("retreat approached flank or failed to separate", cmd, next)
	}
}
