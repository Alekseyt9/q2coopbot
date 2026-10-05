package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestBase1ParasiteCornerEscape(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	clear := true
	s := quake.Snapshot{Health: 44, Weapon: "Blaster", OnGround: true, Self: quake.Vec3{-40.375, -426, 24.125}, Enemies: []quake.Object{{ID: 67, Class: "monster_parasite", Origin: quake.Vec3{67.125, -316.875, 24}, ClearShot: &clear}}}
	p := Planner{Campaign: true, World: World{Geometry: &g, Snapshot: s, Command: CommandDecision{AimSource: "enemy"}}}
	cmd := p.combatRetreat(quake.UserCmd{Buttons: 1}, combatSpacing(s))
	if p.World.Command.MoveSource != "combat_corner_escape" || cmd.Forward == 0 && cmd.Side == 0 || cmd.Buttons != 1 {
		t.Fatal("recorded fatal corner remains blocked", cmd, p.World.Command)
	}
	next := s.Self
	next[0] += float64(cmd.Forward) * .1
	next[1] -= float64(cmd.Side) * .1
	if !coverWalkClear(p.World, s.Self, next) {
		t.Fatal("escape edge blocked", next)
	}
	// Once moving toward the exit lane, ordinary retreat must not send us
	// back to the original wall on the following frame.
	s.Self = next
	s.Frame++
	p.World.Snapshot = s
	p.World.Command = CommandDecision{AimSource: "enemy"}
	continued, ok := p.continueCornerEscape(quake.UserCmd{Buttons: 1}, combatSpacing(s), s.Enemies[0])
	if !ok || continued.Forward == 0 && continued.Side == 0 {
		t.Fatal("escape interrupted before lane", continued)
	}
	s.Health = 0
	p.World.Snapshot = s
	if _, ok := p.continueCornerEscape(quake.UserCmd{}, combatSpacing(s), s.Enemies[0]); ok || p.cornerEscape != nil {
		t.Fatal("escape survived death")
	}
	s.Health = 44
	s.Self = quake.Vec3{-40.375, -426, 24.125}
	// A teammate occupying the only way out must still block this fallback.
	s.Teammate = &s.Self
	p.World.Snapshot = s
	p.World.Command = CommandDecision{AimSource: "enemy"}
	cmd = p.combatRetreat(quake.UserCmd{}, combatSpacing(s))
	if cmd.Forward != 0 || cmd.Side != 0 {
		t.Fatal("escape pushed through teammate", cmd)
	}
}

func TestCornerEscapeRejectsMonsterInsideEdge(t *testing.T) {
	s := quake.Snapshot{Self: quake.Vec3{0, 0, 24}, Enemies: []quake.Object{{ID: 1, Origin: quake.Vec3{32, 0, 24}}}}
	if cornerThreatClear(s, s.Self, quake.Vec3{64, 0, 24}, 1, 32) {
		t.Fatal("endpoint clearance ignored intervening monster")
	}
	s.Enemies = []quake.Object{{ID: 2, Origin: quake.Vec3{40, 0, 24}}}
	if cornerThreatClear(s, s.Self, quake.Vec3{0, 32, 24}, 1, 32) == false {
		t.Fatal("safe flank separation rejected")
	}
	if cornerThreatClear(s, s.Self, quake.Vec3{32, 0, 24}, 1, 32) {
		t.Fatal("escape approached second monster")
	}
}

func TestParasiteMixedAndRecoveryFixtureGeometry(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	self := quake.Vec3{32, -224, 24.125}
	for _, at := range []quake.Vec3{{96, -200, 24.125}, {32, -424, 24.125}} {
		if !g.PlayerMoveClear(at, at) {
			t.Fatal("fixture inside BSP", at)
		}
		if drop, ok := g.GroundDrop(at, 4); !ok || drop > 1 {
			t.Fatal("fixture floor missing", at, drop, ok)
		}
		if !g.ClearShot((quake.Snapshot{Self: self}).EyePoint(), at) {
			t.Fatal("fixture wall hides target", at)
		}
	}
}

func TestParasiteRailFixtureGeometry(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	self, enemy := quake.Vec3{32, -352, 24.125}, quake.Vec3{240, -160, 24.125}
	for _, at := range []quake.Vec3{self, enemy} {
		if !g.PlayerMoveClear(at, at) {
			t.Fatal("fixture inside BSP", at)
		}
		if drop, ok := g.GroundDrop(at, 4); !ok || drop > 1 {
			t.Fatal("fixture floor missing", at, drop, ok)
		}
	}
	if !g.ClearShot((quake.Snapshot{Self: self}).EyePoint(), enemy) {
		t.Fatal("fixture hides parasite")
	}
	d := quake.Distance(self, enemy)
	if d < 256 || d >= 320 {
		t.Fatal("rail fixture must need retreat while permitting rail", d)
	}
}

func TestRecordedMixedCornerDetour(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	clear := true
	s := quake.Snapshot{Health: 30, OnGround: true, Self: quake.Vec3{-43.875, -422.125, 24.125}, Enemies: []quake.Object{{ID: 67, Class: "monster_parasite", Origin: quake.Vec3{161.375, -238.625, 24}, ClearShot: &clear}, {ID: 343, Class: "monster_gunner", Origin: quake.Vec3{30.375, -240, 24}, ClearShot: &clear}}}
	p := Planner{Campaign: true, World: World{Snapshot: s, Geometry: &g, Command: CommandDecision{AimSource: "enemy"}}}
	cmd := p.combatRetreat(quake.UserCmd{Buttons: 1}, combatSpacing(s))
	if p.World.Command.MoveSource != "combat_corner_detour" || p.cornerEscape == nil || cmd.Buttons != 1 {
		t.Fatal("recorded mixed corner still blocked", cmd, p.World.Command)
	}
	c := p.cornerEscape
	from := s.Self
	for _, at := range c.path {
		if !coverWalkClear(p.World, from, at) || !cornerDetourThreatClear(s, from, at, 67, 96, c.threatFloors) {
			t.Fatal("unsafe detour edge", from, at)
		}
		from = at
	}
	// Resource intent must not cancel a checked active escape.
	now := time.Now()
	p.World.Map = "base1"
	p.World.Snapshot.Map = "base1"
	p.World.Snapshot.Frame = 50
	p.World.Snapshot.Weapon = "Blaster"
	c.mapName = "base1"
	c.started = 50
	c.until = 90
	p.World.Updated = now
	p.World.Goal = "recover_health"
	p.World.Campaign = &CampaignDecision{}
	p.commandAt(quake.UserCmd{}, now)
	if p.World.Command.MoveSource != "combat_corner_detour" || p.World.Goal != "recover_health" || p.World.Command.CombatIntent.Action != "recover" {
		t.Fatal("recovery abandoned the active escape", p.World.Command)
	}
	// A newly observed actor in the checked edge still stops execution.
	next := c.path[0]
	p.World.Snapshot.Enemies = append(s.Enemies, quake.Object{ID: 400, Class: "monster_berserk", Origin: next})
	if _, ok := p.continueCornerEscape(quake.UserCmd{}, combatSpacing(s), s.Enemies[0]); ok {
		t.Fatal("new threat did not stop detour")
	}
}

func TestMixedDetourFixedThreatFloor(t *testing.T) {
	s := quake.Snapshot{Self: quake.Vec3{110, 0, 24}, Enemies: []quake.Object{{ID: 1, Class: "monster_parasite", Origin: quake.Vec3{0, 0, 24}}, {ID: 2, Class: "monster_gunner", Origin: quake.Vec3{0, 200, 24}}}}
	floors := map[int]float64{1: 96, 2: 64}
	if cornerDetourThreatClear(s, s.Self, quake.Vec3{90, 0, 24}, 1, 32, floors) {
		t.Fatal("execution relaxed the fixed primary floor")
	}
	if !cornerDetourThreatClear(s, s.Self, quake.Vec3{100, 0, 24}, 1, 32, floors) {
		t.Fatal("within-budget supported threat tradeoff rejected")
	}
}

func TestMixedCornerDetourCancelsPhysicalStall(t *testing.T) {
	clear := true
	s := quake.Snapshot{Map: "base1", Frame: 10, Health: 80, OnGround: true, Enemies: []quake.Object{{ID: 1, Class: "monster_parasite", Origin: quake.Vec3{200, 0, 0}, ClearShot: &clear}}}
	c := &cornerEscape{mapName: "base1", target: 1, started: 10, until: 50, threatFloors: map[int]float64{1: 96}, lastFrame: 10, lastSelf: s.Self, stalled: 1}
	p := Planner{World: World{Snapshot: s}, cornerEscape: c}
	p.World.Snapshot.Frame = 11
	if _, ok := p.continueCornerEscape(quake.UserCmd{}, combatSpacing(s), s.Enemies[0]); ok || p.cornerEscape != nil || p.cornerDetourRetry != 31 {
		t.Fatal("physical stall retained path", p.cornerDetourRetry)
	}
	p.World.Snapshot.Frame = 12
	p.World.Snapshot.Enemies = append(s.Enemies, quake.Object{ID: 2, Class: "monster_gunner", ClearShot: &clear})
	p.combatMixedCornerDetour(quake.UserCmd{}, combatSpacing(s), s.Enemies[0])
	if p.cornerEscape != nil {
		t.Fatal("stall cooldown bypassed")
	}
}
