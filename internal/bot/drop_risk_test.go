package bot

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"q2coopbot/internal/quake"
)

func TestDropDamageBudget(t *testing.T) {
	for _, tc := range []struct {
		height, velocity float64
		gravity, hp      int16
		damage           int
		allowed          bool
	}{
		{40, 0, 800, 7, 0, true}, {128, 0, 800, 7, 0, true},
		{208, 0, 800, 100, 7, true}, {208, 0, 800, 32, 7, true},
		{208, 0, 800, 31, 7, false}, {208, 0, 800, 7, 7, false},
		{240, 0, 800, 40, 10, true}, {400, 0, 800, 100, 24, false},
		{280, 0, 800, 51, 14, true}, {280, 0, 800, 40, 14, false},
		{40, -600, 800, 7, 12, false}, {208, 0, 1600, 100, 33, false},
	} {
		damage := estimatedDropDamage(tc.height, tc.velocity, tc.gravity)
		if damage != tc.damage || affordableDrop(tc.hp, damage) != tc.allowed {
			t.Errorf("%+v: damage=%d affordable=%v", tc, damage, affordableDrop(tc.hp, damage))
		}
	}
	for _, height := range []float64{-1, math.NaN(), math.Inf(1), 1e300} {
		if affordableDrop(100, estimatedDropDamage(height, 0, 800)) {
			t.Fatal("invalid/extreme drop accepted")
		}
	}
	if affordableDrop(100, estimatedDropDamage(208, 0, 0)) || affordableDrop(0, 0) {
		t.Fatal("unknown gravity or dead player accepted")
	}
}

func TestBase1DropHealthBudget(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	nav, err := quake.LoadAAS(filepath.Join(root, "maps/base1.aas"))
	if err != nil {
		t.Fatal(err)
	}
	goal := quake.Vec3{-850, 1292, 136.125}
	for _, hp := range []int16{100, 32, 31, 7} {
		p := &Planner{Nav: nav, World: World{Map: "base1", Geometry: &g}, GameClock: true}
		p.update(quake.Snapshot{Map: "base1", Frame: 100, Health: hp, Gravity: 800, Self: quake.Vec3{-937, 1149, 344.125}, OnGround: true, Teammate: &goal}, "")
		accepted := p.planWalkOff()
		if accepted != (hp >= 32) {
			t.Fatalf("hp=%d accepted=%v route=%+v", hp, accepted, p.World.Route)
		}
		if accepted && (p.jump.expectedDamage != 7 || !p.jump.drop) {
			t.Fatalf("bad damage trace: %+v", p.jump)
		}
	}
}

func TestNearbyDropAndQuickBypass(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base3 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "base3")
	if err != nil {
		t.Fatal(err)
	}
	nav, err := quake.LoadAAS(filepath.Join(root, "maps/base3.aas"))
	if err != nil {
		t.Fatal(err)
	}
	goal := quake.Vec3{980, -574, -487.875}
	for _, hp := range []int16{100, 44, 43, 36, 35, 7} {
		p := &Planner{Nav: nav, GameClock: true, World: World{Map: "base3", Geometry: &g}}
		p.update(quake.Snapshot{Map: "base3", Frame: 100, Health: hp, Gravity: 800, OnGround: true, Self: quake.Vec3{831, -574, -231.875}, Teammate: &goal}, "")
		if accepted := p.planNearbyWalkOff(); accepted != (hp >= 36) {
			t.Fatalf("hp=%d accepted=%v", hp, accepted)
		}
		if p.jump != nil && p.jump.expectedDamage != 11 {
			t.Fatalf("unexpected budget: %+v", p.jump)
		}
		p.jump = nil
		p.World.Route = []quake.Waypoint{{Position: goal, Kind: 2}}
		if p.planNearbyWalkOff() {
			t.Fatal("quick bypass ignored")
		}
	}
	if preferDrop(2, 3, 0) || preferDrop(2, 4, 8) || !preferDrop(2, 6, 8) {
		t.Fatal("time/health tradeoff changed")
	}
}

func TestDeepDropConfirmsLandingAndRetainsTravelBound(t *testing.T) {
	from, landing := quake.Vec3{831, -574, -231.875}, quake.Vec3{855, -574, -487.625}
	makePlanner := func(at quake.Vec3) *Planner {
		return &Planner{jump: &jumpFlight{from: from, landing: landing, frame: 140, airborne: true, phase: 2, drop: true}, World: World{Snapshot: quake.Snapshot{Frame: 153, Health: 99, OnGround: true, Self: at}}}
	}
	p := makePlanner(quake.Vec3{874, -574, -487.875})
	p.jumpCommand(quake.UserCmd{})
	if p.World.Command.LimitReason != "drop_landed" || p.jump != nil {
		t.Fatalf("deep landing rejected: %+v", p.World.Command)
	}
	p = makePlanner(quake.Vec3{1200, -574, -487.875})
	p.jumpCommand(quake.UserCmd{})
	if p.World.Command.LimitReason != "drop_aborted" {
		t.Fatal("unbounded descent accepted")
	}
}
