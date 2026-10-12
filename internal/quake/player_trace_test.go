package quake

import (
	"math"
	"testing"
)

func playerTraceBoxes(boxes ...[2]Vec3) *MapInfo {
	c := &CollisionMap{}
	for _, box := range boxes {
		planes, brush := testBoxBrush(box[0], box[1], len(c.sides))
		for range planes {
			c.sides = append(c.sides, uint16(len(c.sides)))
		}
		c.planes = append(c.planes, planes...)
		c.worldBrushes = append(c.worldBrushes, len(c.brushes))
		c.brushes = append(c.brushes, brush)
	}
	return &MapInfo{collision: c}
}

func TestPlayerTraceHullEntryAndNearestBrush(t *testing.T) {
	m := playerTraceBoxes([2]Vec3{{80, -100, -100}, {90, 100, 100}}, [2]Vec3{{40, -100, -100}, {50, 100, 100}})
	tr := m.TracePlayer(Vec3{}, Vec3{100, 50, 0}, false)
	if !tr.Valid || tr.StartSolid || tr.AllSolid || tr.Normal != (Vec3{-1, 0, 0}) || math.Abs(tr.End[0]-23.96875) > 1e-9 || math.Abs(tr.End[1]-11.984375) > 1e-9 {
		t.Fatalf("expanded hull/nearest brush: %+v", tr)
	}
}

func TestPlayerTraceStartSolidLeavingAndPositionTest(t *testing.T) {
	m := playerTraceBoxes([2]Vec3{{40, -100, -100}, {50, 100, 100}}, [2]Vec3{{100, -100, -100}, {110, 100, 100}})
	tr := m.TracePlayer(Vec3{45, 0, 0}, Vec3{80, 0, 0}, false)
	if !tr.StartSolid || tr.AllSolid || tr.Fraction != 1 {
		t.Fatalf("native escape should be distinguishable: %+v", tr)
	}
	tr = m.TracePlayer(Vec3{45, 0, 0}, Vec3{120, 0, 0}, false)
	if !tr.StartSolid || tr.AllSolid || math.Abs(tr.End[0]-83.96875) > 1e-9 {
		t.Fatalf("exiting one brush hid later collision: %+v", tr)
	}
	tr = m.TracePlayer(Vec3{45, 0, 0}, Vec3{46, 0, 0}, false)
	if !tr.StartSolid || !tr.AllSolid {
		t.Fatal("embedded trajectory accepted", tr)
	}
	tr = m.TracePlayer(Vec3{45, 0, 0}, Vec3{45, 0, 0}, false)
	if !tr.AllSolid || tr.Fraction != 0 {
		t.Fatal("occupied position test", tr)
	}
}

func TestPlayerTraceCrouchAndNativePlayerClipMask(t *testing.T) {
	m := playerTraceBoxes([2]Vec3{{20, -100, 40}, {60, 100, 60}})
	standing := m.TracePlayer(Vec3{0, 0, 24}, Vec3{80, 0, 24}, false)
	ducked := m.TracePlayer(Vec3{0, 0, 24}, Vec3{80, 0, 24}, true)
	if standing.Fraction >= 1 || ducked.Fraction != 1 {
		t.Fatalf("crouch hull mismatch: standing=%+v ducked=%+v", standing, ducked)
	}
	m = playerTraceBoxes([2]Vec3{{40, -100, -100}, {50, 100, 100}})
	m.collision.brushes[0].contents = 0x10000
	if tr := m.TracePlayer(Vec3{}, Vec3{80, 0, 0}, false); tr.Fraction >= 1 {
		t.Fatal("playerclip missing", tr)
	}
	m.collision.brushes[0].contents = 0x20000 // monsterclip does not block players
	if tr := m.TracePlayer(Vec3{}, Vec3{80, 0, 0}, false); tr.Fraction != 1 {
		t.Fatal("monsterclip treated as playerclip", tr)
	}
}

func TestPlayerTraceTouchingFloorAndInvalidInput(t *testing.T) {
	m := playerTraceBoxes([2]Vec3{{-100, -100, -20}, {100, 100, 0}})
	if tr := m.TracePlayer(Vec3{0, 0, 24.125}, Vec3{20, 0, 24.125}, false); tr.Fraction != 1 || tr.StartSolid {
		t.Fatal("flat floor blocked lateral sweep", tr)
	}
	if tr := m.TracePlayer(Vec3{0, 0, 24.125}, Vec3{0, 0, 20}, false); tr.Normal != (Vec3{0, 0, 1}) || math.Abs(tr.End[2]-24.03125) > 1e-9 {
		t.Fatal("floor contact fraction", tr)
	}
	if tr := m.TracePlayer(Vec3{math.NaN(), 0, 0}, Vec3{}, false); tr.Valid {
		t.Fatal("invalid input accepted", tr)
	}
	if tr := (*MapInfo)(nil).TracePlayer(Vec3{}, Vec3{}, false); tr.Valid {
		t.Fatal("unknown map accepted", tr)
	}
}

func TestPlayerTraceObservedMoverPoseAndUnknownModel(t *testing.T) {
	m := playerTraceBoxes([2]Vec3{{-200, -200, -20}, {200, 200, 0}}, [2]Vec3{{40, -100, 0}, {50, 100, 100}})
	m.collision.worldBrushes = []int{0}
	m.collision.modelBrushes = [][]int{nil, {1}}
	from, to := Vec3{0, 0, 24.125}, Vec3{100, 0, 24.125}
	if tr := m.TracePlayer(from, to, false); tr.Fraction != 1 {
		t.Fatal("mover incorrectly part of static world", tr)
	}
	tr := m.TracePlayerWithMovers(from, to, false, []Mover{{Model: 1, Origin: Vec3{20, 0, 0}}})
	if !tr.Valid || math.Abs(tr.End[0]-43.96875) > 1e-9 || tr.Normal != (Vec3{-1, 0, 0}) {
		t.Fatal("translated mover trace", tr)
	}
	tr = m.TracePlayerWithMovers(from, Vec3{0, 100, 24.125}, false, []Mover{{Model: 1, Angles: Vec3{0, 90, 0}}})
	if !tr.Valid || math.Abs(tr.End[1]-23.96875) > 1e-9 || math.Abs(tr.Normal[1]+1) > 1e-9 {
		t.Fatal("native transformed mover trace", tr)
	}
	if tr = m.TracePlayerWithMovers(from, to, false, []Mover{{Model: 2}}); tr.Valid {
		t.Fatal("unknown mover cleared", tr)
	}
}
