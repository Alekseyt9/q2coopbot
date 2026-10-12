package quake

import (
	"math"
	"testing"
)

func TestPlayerStepChoosesSupportedLowStep(t *testing.T) {
	for _, height := range []float64{16, 24} {
		m := playerTraceBoxes([2]Vec3{{-100, -100, -20}, {100, 100, 0}}, [2]Vec3{{20, -100, 0}, {100, 100, height}})
		slide, ok := m.StepSlidePlayer(Vec3{0, 0, 24.125}, Vec3{300, 0, 0}, .1, false)
		if !ok {
			t.Fatal("supported trajectory unavailable", height, slide)
		}
		if height == 16 && (math.Abs(slide.Position[0]-30) > 1e-9 || math.Abs(slide.Position[2]-40.03125) > 1e-9) {
			t.Fatal("supported step not climbed", slide)
		}
		if height == 24 && slide.Position[0] >= 4 {
			t.Fatal("climbed step above native limit", slide)
		}
	}
}

func TestPlayerStepRetainsOrdinarySlideUnderLowCeiling(t *testing.T) {
	m := playerTraceBoxes([2]Vec3{{-100, -100, -20}, {100, 100, 0}},
		[2]Vec3{{40, -100, 0}, {100, 100, 100}}, [2]Vec3{{-100, -100, 64}, {100, 100, 80}})
	from, velocity := Vec3{0, 0, 24.125}, Vec3{300, 100, 0}
	ordinary, _ := m.SlidePlayer(from, velocity, .1, false)
	step, ok := m.StepSlidePlayer(from, velocity, .1, false)
	if !ok || ordinary.Position != step.Position || ordinary.Velocity != step.Velocity {
		t.Fatal("blocked raised hull replaced ordinary slide", ordinary, step, ok)
	}
}

func TestPlayerStepDoesNotInventUnsupportedLanding(t *testing.T) {
	m := playerTraceBoxes([2]Vec3{{-100, -100, -20}, {10, 100, 0}})
	from, velocity := Vec3{0, 0, 24.125}, Vec3{300, 0, 0}
	ordinary, _ := m.SlidePlayer(from, velocity, .1, false)
	step, ok := m.StepSlidePlayer(from, velocity, .1, false)
	if !ok || step.Position != ordinary.Position {
		t.Fatal("unsupported raised landing chosen", ordinary, step, ok)
	}
	// Collision success is not ground safety: caller must reject this floor gap.
	if _, supported := m.GroundDrop(step.Position, 24); supported {
		t.Fatal("fixture still supported beyond edge")
	}
}
