package quake

import (
	"math"
	"testing"
)

func TestPlayerSlideAlongWallPreservesTangentialVelocity(t *testing.T) {
	m := playerTraceBoxes([2]Vec3{{40, -100, -100}, {50, 100, 100}})
	slide, ok := m.SlidePlayer(Vec3{}, Vec3{300, 100, 0}, .1, false)
	if !ok || len(slide.Contacts) != 1 || len(slide.Segments) != 2 || math.Abs(slide.Position[1]-10) > 1e-9 || slide.Position[0] >= 24 || slide.Velocity[1] != 100 || math.Abs(slide.Velocity[0]+3) > 1e-9 {
		t.Fatalf("lost native wall slide: %+v accepted=%v", slide, ok)
	}
	for _, segment := range slide.Segments {
		if segment[0][0] > 24 || segment[1][0] > 24 {
			t.Fatal("hull crossed wall", segment)
		}
	}
}

func TestPlayerSlideCornerAndUncertainStarts(t *testing.T) {
	m := playerTraceBoxes([2]Vec3{{40, -100, -100}, {50, 100, 100}}, [2]Vec3{{-100, 40, -100}, {100, 50, 100}})
	slide, ok := m.SlidePlayer(Vec3{}, Vec3{300, 300, 0}, .1, false)
	if !ok || slide.Position[0] >= 24 || slide.Position[1] >= 24 || len(slide.Contacts) < 2 {
		t.Fatal("corner penetration", slide, ok)
	}
	if escape, ok := m.SlidePlayer(Vec3{45, 0, 0}, Vec3{-300, 0, 0}, .1, false); !ok || escape.Position[0] != 15 {
		t.Fatal("native escape collision phase lost", escape, ok)
	}
	if _, ok := m.SlidePlayer(Vec3{}, Vec3{math.Inf(1), 0, 0}, .1, false); ok {
		t.Fatal("invalid velocity accepted")
	}
	if _, ok := m.SlidePlayer(Vec3{}, Vec3{100, 0, 0}, .2, false); ok {
		t.Fatal("unbounded tick accepted")
	}
}
