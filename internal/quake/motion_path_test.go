package quake

import "testing"

func motionFloor() *MapInfo {
	m := playerTraceBoxes([2]Vec3{{-100, -100, -20}, {100, 100, 0}})
	m.Models = []BSPModel{{}}
	m.collision.sideSurfaces = make([]groundSurface, len(m.collision.sides))
	for i := range m.collision.sideSurfaces {
		m.collision.sideSurfaces[i] = groundSurface{known: true}
	}
	return m
}

func TestMotionPathRejectsFloorEdgeSurfaceAndLaser(t *testing.T) {
	m := motionFloor()
	from := Vec3{0, 0, 24.125}
	to := Vec3{20, 0, 24.125}
	if h := m.MotionPathHazard(from, to, false); h != "" {
		t.Fatal("flat path", h)
	}
	if h := m.MotionPathHazard(Vec3{80, 0, 24.125}, Vec3{110, 0, 24.125}, false); h == "" {
		t.Fatal("floor edge accepted")
	}
	m.collision.sideSurfaces[4].flags = 2
	if h := m.MotionPathHazard(from, to, false); h == "" {
		t.Fatal("slick support accepted")
	}
	m = motionFloor()
	m.laserBeams = []laserBeam{{start: Vec3{10, -50, 24}, end: Vec3{10, 50, 24}, damage: 100}}
	if h := m.MotionPathHazard(from, to, false); h != "static_laser_hazard" {
		t.Fatal("laser crossing accepted", h)
	}
}
