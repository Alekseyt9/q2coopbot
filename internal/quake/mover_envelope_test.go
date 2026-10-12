package quake

import "testing"

func TestMoverEnvelopeDoesNotFreezeOpenDoor(t *testing.T) {
	m := &MapInfo{Models: []BSPModel{{}, {Min: Vec3{40, -20, 0}, Max: Vec3{72, 20, 100}}}, Entities: []MapEntity{{Class: "func_door", Model: 1, Lip: 8}}}
	live := Mover{Model: 1, Origin: Vec3{24, 0, 0}, SolidKnown: true, Solid: 31}
	if !m.MoverNearMotion(live, Vec3{45, -50, 24}, Vec3{45, 50, 24}) {
		t.Fatal("closing travel ignored")
	}
	if m.MoverNearMotion(live, Vec3{-100, -50, 24}, Vec3{-100, 50, 24}) {
		t.Fatal("distant door blocks unrelated path")
	}
	live.Solid = 0
	if m.MoverNearMotion(live, Vec3{45, -50, 24}, Vec3{45, 50, 24}) {
		t.Fatal("nonsolid mover treated as blocker")
	}
}

func TestMoverEnvelopeBoundsTrainLoop(t *testing.T) {
	m := &MapInfo{Models: []BSPModel{{}, {Min: Vec3{-16, -16, 0}, Max: Vec3{16, 16, 32}}}, Entities: []MapEntity{
		{Class: "func_train", Model: 1, Target: "a"}, {Class: "path_corner", TargetName: "a", Target: "b", Origin: Vec3{100, 100, 0}},
		{Class: "path_corner", TargetName: "b", Target: "a", Origin: Vec3{200, 100, 0}}}}
	if !m.MoverNearMotion(Mover{Model: 1}, Vec3{150, 50, 24}, Vec3{150, 150, 24}) {
		t.Fatal("train travel crossed without guard")
	}
	if m.MoverNearMotion(Mover{Model: 1}, Vec3{-1000, 0, 24}, Vec3{-900, 0, 24}) {
		t.Fatal("finite train loop blocks distant path")
	}
}
