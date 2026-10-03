package quake

import "testing"

func TestTouchActivationTargetChain(t *testing.T) {
	m := &MapInfo{Models: make([]BSPModel, 4), Entities: []MapEntity{{Class: "func_door", Model: 1, TargetName: "door"}, {Class: "trigger_once", Model: 2, Target: "relay"}, {Class: "trigger_relay", TargetName: "relay", Target: "lever"}, {Class: "func_door_rotating", Model: 3, TargetName: "lever", Target: "door"}}}
	got := m.TouchActivationsForDoor(1)
	if len(got) != 1 || len(got[0].Chain) != 3 {
		t.Fatal(got)
	}
	m.Entities[3].Target = "relay"
	if len(m.TouchActivationsForDoor(1)) != 0 {
		t.Fatal("cycle accepted")
	}
	m.Entities[3].Target = "door"
	for _, flag := range []int{2, 4} {
		m.Entities[1].SpawnFlags = flag
		if len(m.TouchActivationsForDoor(1)) != 0 {
			t.Fatal("disabled or non-player trigger accepted")
		}
	}
	m.Entities[1].SpawnFlags = 0
	m.Entities[1].TargetName = "remote"
	if len(m.TouchActivationsForDoor(1)) != 0 {
		t.Fatal("remote trigger offered as local touch")
	}
}
