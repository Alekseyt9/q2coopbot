package quake

import "testing"

func TestUnitConditionsUseFlagsNotCrossMapTargetNames(t *testing.T) {
	locked := &MapInfo{Name: "locked", Models: make([]BSPModel, 2), Entities: []MapEntity{
		{Class: "func_door", Model: 1, TargetName: "door"},
		{Class: "target_crosslevel_target", SpawnFlags: 3, Target: "relay"},
		{Class: "trigger_relay", TargetName: "relay", Target: "door"},
	}}
	remote := &MapInfo{Name: "remote", Models: make([]BSPModel, 3), Entities: []MapEntity{
		{Class: "func_button", Model: 1, Target: "set_flag"},
		{Class: "target_crosslevel_trigger", TargetName: "set_flag", SpawnFlags: 1},
		{Class: "trigger_once", Model: 2, Target: "door"}, // unrelated local name
	}}
	conditions := locked.UnitDoorConditions(1)
	if len(conditions) != 1 || conditions[0].RequiredFlags != 3 || conditions[0].FlagState != "unknown" || len(conditions[0].Chain) != 2 {
		t.Fatal(conditions)
	}
	actions := remote.UnitActions(3)
	if len(actions) != 1 || actions[0].Map != "remote" || actions[0].SetsFlags != 1 || actions[0].Action != "touch" || len(actions[0].Activation.Chain) != 2 {
		t.Fatal("one action was mistaken for satisfying both flags", actions)
	}
	remote.Entities[0].Health = 1
	if remote.UnitActions(3)[0].Action != "shoot" {
		t.Fatal("shootable button classified as touch")
	}
	remote.Entities[0].TargetName = "remote_only"
	if len(remote.UnitActions(3)) != 0 {
		t.Fatal("remotely operated button offered as physical action")
	}
	locked.Entities[1].SpawnFlags = 256
	if len(locked.UnitDoorConditions(1)) != 0 {
		t.Fatal("unmatchable native flag mask accepted")
	}
	locked.Entities[1].SpawnFlags = 0
	if len(locked.UnitDoorConditions(1)) != 0 {
		t.Fatal("unconditional load-time action treated as remote requirement")
	}
}

func TestUnitTouchActionRelayAndCycle(t *testing.T) {
	m := &MapInfo{Name: "a", Models: make([]BSPModel, 2), Entities: []MapEntity{
		{Class: "trigger_once", Model: 1, Target: "relay"},
		{Class: "trigger_relay", TargetName: "relay", Target: "setter"},
		{Class: "target_crosslevel_trigger", TargetName: "setter", SpawnFlags: 2},
	}}
	if got := m.UnitActions(3); len(got) != 1 || len(got[0].Activation.Chain) != 3 {
		t.Fatal(got)
	}
	if len(m.UnitActions(1)) != 0 {
		t.Fatal("unrelated flag offered")
	}
	m.Entities[1].Target = "relay"
	if len(m.UnitActions(3)) != 0 {
		t.Fatal("cycle accepted")
	}
	m.Entities[1].Target = "setter"
	m.Entities[0].SpawnFlags = 2
	if len(m.UnitActions(3)) != 0 {
		t.Fatal("monster-only trigger offered")
	}
}

func TestUnitMapPathsKeepRevisitsButRespectUnitBoundary(t *testing.T) {
	makeMap := func(name string, destinations ...string) *MapInfo {
		info := &MapInfo{Name: name, Models: []BSPModel{{}, {Min: Vec3{}, Max: Vec3{10, 10, 10}}}}
		for _, destination := range destinations {
			info.Entities = append(info.Entities, MapEntity{Class: "trigger_changelevel", Model: 1, Map: destination})
		}
		return info
	}
	a := makeMap("a", "b$from_a", "*c$start")
	b := makeMap("b", "a$return", "missing")
	c := makeMap("c", "d")
	d := makeMap("d")
	paths := UnitMapPaths("a", []*MapInfo{a, b, c, d})
	if len(paths) != 2 || len(paths["b"]) != 2 || paths["b"][0] != "a" || paths["b"][1] != "b" {
		t.Fatal("unit reset or missing maps crossed", paths)
	}
}
