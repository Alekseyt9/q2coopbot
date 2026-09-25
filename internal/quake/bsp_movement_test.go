package quake

import "testing"

func testBoxBrush(mins, maxs Vec3, first int) ([]bspPlane, bspBrush) {
	planes := []bspPlane{
		{Vec3{1, 0, 0}, maxs[0]}, {Vec3{-1, 0, 0}, -mins[0]},
		{Vec3{0, 1, 0}, maxs[1]}, {Vec3{0, -1, 0}, -mins[1]},
		{Vec3{0, 0, 1}, maxs[2]}, {Vec3{0, 0, -1}, -mins[2]},
	}
	return planes, bspBrush{first: first, count: len(planes), contents: 1}
}

func TestPlayerMoveClearAndGroundDrop(t *testing.T) {
	floor, floorBrush := testBoxBrush(Vec3{-100, -100, -20}, Vec3{100, 100, 0}, 0)
	wall, wallBrush := testBoxBrush(Vec3{40, -100, 0}, Vec3{60, 100, 100}, len(floor))
	c := &CollisionMap{planes: append(floor, wall...), brushes: []bspBrush{floorBrush, wallBrush}, worldBrushes: []int{0, 1}}
	for i := range c.planes {
		c.sides = append(c.sides, uint16(i))
	}
	m := &MapInfo{collision: c}
	if !m.PlayerMoveClear(Vec3{0, 0, 24}, Vec3{20, 0, 24}) {
		t.Fatal("clear movement over floor was blocked")
	}
	if m.PlayerMoveClear(Vec3{0, 0, 24}, Vec3{40, 0, 24}) {
		t.Fatal("standing player hull crossed a wall")
	}
	if drop, ok := m.GroundDrop(Vec3{20, 0, 24}, 24); !ok || drop > 1 {
		t.Fatalf("floor support missing: drop=%f ok=%t", drop, ok)
	}
	if _, ok := m.GroundDrop(Vec3{120, 0, 24}, 24); ok {
		t.Fatal("unsupported position was marked grounded")
	}
	if got := m.GroundMoveHazard(nil, Vec3{0, 0, 24}, 40, 0); got != "static_hull_blocked" {
		t.Fatalf("wall hazard=%q", got)
	}
	if got := m.GroundMoveHazardStep(nil, Vec3{0, 0, 24}, 40, 0, 8); got != "" {
		t.Fatalf("slow approach was blocked early: %q", got)
	}
	if got := m.GroundMoveHazard(nil, Vec3{90, 0, 24}, 40, 0); got != "no_ground_support" {
		t.Fatalf("edge hazard=%q", got)
	}
	if got := m.GroundMoveHazard(nil, Vec3{90, 0, 24}, 1, 0); got != "no_ground_support" {
		t.Fatalf("full-speed detour edge hazard=%q", got)
	}
	n := &Navigator{Areas: []Area{{}, {Min: Vec3{110, -20, 0}, Max: Vec3{140, 20, 40}, Flags: 1}}}
	if got := m.GroundMoveHazard(n, Vec3{90, 0, 24}, 40, 0); got != "" {
		t.Fatalf("AAS grounded support was ignored: %q", got)
	}
}

func TestGroundedNearDoesNotTrustDistantRouteFallback(t *testing.T) {
	n := &Navigator{Areas: []Area{{}, {Min: Vec3{0, 0, 0}, Max: Vec3{20, 20, 40}, Flags: 1}}}
	if !n.GroundedNear(Vec3{21, 10, 24}) {
		t.Fatal("one-unit AAS rounding gap was rejected")
	}
	if n.GroundedNear(Vec3{30, 10, 24}) {
		t.Fatal("distant grounded area was used as support")
	}
}

func TestDoorMoveHazardUsesObservedTranslatingDoor(t *testing.T) {
	m := &MapInfo{
		Models:   []BSPModel{{}, {Min: Vec3{80, -252, -16}, Max: Vec3{112, -236, 112}}},
		Entities: []MapEntity{{Class: "func_door", Model: 1}},
	}
	start := Vec3{96, -300, 24}
	closed := []Mover{{Model: 1, Origin: Vec3{}}}
	if got := m.DoorMoveHazard(closed, start, 0, 140); got != "dynamic_door_blocked" {
		t.Fatalf("closed door hazard=%q", got)
	}
	if model, reason := m.DoorMoveBlock(closed, start, 0, 140); model != 1 || reason != "dynamic_door_blocked" {
		t.Fatalf("closed door model=%d reason=%q", model, reason)
	}
	if got := m.DoorMoveHazard(closed, start, 0, 1); got != "dynamic_door_blocked" {
		t.Fatalf("full-speed detour door hazard=%q", got)
	}
	if got := m.DoorMoveHazard(closed, Vec3{96, -260, 24}, 0, -140); got != "" {
		t.Fatalf("movement out of a door overlap was blocked: %q", got)
	}
	if !m.DoorShotBlocked(closed, Vec3{96, -300, 30}, Vec3{96, -96, 30}) {
		t.Fatal("closed door did not block line of fire")
	}
	if got := m.DoorMoveHazard(nil, start, 0, 140); got != "dynamic_door_unobserved" {
		t.Fatalf("unobserved door hazard=%q", got)
	}
	if !m.DoorShotBlocked(nil, Vec3{96, -300, 30}, Vec3{96, -96, 30}) {
		t.Fatal("unobserved door was treated as a clear shot")
	}
	if got := m.DoorMoveHazard(nil, Vec3{200, -300, 24}, 0, 140); got != "" {
		t.Fatalf("unobserved distant door blocked unrelated movement: %q", got)
	}
	open := []Mover{{Model: 1, Origin: Vec3{200, 0, 0}}}
	if got := m.DoorMoveHazard(open, start, 0, 140); got != "" {
		t.Fatalf("moved door hazard=%q", got)
	}
	if m.DoorShotBlocked(open, Vec3{96, -300, 30}, Vec3{96, -96, 30}) {
		t.Fatal("moved door still blocked line of fire")
	}
	m.Entities[0].Class = "func_door_rotating"
	if got := m.DoorMoveHazard(closed, start, 0, 140); got != "" {
		t.Fatalf("rotating door was treated as translating: %q", got)
	}
}

func TestMovementCompleteRequiresBrushModels(t *testing.T) {
	m := &MapInfo{collision: &CollisionMap{planes: []bspPlane{{}}, sides: []uint16{0},
		brushes: []bspBrush{{}}, worldBrushes: []int{0}}, Models: []BSPModel{{}, {}},
		Entities: []MapEntity{{Class: "func_door", Model: 1}}}
	if !m.MovementComplete() {
		t.Fatal("complete BSP was rejected")
	}
	m.collision.worldBrushes = nil
	if m.MovementComplete() {
		t.Fatal("missing static world brushes were accepted")
	}
	m.collision.worldBrushes = []int{0}
	m.Entities[0].Model = 2
	if m.MovementComplete() {
		t.Fatal("out-of-range door model was accepted")
	}
	m.Entities[0].Model = 1
	m.Models = nil
	if m.MovementComplete() {
		t.Fatal("missing model lump was accepted")
	}
}

func TestButtonForDoorChecksLinkAndActivation(t *testing.T) {
	m := &MapInfo{Models: make([]BSPModel, 4), Entities: []MapEntity{
		{Class: "func_door", Model: 1, TargetName: "gate"},
		{Class: "func_button", Model: 2, Target: "gate"},
	}}
	button, action, ok := m.ButtonForDoor(1)
	if !ok || button.Model != 2 || action != "touch" {
		t.Fatalf("touch button lookup=%+v %q %t", button, action, ok)
	}
	m.Entities[1].Health = 10
	_, action, ok = m.ButtonForDoor(1)
	if !ok || action != "shoot" {
		t.Fatalf("shoot button lookup=%q %t", action, ok)
	}
	m.Entities[1].Health = 0
	m.Entities[1].TargetName = "remote"
	if _, _, ok := m.ButtonForDoor(1); ok {
		t.Fatal("remotely activated button was offered as a local action")
	}
}

func TestParsedButtonHealthSelectsShoot(t *testing.T) {
	entities := parseMapEntities(`{"classname" "func_door" "model" "*1" "targetname" "gate"}
{"classname" "func_button" "model" "*2" "target" "gate" "health" "20" "spawnflags" "8"}`)
	m := &MapInfo{Models: make([]BSPModel, 3), Entities: entities}
	button, action, ok := m.ButtonForDoor(1)
	if !ok || button.Model != 2 || button.SpawnFlags != 8 || action != "shoot" {
		t.Fatalf("parsed button action=%q button=%+v ok=%t", action, button, ok)
	}
}
