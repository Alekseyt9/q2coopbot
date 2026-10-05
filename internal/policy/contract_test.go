package policy

import (
	"math"
	"testing"

	"q2coopbot/internal/quake"
)

func testObservation() Observation {
	return Observation{Version: ObservationVersion, Identity: Identity{Map: "base1", Connection: 2, Spawncount: 3, Actor: 1, Frame: 18}, Health: 100}
}

func TestCommandRejectsCrossSessionAndInvalidActions(t *testing.T) {
	o := testObservation()
	a := Action{Version: ActionVersion, Identity: o.Identity, Vertical: "release"}
	cases := []struct {
		name   string
		mutate func(*Action)
	}{
		{"frame", func(a *Action) { a.Identity.Frame-- }},
		{"connection", func(a *Action) { a.Identity.Connection++ }},
		{"map", func(a *Action) { a.Identity.Map = "base2" }},
		{"spawncount", func(a *Action) { a.Identity.Spawncount++ }},
		{"actor", func(a *Action) { a.Identity.Actor++ }},
		{"version", func(a *Action) { a.Version = "v2" }},
		{"nan", func(a *Action) { a.Forward = math.NaN() }},
		{"inf", func(a *Action) { a.YawDelta = math.Inf(1) }},
		{"range", func(a *Action) { a.Side = 1.1 }},
		{"pose", func(a *Action) { a.Vertical = "fly" }},
		{"weapon injection", func(a *Action) { a.Weapon = "Blaster;quit" }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := a
			tc.mutate(&b)
			if _, err := Command(o, b, [3]int16{}); err == nil {
				t.Fatal("accepted invalid action")
			}
		})
	}
}

func TestIndependentControlAndJumpRelease(t *testing.T) {
	o := testObservation()
	o.ViewAngles[1] = 30000
	deltas := [3]int16{1200, -9000, 0}
	a := Action{Version: ActionVersion, Identity: o.Identity, Forward: -.25, Side: .75, YawDelta: 90, PitchDelta: 15, Attack: true, Vertical: "jump"}
	cmd, err := Command(o, a, deltas)
	if err != nil {
		t.Fatal(err)
	}
	if cmd.Forward != -100 || cmd.Side != 300 || cmd.Up != 400 || cmd.Buttons != 1 || cmd.Msec != 100 {
		t.Fatalf("independent control lost: %+v", cmd)
	}
	back := FromCommand(o, cmd, deltas, "")
	if math.Abs(back.YawDelta-a.YawDelta) > .01 || math.Abs(back.PitchDelta-a.PitchDelta) > .01 {
		t.Fatalf("angle wrap/delta wrong: %+v", back)
	}
	a.Vertical = "release"
	a.Attack = false
	cmd, err = Command(o, a, deltas)
	if err != nil || cmd.Up != 0 || cmd.Buttons != 0 {
		t.Fatalf("jump/attack release lost: %+v %v", cmd, err)
	}
	a.Vertical = "crouch"
	cmd, err = Command(o, a, deltas)
	if err != nil || cmd.Up != -400 {
		t.Fatalf("crouch lost: %+v %v", cmd, err)
	}
}

func TestStaleLifeAndProtocolMetadata(t *testing.T) {
	o := testObservation()
	a := Action{Version: ActionVersion, Identity: o.Identity, Vertical: "release"}
	o.AgeMS = 301
	if _, err := Command(o, a, [3]int16{}); err == nil {
		t.Fatal("stale observation accepted")
	}
	o.AgeMS = 0
	o.Health = 0
	if _, err := Command(o, a, [3]int16{}); err == nil {
		t.Fatal("dead life accepted")
	}
	x, y := quake.UserCmd{Msec: 50}, quake.UserCmd{Msec: 100, Light: 80}
	if ControlChanged(x, y) {
		t.Fatal("protocol timing/light marked as tactical intervention")
	}
	y.Buttons = 1
	if !ControlChanged(x, y) {
		t.Fatal("attack intervention missing")
	}
}

func TestObservationMasksHiddenEnemiesAndCopiesInventory(t *testing.T) {
	clear, blocked := true, false
	s := quake.Snapshot{Health: 100, Self: quake.Vec3{10, 20, 30}, Enemies: []quake.Object{
		{ID: 1, Origin: quake.Vec3{15, 28, 30}, ClearShot: &clear},
		{ID: 2, Origin: quake.Vec3{999, 999, 999}, ClearShot: &blocked},
		{ID: 3, Origin: quake.Vec3{777, 777, 777}},
	}}
	o := Observe(s, testObservation().Identity, quake.UserCmd{})
	if o.Inventory != nil || o.InventoryAgeFrames != nil || o.Teammate != nil || len(o.Enemies) != 1 || o.Enemies[0].Relative != (quake.Vec3{5, 8, 0}) {
		t.Fatalf("observation leaked unknown/hidden data: %+v", o)
	}
	s.InventoryKnown = true
	s.Inventory = []quake.InventoryItem{{Name: "Railgun", Count: 1}}
	o = Observe(s, testObservation().Identity, quake.UserCmd{})
	s.Inventory[0].Count = 0
	if (*o.Inventory)[0].Count != 1 {
		t.Fatal("inventory aliases mutable snapshot")
	}
	a := Action{Version: ActionVersion, Identity: o.Identity, Vertical: "release", Weapon: "Railgun"}
	if _, err := Command(o, a, [3]int16{}); err != nil {
		t.Fatal(err)
	}
	*o.InventoryAgeFrames = 21
	if _, err := Command(o, a, [3]int16{}); err == nil {
		t.Fatal("stale weapon inventory accepted")
	}
}
