package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestRecordedBase2CrouchLaunchSafety(t *testing.T) {
	s := quake.Snapshot{Health: 100, Weapon: "Blaster", Self: quake.Vec3{516.625, 2531.75, -231.875}, SelfVelocity: quake.Vec3{-1.875, -38.375, 0}, DeltaAngles: [3]int16{0, -32768, 0}, Barrels: []quake.Object{{ID: 324, Origin: quake.Vec3{480, 2448, -255.875}}, {ID: 325, Origin: quake.Vec3{544.25, 2494.125, -255.875}}}}
	mate := quake.Vec3{472.375, 2522.75, -231.875}
	s.Teammate = &mate
	cmd := quake.UserCmd{Pitch: -2593, Yaw: 9046, Forward: -87, Side: -361, Up: -400, Buttons: 1, Msec: 100}
	p := Planner{}
	if got := p.guardBarrelShot(s, cmd); got.Buttons&1 != 0 || got.Forward != cmd.Forward || got.Up != cmd.Up {
		t.Fatal("recorded barrel launch allowed or movement changed", got)
	}
	s.Self = quake.Vec3{657.75, 2515.625, -231.875}
	mate = quake.Vec3{644.875, 2455.375, -231.875}
	cmd = quake.UserCmd{Pitch: -2929, Yaw: 13635, Forward: -167, Side: -70, Up: -400, Buttons: 1, Msec: 100}
	if got := p.guardBoltTeammate(s, cmd); got.Buttons&1 != 0 || got.Forward != cmd.Forward || got.Up != cmd.Up {
		t.Fatal("recorded partner launch allowed or movement changed", got)
	}
	mate = quake.Vec3{1000, 3000, -231.875}
	if got := p.guardBoltTeammate(s, cmd); got.Buttons&1 == 0 {
		t.Fatal("far off-axis teammate blocks all fire")
	}
}

func TestBoltLaunchCommandPosture(t *testing.T) {
	s := quake.Snapshot{Self: quake.Vec3{0, 0, 24}}
	_, starts, _ := boltLaunchPaths(s, quake.UserCmd{Up: -400})
	if len(starts) != 1 || starts[0] != (quake.Vec3{24, -8, 14}) {
		t.Fatal(starts)
	}
	s.Ducked = true
	_, starts, _ = boltLaunchPaths(s, quake.UserCmd{})
	if len(starts) != 2 || starts[1] != (quake.Vec3{24, -8, 38}) {
		t.Fatal(starts)
	}
}
