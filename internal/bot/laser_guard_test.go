package bot

import (
	"os"
	"testing"

	"q2coopbot/internal/quake"
)

func TestBase3LaserGuardBrakesBeforeLethalBeam(t *testing.T) {
	root := "../../workspace/runtime/q2go/baseq2"
	if _, err := os.Stat(root + "/maps/base3.aas"); err != nil {
		t.Skip("local base3 assets unavailable")
	}
	geometry, err := quake.LoadMap(root, "base3")
	if err != nil {
		t.Fatal(err)
	}
	if !geometry.HasStaticLethalLasers() {
		t.Fatal("base3 lethal laser missing")
	}
	s := quake.Snapshot{Map: "base3", Frame: 131, Self: quake.Vec3{-736, -457.5, -279.875}, SelfVelocity: quake.Vec3{132.5, 56.75, 0}, Health: 100, OnGround: true}
	s.DeltaAngles[1] = -32768
	p := &Planner{World: World{Geometry: &geometry}}
	cmd := quake.UserCmd{Yaw: -16384, Forward: 400, Msec: 50}
	if !p.laserCommandUnsafe(s, cmd) {
		t.Fatal("original command did not detect beam crossing after momentum")
	}
	guarded := p.limitLaserMovement(s, cmd)
	if guarded.Forward != 0 || guarded.Side != 0 || guarded.Up != 0 || p.World.Command.MoveLimitReason != "static_laser_hazard" {
		t.Fatalf("unsafe command not stopped: %+v %+v", guarded, p.World.Command)
	}
	if p.laserCommandUnsafe(s, guarded) {
		t.Fatal("neutral stop also reaches beam; braking must begin earlier")
	}
}
