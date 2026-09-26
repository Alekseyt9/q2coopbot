package bot

import (
	"math"
	"testing"
	"time"

	"q2coopbot/internal/quake"
)

func TestWorldMoveMatchesDryPmoveWhileAiming(t *testing.T) {
	for _, pitch := range []float64{-89, -60, 0, 60, 89} {
		for _, yaw := range []float64{-170, 0, 75} {
			s := quake.Snapshot{DeltaAngles: [3]int16{1234, -2345, 0}}
			cmd := quake.UserCmd{Pitch: int16(pitch*65536/360) - s.DeltaAngles[0], Yaw: int16(yaw*65536/360) - s.DeltaAngles[1], Buttons: 1}
			got := worldMove(cmd, s, 3, 4, 120, true)
			// Reconstruct the server's PM_AirMove XY wish velocity, which
			// divides view pitch by three before forming its forward basis.
			p := float64(int16(uint16(got.Pitch)+uint16(s.DeltaAngles[0]))) * 2 * math.Pi / 65536 / 3
			y := float64(int16(uint16(got.Yaw)+uint16(s.DeltaAngles[1]))) * 2 * math.Pi / 65536
			vx := math.Cos(p)*math.Cos(y)*float64(got.Forward) + math.Sin(y)*float64(got.Side)
			vy := math.Cos(p)*math.Sin(y)*float64(got.Forward) - math.Cos(y)*float64(got.Side)
			if math.Hypot(vx-72, vy-96) > 1 || got.Pitch != cmd.Pitch || got.Yaw != cmd.Yaw || got.Buttons != 1 {
				t.Fatalf("pitch=%g yaw=%g: velocity=(%g,%g), command=%+v", pitch, yaw, vx, vy, got)
			}
		}
	}
}

func TestRouteBrakesForDescentButKeepsGapApproachSpeed(t *testing.T) {
	for _, descent := range []bool{false, true} {
		route := []quake.Waypoint{{Position: quake.Vec3{100, 0, 240}, Kind: 7, ToArea: 2}}
		want := int16(400)
		if descent {
			route = append(route, quake.Waypoint{Position: quake.Vec3{100, 0, 24}, Kind: 7, ToArea: 2})
			want = 120
		}
		now := time.Now()
		p := &Planner{hasGoal: true, goalPoint: quake.Vec3{200, 0, 240}, World: World{
			Map: "test", Updated: now, Goal: "follow_teammate", Navigation: "ready", Route: route,
			Snapshot: quake.Snapshot{Frame: 1, Health: 100, OnGround: true, Self: quake.Vec3{0, 0, 240}},
		}}
		cmd := p.commandAt(quake.UserCmd{}, now)
		if cmd.Forward != want || cmd.Up != 0 {
			t.Fatalf("descent=%v: got %+v, want forward=%d without jumping", descent, cmd, want)
		}
	}
}
