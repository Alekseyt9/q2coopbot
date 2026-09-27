package bot

import (
	"os"
	"testing"
	"time"

	"q2coopbot/internal/quake"
)

func TestSetupHoldDoesNotAccumulateStuckRecovery(t *testing.T) {
	p := &Planner{GameClock: true, testSetupHold: true, routeKnown: true, failures: 4, detourUntil: time.Unix(1000, 0), World: World{Map: "test"}}
	s := quake.Snapshot{Map: "test", Frame: 140, Health: 74, Self: quake.Vec3{110, -243, 24}}
	p.update(s, "")
	if p.routeKnown || p.failures != 0 || !p.detourUntil.IsZero() || !p.lastProgress.Equal(p.navigationNow(140)) {
		t.Fatal("release inherits a route or a stuck recovery from setup")
	}
}

// Captured stalled frame 240 from episode-regressions-20260927-013504-479.
func TestBunk1OpenDoorShortWaypoint(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local bunk1 BSP")
	}
	g, err := quake.LoadMap(root, "bunk1")
	if err != nil {
		t.Fatal(err)
	}
	from := quake.Vec3{110.75, -243.125, 24.125}
	target := quake.Vec3{96, -242.89999389648438, 24}
	movers := []quake.Mover{{Model: 118, Origin: quake.Vec3{0, 0, -122}}, {Model: 125, Origin: quake.Vec3{-58, 0, 0}}, {Model: 135, Origin: quake.Vec3{58, 0, 0}}}
	for _, tc := range []struct {
		name            string
		closed, unknown bool
	}{{"open", false, false}, {"closed", true, false}, {"unknown", false, true}} {
		t.Run(tc.name, func(t *testing.T) {
			live := append([]quake.Mover(nil), movers...)
			if tc.closed {
				live[1].Origin = quake.Vec3{}
				live[2].Origin = quake.Vec3{}
			}
			if tc.unknown {
				live = live[:1]
			}
			// Closed doors overlap the captured open-door origin. Use a point
			// outside their expanded hull to exercise conservative approach.
			self, wp := from, target
			if tc.closed || tc.unknown {
				self = quake.Vec3{96, -268, 24.125}
				wp = quake.Vec3{96, -252, 24.125}
			}
			now := time.Now()
			p := &Planner{hasGoal: true, goalPoint: wp, World: World{Map: "bunk1", Geometry: &g, GeometryStatus: "ready", Navigation: "ready", Goal: "follow_teammate", Updated: now, Snapshot: quake.Snapshot{Map: "bunk1", Frame: 240, Self: self, OnGround: true, Health: 74, Movers: live}, Route: []quake.Waypoint{{Position: wp, Kind: 2}}}}
			cmd := p.commandAt(quake.UserCmd{}, now)
			moves := cmd.Forward != 0 || cmd.Side != 0
			if moves == (tc.closed || tc.unknown) {
				t.Fatalf("cmd=%+v decision=%+v", cmd, p.World.Command)
			}
			if moves && (cmd.Forward > 80 || cmd.Forward < -80 || cmd.Side > 80 || cmd.Side < -80 || cmd.Up != 0 || p.World.Command.MoveLimitReason != "door_short_approach") {
				t.Fatalf("short probe without slow grounded movement: %+v", cmd)
			}
		})
	}
}
