package bot

import (
	"encoding/json"
	"math"
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestPmoveGroundObservedDoorMatchesNativeFrame(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires BSP assets")
	}
	var recorded struct {
		Snapshot     quake.Snapshot
		Command      quake.UserCmd
		ObservedNext quake.Vec3 `json:"observed_next"`
	}
	b, err := os.ReadFile("testdata/pmove-jail1-door-frame194.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &recorded); err != nil {
		t.Fatal(err)
	}
	g, err := quake.LoadMap(root, recorded.Snapshot.Map)
	if err != nil {
		t.Fatal(err)
	}
	p := predictPmoveGround(recorded.Snapshot, recorded.Command, &g)
	if p == nil {
		t.Fatal("door prediction missing")
	}
	distance := func(a, b quake.Vec3) float64 {
		return math.Sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]))
	}
	if e := distance(p.Position, recorded.ObservedNext); e > .25 {
		t.Fatal("observed door mismatch", e, p.Position, recorded.ObservedNext)
	}
	withoutDoor := recorded.Snapshot
	withoutDoor.Movers = nil
	old := predictPmoveGround(withoutDoor, recorded.Command, &g)
	if old == nil || distance(old.Position, recorded.ObservedNext) < 20 {
		t.Fatal("fixture no longer distinguishes ignored door", old)
	}
}

func TestPmoveGroundRecordedVisibleBodies(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires BSP assets")
	}
	var fixtures struct {
		Cases []struct {
			Snapshot     quake.Snapshot
			Command      quake.UserCmd
			ObservedNext quake.Vec3 `json:"observed_next"`
			Frame        int
		}
	}
	b, err := os.ReadFile("testdata/pmove-observed-body-native-regressions.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures.Cases) != 29 {
		t.Fatal("regression cases lost")
	}
	maps := map[string]quake.MapInfo{}
	valid, close, primaryClose := 0, 0, 0
	maxError := 0.0
	for _, c := range fixtures.Cases {
		g, exists := maps[c.Snapshot.Map]
		if !exists {
			g, err = quake.LoadMap(root, c.Snapshot.Map)
			if err != nil {
				t.Fatal(err)
			}
			maps[c.Snapshot.Map] = g
		}
		p := predictPmoveGround(c.Snapshot, c.Command, &g)
		if p == nil {
			if c.Snapshot.Map != "ware2" || (c.Frame != 109 && c.Frame != 261) {
				t.Fatal("new unsupported regression", c.Snapshot.Map, c.Frame)
			}
			t.Logf("unsupported: map=%s frame=%d", c.Snapshot.Map, c.Frame)
			continue
		}
		valid++
		dx, dy, dz := p.Position[0]-c.ObservedNext[0], p.Position[1]-c.ObservedNext[1], p.Position[2]-c.ObservedNext[2]
		e := math.Sqrt(dx*dx + dy*dy + dz*dz)
		if e <= .25 {
			primaryClose++
		}
		if p.SupportAmbiguous && p.Alternative != nil {
			a := p.Alternative.Position
			x, y, z := a[0]-c.ObservedNext[0], a[1]-c.ObservedNext[1], a[2]-c.ObservedNext[2]
			e = math.Min(e, math.Sqrt(x*x+y*y+z*z))
		}
		if e > .25 {
			if c.Snapshot.Map != "city1" || (c.Frame != 107 && c.Frame != 108) {
				t.Fatal("new native body mismatch", c.Snapshot.Map, c.Frame, e)
			}
			t.Logf("remaining error: map=%s frame=%d error=%g", c.Snapshot.Map, c.Frame, e)
		}
		if e <= .25 {
			close++
		}
		maxError = math.Max(maxError, e)
	}
	t.Logf("body regressions: predicted=%d primary_within0.25=%d covered_by_supported_trajectories_within0.25=%d max_min_error=%g", valid, primaryClose, close, maxError)
	if close != 26 || valid != 26 || primaryClose != 24 {
		t.Fatal("native body collisions remain unmodelled", valid, close, maxError)
	}
}

func TestPmoveGroundUsesVerticalSpeedBeforeFriction(t *testing.T) {
	c := policyClient(t, "learned")
	s := c.planner.World.Snapshot
	s.PMStateKnown = true
	s.Gravity = 800
	s.SelfVelocity = quake.Vec3{50, 0, 80}
	p := predictPmoveGround(s, quake.UserCmd{Msec: 100}, c.planner.World.Geometry)
	expected := 50 * (1 - 60/math.Sqrt(50*50+80*80))
	if p == nil || math.Abs(p.Velocity[0]-expected) > 1e-8 || p.Velocity[2] != 0 {
		t.Fatal("3D friction lost", p, expected)
	}
	for _, mutate := range []func(*quake.Snapshot){
		func(s *quake.Snapshot) { s.PMTime = 7 }, func(s *quake.Snapshot) { s.PMType = 2 },
		func(s *quake.Snapshot) { s.PMStateKnown = false }, func(s *quake.Snapshot) { s.OnGround = false },
	} {
		bad := s
		mutate(&bad)
		if predictPmoveGround(bad, quake.UserCmd{Msec: 100}, c.planner.World.Geometry) != nil {
			t.Fatal("unsupported state predicted", bad)
		}
	}
}

func TestPmoveGroundRetainsRecordedWallTrajectory(t *testing.T) {
	c := policyClient(t, "learned")
	s := c.planner.World.Snapshot
	s.Self = quake.Vec3{-47.875, -223.75, 24.125}
	s.PMStateKnown = true
	s.Gravity = 800
	s.DeltaAngles = [3]int16{0, 24576, 0}
	cmd := quake.UserCmd{Pitch: -703, Yaw: -23402, Forward: -141, Side: 316, Msec: 100}
	p := predictPmoveGround(s, cmd, c.planner.World.Geometry)
	if p == nil || len(p.Contacts) == 0 || math.Hypot(p.Position[0]-s.Self[0], p.Position[1]-s.Self[1]) < .125 {
		t.Fatal("native wall slide missing", p)
	}
}
