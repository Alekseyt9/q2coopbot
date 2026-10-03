package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestUnitDoorProbeRequiresNativeDisplacement(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires BSP/AAS")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base1.aas")
	if err != nil {
		t.Fatal(err)
	}
	p, s := unitTripPlanner()
	p.startCampaignUnitTrip(s)
	p.World.Geometry = &g
	p.Nav = n
	trip := p.campaignUnitTrip
	trip.State = "verify_effect"
	trip.DoorModel = 32
	trip.Dependency.initial = quake.Vec3{1792, -1792, -104}
	trip.Dependency.ProbeFrom = quake.Vec3{135, -304, 24.125}
	trip.Dependency.ProbeTo = quake.Vec3{95, -304, 24.125}
	s.Self = trip.Dependency.ProbeFrom
	s.OnGround = true
	if p.campaignUnitProbePassed(s) {
		t.Fatal("PVS absence confirmed effect")
	}
	cmd, active := p.campaignUnitProbeCommand(s, quake.UserCmd{Buttons: 1, Up: 200})
	if !active || cmd.Buttons != 0 || cmd.Up != 0 {
		t.Fatal("safe probe not isolated", cmd)
	}
	if p.campaignUnitProbePassed(s) {
		t.Fatal("command intent confirmed effect")
	}
	s.Self[0] = 111
	if !p.campaignUnitProbePassed(s) {
		t.Fatal("native displacement into old closed hull not accepted")
	}
	for _, mutate := range []func(*quake.Snapshot){
		func(s *quake.Snapshot) { s.Self[1] += 8 },
		func(s *quake.Snapshot) { s.Self[0] = 90 },
		func(s *quake.Snapshot) { s.OnGround = false },
		func(s *quake.Snapshot) { s.Health = 0 },
		func(s *quake.Snapshot) { s.Map = "remote" },
	} {
		bad := s
		mutate(&bad)
		if p.campaignUnitProbePassed(bad) {
			t.Fatal("unsafe displacement accepted", bad)
		}
	}
	s.Self = trip.Dependency.ProbeFrom
	s.Movers = []quake.Mover{{Model: 32, Origin: trip.Dependency.initial}}
	if _, active := p.campaignUnitProbeCommand(s, cmd); active {
		t.Fatal("observed blocker ignored")
	}
	s.Movers = nil
	trip.ProbeFrames = 13
	s.Self[0] = 111
	if p.campaignUnitProbePassed(s) {
		t.Fatal("late snapshot exceeded confirmation budget")
	}
	if _, active := p.campaignUnitProbeCommand(s, cmd); active {
		t.Fatal("probe exceeded frame budget")
	}
}
