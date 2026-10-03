package bot

import (
	"encoding/json"
	"q2coopbot/internal/quake"
	"testing"
)

func TestUnitTripCheckpointRoundTripAndIsolation(t *testing.T) {
	for _, stage := range []string{"travel_outbound", "activate", "travel_return", "verify_effect"} {
		t.Run(stage, func(t *testing.T) {
			p, s := unitTripPlanner()
			p.CampaignUnitMaps = []string{"remote"}
			p.startCampaignUnitTrip(s)
			trip := p.campaignUnitTrip
			trip.Dependency.initial = quake.Vec3{10, 20, 30}
			trip.ElapsedFrames = 70
			count := map[string]int{"travel_outbound": 4, "activate": 3, "travel_return": 2, "verify_effect": 1}[stage]
			trip.Stack = trip.Stack[:count]
			trip.State = stage
			if count < 4 && count > 1 {
				s.Map = "remote"
			}
			if count <= 2 {
				trip.Attempted = true
			}
			if count == 3 {
				trip.Stack[2].contactFrames = 2
			}
			if count == 1 {
				trip.verifyStarted = 90
				trip.probing = true
				trip.ProbeFrames = 5
			}
			s.OnGround = true
			p.World.Snapshot = s
			p.World.Goal = "reach_level_exit"
			state, err := p.CaptureCheckpoint()
			if err != nil {
				t.Fatal(err)
			}
			data, err := json.Marshal(state)
			if err != nil {
				t.Fatal(err)
			}
			var decoded PlannerCheckpoint
			if err = json.Unmarshal(data, &decoded); err != nil {
				t.Fatal(err)
			}
			fresh := s
			fresh.Frame = 30
			fresh.Health = 73
			q := &Planner{Campaign: true, CampaignRoute: []string{"home", "finish"}, CampaignUnitMaps: []string{"remote"}}
			if err = q.RestoreCheckpoint(decoded, fresh, ""); err != nil {
				t.Fatal(err)
			}
			r := q.campaignUnitTrip
			if r == nil || r.State != stage || r.ElapsedFrames != 70 || r.Dependency.initial != trip.Dependency.initial || r.probing || r.lastFrame != 30 || q.campaignDestination != "finish" || q.campaignRouteIndex != 0 || q.World.Snapshot.Health != 73 {
				t.Fatal("restore lost intention or restored motor/native state", r)
			}
			if count == 3 && r.Stack[2].contactFrames != 0 {
				t.Fatal("old touch evidence restored")
			}
			if count == 1 && (r.verifyStarted != 20 || r.ProbeFrames != 5) {
				t.Fatal("verification budget refreshed")
			}
			r.Dependency.ProbeFrom[0] = 999
			if decoded.Campaign.UnitTrip.Trip.Dependency.ProbeFrom[0] == 999 || trip.Dependency.ProbeFrom[0] == 999 {
				t.Fatal("checkpoint aliases active planner")
			}
		})
	}
}

func TestUnitCheckpointRejectsMalformedStackAndConfig(t *testing.T) {
	p, s := unitTripPlanner()
	p.CampaignUnitMaps = []string{"remote"}
	p.startCampaignUnitTrip(s)
	s.OnGround = true
	p.World.Snapshot = s
	state, err := p.CaptureCheckpoint()
	if err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*PlannerCheckpoint){
		func(s *PlannerCheckpoint) { s.Campaign.UnitTrip.Trip.Stack[3].Leg = 20 },
		func(s *PlannerCheckpoint) { s.Campaign.UnitTrip.Trip.Stack[0].Kind = "activate" },
		func(s *PlannerCheckpoint) { s.Campaign.UnitTrip.Trip.Dependency = nil },
		func(s *PlannerCheckpoint) { s.Campaign.UnitTrip.Trip.ElapsedFrames = 1801 },
		func(s *PlannerCheckpoint) { s.Campaign.UnitTrip.Trip.EffectEvidence = "native_probe_passed" },
		func(s *PlannerCheckpoint) { s.Campaign.UnitMaps = []string{"../bad"} },
	} {
		data, _ := json.Marshal(state)
		var bad PlannerCheckpoint
		json.Unmarshal(data, &bad)
		mutate(&bad)
		if bad.validate(s.Map) == nil {
			t.Fatal("malformed unit checkpoint accepted")
		}
	}
	q := &Planner{Campaign: true, CampaignRoute: []string{"home", "finish"}}
	if q.RestoreCheckpoint(state, s, "") == nil {
		t.Fatal("different unit map catalog accepted")
	}
}

func TestUnitVerificationCheckpointZeroRebasedStart(t *testing.T) {
	p, s := unitTripPlanner()
	p.CampaignUnitMaps = []string{"remote"}
	p.startCampaignUnitTrip(s)
	trip := p.campaignUnitTrip
	trip.Stack = trip.Stack[:1]
	trip.State = "verify_effect"
	trip.Attempted = true
	trip.verifyStarted = 99
	s.OnGround = true
	p.World.Snapshot = s
	state, err := p.CaptureCheckpoint()
	if err != nil {
		t.Fatal(err)
	}
	fresh := s
	fresh.Frame = 1
	q := &Planner{Campaign: true, CampaignRoute: p.CampaignRoute, CampaignUnitMaps: p.CampaignUnitMaps}
	if err = q.RestoreCheckpoint(state, fresh, ""); err != nil {
		t.Fatal(err)
	}
	if q.campaignUnitTrip.verifyStarted != 0 || !q.campaignUnitTrip.verifyStartedSet {
		t.Fatal("rebased zero confused with unset timer")
	}
	fresh.Frame = 2
	q.campaignUnitGoal(fresh, &CampaignDecision{})
	if q.campaignUnitTrip.verifyStarted != 0 {
		t.Fatal("loading refreshed verification deadline")
	}
}
