package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func unitTripPlanner() (*Planner, quake.Snapshot) {
	a := quake.UnitAction{Map: "remote", Action: "touch", SetsFlags: 3, TravelMaps: []string{"home", "remote"}, ReturnMaps: []string{"remote", "home"}, Activation: quake.Activation{Trigger: quake.MapEntity{Model: 1}}}
	d := &CampaignDependency{State: "unit_activation_required", DoorModel: 2, UnitConditions: []quake.UnitCondition{{RequiredFlags: 3, FlagState: "unknown", Activations: []quake.UnitAction{a}}}}
	p := &Planner{Campaign: true, CampaignRoute: []string{"home", "finish"}, campaignMap: "home", campaignDestination: "finish", campaignDependency: d}
	return p, quake.Snapshot{Map: "home", Frame: 100, Health: 100}
}

func TestUnitActivationRequiresDistinctFramesAndLocalTouch(t *testing.T) {
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
	p.campaignUnitTrip.Stack = p.campaignUnitTrip.Stack[:3]
	p.campaignUnitTrip.Stack[2].Action.Activation.Trigger.Model = 16
	s.Map = "remote"
	s.Frame = 1
	s.Self = quake.Vec3{128, -320, 24.125}
	d := &CampaignDecision{}
	p.campaignUnitGoal(s, d)
	for i := 0; i < 10; i++ {
		p.campaignUnitGoal(s, d)
	}
	if p.campaignUnitTrip.Attempted {
		t.Fatal("duplicate snapshot completed touch")
	}
	s.Frame = 2
	p.campaignUnitGoal(s, d)
	s.Frame = 3
	_, _, handled, next := p.campaignUnitGoal(s, d)
	if !p.campaignUnitTrip.Attempted || handled || next != "home" {
		t.Fatal("touch did not start return")
	}
}

func TestUnitTripSuspendsCampaignAndRequiresObservedEffect(t *testing.T) {
	p, s := unitTripPlanner()
	if !p.startCampaignUnitTrip(s) {
		t.Fatal("no trip")
	}
	d := &CampaignDecision{}
	goal, ok, handled, next := p.campaignUnitGoal(s, d)
	if ok || handled || next != "remote" || goal != (quake.Vec3{}) || p.campaignRouteIndex != 0 {
		t.Fatal("outbound mixed with campaign progress", d)
	}
	trip := p.campaignUnitTrip
	// Travel is driven only by observed map identity, never local coordinates.
	s.Map = "remote"
	s.Frame = 1
	trip.Stack = trip.Stack[:3]
	trip.Stack[2].Kind = "travel_outbound"
	trip.Stack[2].Path = []string{"home", "remote"}
	// Focus return/effect ownership separately from BSP contact geometry.
	trip.Stack = trip.Stack[:2]
	trip.Attempted = true
	_, _, handled, next = p.campaignUnitGoal(s, d)
	if handled || next != "home" || p.campaignRouteIndex != 0 {
		t.Fatal("return itinerary lost")
	}
	s.Map = "home"
	s.Frame = 1
	_, ok, handled, _ = p.campaignUnitGoal(s, d)
	if ok || !handled || trip.State != "verify_effect" {
		t.Fatal("contact was treated as effect")
	}
	if trip.Dependency.UnitConditions[0].FlagState != "unknown" {
		t.Fatal("invented server flags")
	}
	s.Movers = []quake.Mover{{Model: 99, Origin: quake.Vec3{0, 0, 100}}}
	s.Frame = 2
	p.campaignUnitGoal(s, d)
	if p.campaignUnitTrip == nil {
		t.Fatal("unrelated mover confirmed door")
	}
	s.Movers = []quake.Mover{{Model: 2, Origin: quake.Vec3{0, 0, 100}}}
	s.Frame = 3
	_, _, handled, next = p.campaignUnitGoal(s, d)
	if handled || next != "" || p.campaignUnitTrip != nil || trip.State != "effect_confirmed" || p.campaignRouteIndex != 0 || p.campaignDestination != "finish" {
		t.Fatal("original objective not resumed")
	}
}

func TestUnitTripRejectsPartialMasksOneWayTravelAndUnexpectedMaps(t *testing.T) {
	for _, change := range []func(*quake.UnitAction){func(a *quake.UnitAction) { a.SetsFlags = 1 }, func(a *quake.UnitAction) { a.ReturnMaps = nil }, func(a *quake.UnitAction) { a.Action = "shoot" }} {
		p, s := unitTripPlanner()
		change(&p.campaignDependency.UnitConditions[0].Activations[0])
		if p.startCampaignUnitTrip(s) {
			t.Fatal("unsupported action started")
		}
	}
	p, s := unitTripPlanner()
	p.startCampaignUnitTrip(s)
	s.Map = "unexpected"
	d := &CampaignDecision{}
	_, ok, handled, _ := p.campaignUnitGoal(s, d)
	if ok || !handled || d.State != "unexpected_unit_map" {
		t.Fatal("unexpected map authorized")
	}
	if _, err := p.CaptureCheckpoint(); err == nil {
		t.Fatal("active itinerary silently omitted from save")
	}
}
