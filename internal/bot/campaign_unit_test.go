package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestRemoteCampaignDependencyNeverUsesRemoteCoordinatesLocally(t *testing.T) {
	d := &CampaignDependency{State: "unit_activation_required", DoorModel: 1,
		UnitConditions: []quake.UnitCondition{{RequiredFlags: 3, FlagState: "unknown", Activations: []quake.UnitAction{{Map: "remote", SetsFlags: 1}}}}}
	p := &Planner{campaignDependency: d, World: World{Campaign: &CampaignDecision{}}}
	if goal, ok, active := p.campaignDependencyGoal(quake.Snapshot{Frame: 1000}); ok || !active || goal != (quake.Vec3{}) || p.World.Campaign.State != "unit_activation_required" {
		t.Fatal("remote condition became a local move", p.World.Campaign)
	}
	if d.UnitConditions[0].FlagState != "unknown" {
		t.Fatal("flags inferred from BSP rather than native execution")
	}
	s := quake.Snapshot{Movers: []quake.Mover{{Model: 1, Origin: quake.Vec3{0, 0, 100}}}}
	if _, _, active := p.campaignDependencyGoal(s); active || p.campaignDependency != nil {
		t.Fatal("observed opening did not release remote dependency")
	}
}
