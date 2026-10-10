package bot

import (
	"os"
	"testing"

	"q2coopbot/internal/quake"
)

func TestCampaignLeaderRetainsTeammateAndRoute(t *testing.T) {
	for _, leader := range []bool{false, true} {
		p := policyClient(t, "rules").planner
		p.Campaign, p.CampaignLeader = true, leader
		p.CampaignNextMap = "base2"
		s := p.World.Snapshot
		mate := quake.Vec3{60, -260, 24.125}
		s.Teammate, s.TeammateEntity = &mate, 2
		p.update(s, os.Getenv("Q2_SEARCH_SCAN_ROOT"))
		if (p.World.Campaign != nil) != leader {
			t.Fatalf("leader=%v campaign=%+v goal=%s", leader, p.World.Campaign, p.World.Goal)
		}
		if p.World.Snapshot.Teammate == nil || *p.World.Snapshot.Teammate != mate || p.World.Snapshot.TeammateEntity != 2 {
			t.Fatal("route ownership hid teammate from combat observations")
		}
		if leader && p.goalPoint == mate {
			t.Fatal("campaign destination was replaced by teammate position")
		}
	}
}

func TestCampaignLeaderButtonOwnership(t *testing.T) {
	mate := quake.Vec3{200, 0, 24}
	s := quake.Snapshot{Health: 100, Frame: 10, Teammate: &mate}
	for _, leader := range []bool{false, true} {
		p := &Planner{Campaign: true, CampaignLeader: leader, button: &buttonTask{campaign: true}}
		p.validateButtonOwner(s)
		if (p.button != nil) != leader {
			t.Fatalf("leader=%v button task retained=%v", leader, p.button != nil)
		}
		p.Campaign = false
		p.validateButtonOwner(s)
		if p.button != nil {
			t.Fatal("campaign task survived loss of campaign role")
		}
	}
}
