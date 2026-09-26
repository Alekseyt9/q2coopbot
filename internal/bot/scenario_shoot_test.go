package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestScenarioShotRequiresKnownLiveTargetAndGeometry(t *testing.T) {
	p := quake.Vec3{100, 0, 24}
	for _, s := range []quake.Snapshot{{Health: 100}, {Health: 0, Teammate: &p}, {Health: 100, Teammate: &p}} {
		if scenarioShootTeammate(s, nil).Buttons != 0 {
			t.Fatal("shot without necessary observation/geometry")
		}
	}
}
