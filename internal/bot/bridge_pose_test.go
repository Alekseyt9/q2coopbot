package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestBridgeRotationInvalidatesStationaryPose(t *testing.T) {
	old := quake.Mover{Model: 40, Origin: quake.Vec3{450, -958, -18}}
	current := old
	current.Angles[2] = 90
	p := &Planner{World: World{Snapshot: quake.Snapshot{Map: "base2", Frame: 11, Movers: []quake.Mover{current}}}, doorPrevious: quake.Snapshot{Map: "base2", Frame: 10, Movers: []quake.Mover{old}}}
	if p.stationaryBridge(40, old.Origin) {
		t.Fatal("unchanged pivot mistaken for stationary rotating geometry")
	}
	p.doorPrevious.Movers[0] = current
	if !p.stationaryBridge(40, old.Origin) {
		t.Fatal("observed stable orientation rejected")
	}
	p.bridgeLink = &bridgeLink{model: 40, origin: old.Origin}
	if _, handled := p.bridgeLinkCommand(quake.UserCmd{}); !handled || p.bridgeLink != nil || p.World.Command.MoveLimitReason != "bridge_link_orientation_changed" {
		t.Fatal("stale link survived new stable orientation")
	}
}
