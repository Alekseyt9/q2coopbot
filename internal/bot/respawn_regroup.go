package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// A same-map death does not invalidate the last visual observation. This
// target is a rendezvous point, never a fabricated visible player entity.
type respawnRegroup struct {
	entity int
	target quake.Vec3
}

func (p *Planner) observeRespawnRegroup(previous, s quake.Snapshot) {
	if s.Teammate != nil || s.LastTeammate == nil || s.LastTeammateEntity <= 0 || previous.Map != s.Map || s.Frame < previous.Frame {
		p.respawnRegroup = nil
		return
	}
	if p.respawnRegroup != nil && p.respawnRegroup.entity != s.LastTeammateEntity {
		p.respawnRegroup = nil
	}
	if previous.Frame > 0 && previous.Health <= 0 && s.Health > 0 {
		p.respawnRegroup = &respawnRegroup{entity: s.LastTeammateEntity, target: *s.LastTeammate}
	}
}

func (p *Planner) respawnRegroupGoal(s quake.Snapshot) (quake.Vec3, bool) {
	r := p.respawnRegroup
	if r == nil || s.Health <= 0 || p.TestDisableSearch || p.testSetupHold || p.Nav == nil || p.World.GeometryStatus != "ready" {
		return quake.Vec3{}, false
	}
	if quake.Horizontal(s.Self, r.target) <= 64 && math.Abs(s.Self[2]-r.target[2]) <= 40 {
		p.respawnRegroup = nil
		return quake.Vec3{}, false
	}
	return r.target, true
}
