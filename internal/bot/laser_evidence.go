package bot

import "q2coopbot/internal/quake"

// LaserEvidence describes the network observation behind the laser guard.
// An absent beam is unknown unless its entity was removed in a stable view.
type LaserEvidence struct {
	Origin quake.Vec3 `json:"origin"`
	State  string     `json:"state"`
	Reason string     `json:"reason,omitempty"`
	Frame  int        `json:"frame"`
}

type laserMemory struct {
	pendingFrame int
	pendingSelf  quake.Vec3
	offUntil     int
	lastOn       int
}

func sameNearbyMovers(a, b []quake.Mover, self, emitter quake.Vec3) bool {
	relevant := func(m quake.Mover) bool {
		return quake.Distance(m.Origin, self) < 320 || quake.Distance(m.Origin, emitter) < 320
	}
	for _, x := range a {
		if !relevant(x) {
			continue
		}
		found := false
		for _, y := range b {
			if x.ID == y.ID && x.Model == y.Model && quake.Distance(x.Origin, y.Origin) < 1 {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	for _, y := range b {
		if !relevant(y) {
			continue
		}
		found := false
		for _, x := range a {
			if x.ID == y.ID && x.Model == y.Model && quake.Distance(x.Origin, y.Origin) < 1 {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func beamAt(beams []quake.BeamObservation, origin quake.Vec3) (quake.BeamObservation, bool) {
	for _, beam := range beams {
		if quake.Distance(beam.Origin, origin) < 1 {
			return beam, true
		}
	}
	return quake.BeamObservation{}, false
}

func removedID(ids []int, id int) bool {
	for _, removed := range ids {
		if removed == id {
			return true
		}
	}
	return false
}

func (p *Planner) observeLasers(previous, s quake.Snapshot) {
	p.World.LaserEvidence = nil
	if p.World.Geometry == nil || !p.World.Geometry.HasStaticLethalLasers() {
		return
	}
	if previous.Frame >= s.Frame || previous.Map != "" && previous.Map != s.Map {
		p.laserEvidence = nil
	}
	if p.laserEvidence == nil {
		p.laserEvidence = make(map[quake.Vec3]*laserMemory)
	}
	for _, origin := range p.World.Geometry.StaticLethalLaserOrigins() {
		continuous := previous.Map == s.Map && previous.Frame > 0 && s.Frame == previous.Frame+1 &&
			previous.Suppressed == 0 && s.Suppressed == 0 && sameNearbyMovers(previous.Movers, s.Movers, s.Self, origin)
		stable := continuous && quake.Distance(previous.Self, s.Self) < 4
		memory := p.laserEvidence[origin]
		if memory == nil {
			memory = &laserMemory{}
			p.laserEvidence[origin] = memory
		}
		if !continuous {
			memory.pendingFrame, memory.offUntil = 0, 0
		}
		state := "unknown"
		reason := "unstable_view"
		if _, on := beamAt(s.Beams, origin); on {
			memory.pendingFrame, memory.offUntil, memory.lastOn = 0, 0, s.Frame
			state = "on"
			reason = "network_beam"
		} else if stable && memory.pendingFrame == previous.Frame && quake.Distance(memory.pendingSelf, s.Self) < 4 {
			memory.pendingFrame = 0
			memory.offUntil = s.Frame + 6
			state = "off"
			reason = "stable_absence"
		} else {
			if memory.pendingFrame > 0 && (!stable || s.Frame > memory.pendingFrame+1) {
				memory.pendingFrame = 0
			}
			if stable {
				reason = "no_complete_absence"
				if old, seen := beamAt(previous.Beams, origin); seen && (removedID(s.RemovedEntities, old.ID) || s.DeltaFrame <= 0) {
					memory.pendingFrame, memory.pendingSelf = s.Frame, s.Self
					reason = "pending_absence"
				}
			}
			if memory.offUntil >= s.Frame && memory.offUntil > 0 {
				state = "off"
				reason = "recent_absence"
			}
		}
		p.World.LaserEvidence = append(p.World.LaserEvidence, LaserEvidence{Origin: origin, State: state, Reason: reason, Frame: s.Frame})
	}
}

func (p *Planner) laserOffOrigins(frame int) []quake.Vec3 {
	var off []quake.Vec3
	for origin, memory := range p.laserEvidence {
		if memory.offUntil >= frame && memory.offUntil > 0 {
			off = append(off, origin)
		}
	}
	return off
}
