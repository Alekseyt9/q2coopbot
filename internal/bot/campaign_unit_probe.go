package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// Disappearance from PVS is not opening evidence. Only authoritative native
// displacement into a volume occupied by the observed closed door proves that
// it became passable. Never infer or publish the hidden game.serverflags.
func (p *Planner) campaignUnitProbePassed(s quake.Snapshot) bool {
	t := p.campaignUnitTrip
	if t == nil || !t.probing || t.State != "verify_effect" || !s.OnGround || s.Health <= 0 || s.Map != t.OriginMap || p.World.Geometry == nil {
		return false
	}
	d := t.Dependency
	dx, dy := d.ProbeTo[0]-d.ProbeFrom[0], d.ProbeTo[1]-d.ProbeFrom[1]
	distance := math.Hypot(dx, dy)
	if distance < 24 || distance > 40.01 {
		return false
	}
	px, py := s.Self[0]-d.ProbeFrom[0], s.Self[1]-d.ProbeFrom[1]
	along := (px*dx + py*dy) / distance
	cross := math.Abs(px*dy-py*dx) / distance
	closed := quake.Mover{Model: t.DoorModel, Origin: d.initial}
	return along >= 24 && along <= distance+4 && cross <= 4 && math.Abs(s.Self[2]-d.ProbeFrom[2]) <= 18 &&
		!p.World.Geometry.MoverHullClear(closed, s.Self, s.Self)
}

func (p *Planner) campaignUnitProbeCommand(s quake.Snapshot, cmd quake.UserCmd) (quake.UserCmd, bool) {
	t := p.campaignUnitTrip
	if t == nil || t.State != "verify_effect" || s.Map != t.OriginMap || !s.OnGround || s.Health <= 0 || t.ProbeFrames > 12 || p.World.Geometry == nil {
		return cmd, false
	}
	d, g := t.Dependency, p.World.Geometry
	closed := quake.Mover{Model: t.DoorModel, Origin: d.initial}
	if quake.Horizontal(d.ProbeFrom, d.ProbeTo) > 40.01 || !g.MoverHullClear(closed, d.ProbeFrom, d.ProbeFrom) || g.MoverHullClear(closed, d.ProbeFrom, d.ProbeTo) {
		return cmd, false
	}
	if d.ProbeFrom == d.ProbeTo || !t.probing && quake.Horizontal(s.Self, d.ProbeFrom) > 4 {
		return cmd, false
	}
	// Keep observed blockers strict. This probe is only for a door which was
	// observed before departure and is absent from the returning snapshot.
	for _, m := range s.Movers {
		if m.Model == t.DoorModel {
			return cmd, false
		}
	}
	if quake.Horizontal(s.Self, d.ProbeFrom) > 44 {
		return cmd, false
	}
	dx, dy := d.ProbeTo[0]-s.Self[0], d.ProbeTo[1]-s.Self[1]
	distance := math.Hypot(dx, dy)
	step := math.Min(8, distance)
	if step < .25 {
		return cmd, false
	}
	end := s.Self
	end[0] += dx / distance * step
	end[1] += dy / distance * step
	if !g.PlayerMoveClear(s.Self, end) || g.GroundMoveHazardStep(p.Nav, s.Self, dx, dy, step) != "" || g.PlayerTouchesHazard(end) || g.LaserMoveHazard(s.Self, end) {
		return cmd, false
	}
	for _, m := range s.Movers {
		if !g.MoverHullClear(m, s.Self, end) {
			return cmd, false
		}
	}
	// Bound total corridor and preserve floor support; don't probe cliffs.
	if !g.PlayerMoveClear(s.Self, d.ProbeTo) {
		return cmd, false
	}
	for i := 0; i <= 5; i++ {
		at := s.Self
		at[0] += (d.ProbeTo[0] - s.Self[0]) * float64(i) / 5
		at[1] += (d.ProbeTo[1] - s.Self[1]) * float64(i) / 5
		if _, ok := cornerFooting(g, at, 18.5); !ok || g.PlayerTouchesHazard(at) {
			return cmd, false
		}
	}
	t.probing = true
	p.World.Command = CommandDecision{MoveSource: "unit_effect_probe", AimSource: "route", Skill: "unit_effect_probe", LimitReason: "bounded_native_probe", MovePoint: &d.ProbeTo}
	return worldMove(quake.UserCmd{Yaw: cmd.Yaw}, s, dx, dy, step*10, false), true
}
