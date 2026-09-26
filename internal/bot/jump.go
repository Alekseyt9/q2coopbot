package bot

import (
	"math"

	"q2coopbot/internal/quake"
)

// A short jump owns movement until landing; airborne AAS areas need not have
// outgoing walking reaches. All timings use server frames (10 Hz).
type jumpFlight struct {
	from, landing quake.Vec3
	frame         int
	airborne      bool
	speed         float64
	runup         quake.Vec3
	phase         int // 0: retreat; 1: accelerate; 2: takeoff/flight
}

func (p *Planner) planGapJump() bool {
	s := p.World.Snapshot
	if p.World.Goal != "follow_teammate" || !s.OnGround || s.Health <= 0 || p.button != nil || p.elevator != nil {
		return false
	}
	// Prefer the nearest supported point along the remaining route, rather
	// than the endpoint of a walk-off reach, which may still be over a gap.
	candidates := append([]quake.Waypoint(nil), p.World.Route...)
	for _, wp := range p.World.Route {
		if wp.Kind == 11 {
			break
		}
		if d := quake.Horizontal(s.Self, wp.Position); d < 48 || d > 180 {
			continue
		}
		for _, dx := range []float64{0, -24, 24} {
			for _, dy := range []float64{-24, 24, 0} {
				at := wp.Position
				at[0] += dx
				at[1] += dy
				candidates = append(candidates, quake.Waypoint{Position: at})
			}
		}
	}
	for i := 0; i+1 < len(p.World.Route); i++ {
		a, b := p.World.Route[i], p.World.Route[i+1]
		if a.Kind == 11 || b.Kind == 11 {
			break
		}
		d := quake.Horizontal(a.Position, b.Position)
		for offset := 8.0; offset < d && offset <= 64; offset += 8 {
			u := offset / d
			at := a.Position
			for axis := range at {
				at[axis] += (b.Position[axis] - at[axis]) * u
			}
			candidates = append(candidates, quake.Waypoint{Position: at})
		}
	}
	for _, wp := range candidates {
		if wp.Kind == 11 {
			break
		}
		landing := wp.Position
		landing[2] += 0.125
		d := quake.Horizontal(s.Self, landing)
		if d < 48 || d > 180 || math.Abs(landing[2]-s.Self[2]) > 16 {
			continue
		}
		g := p.World.Geometry
		if !g.PlayerMoveClear(landing, landing) {
			continue
		}
		drop, ok := g.GroundDrop(landing, 4)
		if !ok || drop > 4 {
			continue
		}
		supported := true
		for _, offset := range []quake.Vec3{{12, 12, 0}, {12, -12, 0}, {-12, 12, 0}, {-12, -12, 0}} {
			at := landing
			at[0] += offset[0]
			at[1] += offset[1]
			if _, ok := g.GroundDrop(at, 4); !ok {
				supported = false
				break
			}
		}
		if !supported {
			continue
		}
		if p.Nav == nil {
			continue
		}
		if _, ok := p.Nav.Route(landing, p.goalPoint); !ok {
			continue
		}
		// Standard Quake II jump: vertical impulse 270, gravity 800.
		duration := (270 + math.Sqrt(270*270-1600*(landing[2]-s.Self[2]))) / 800
		speed := d / duration
		if speed > 280 || speed < 60 {
			continue
		}
		prev, clear := s.Self, true
		for i := 1; i <= 24; i++ {
			u := float64(i) / 24
			tm := duration * u
			at := quake.Vec3{s.Self[0] + (landing[0]-s.Self[0])*u, s.Self[1] + (landing[1]-s.Self[1])*u, s.Self[2] + 270*tm - 400*tm*tm}
			if !g.PlayerMoveClear(prev, at) || g.DoorShotBlocked(s.Movers, prev, at) {
				clear = false
				break
			}
			prev = at
		}
		if !clear {
			continue
		}
		back := s.Self
		back[0] -= (landing[0] - s.Self[0]) / d * 48
		back[1] -= (landing[1] - s.Self[1]) / d * 48
		prev = s.Self
		for i := 1; i <= 6; i++ {
			at := s.Self
			at[0] += (back[0] - s.Self[0]) * float64(i) / 6
			at[1] += (back[1] - s.Self[1]) * float64(i) / 6
			if g.GroundMoveHazardStep(p.Nav, prev, at[0]-prev[0], at[1]-prev[1], 8) != "" || g.DoorMoveHazard(s.Movers, prev, at[0]-prev[0], at[1]-prev[1]) != "" {
				clear = false
				break
			}
			prev = at
		}
		if !clear {
			continue
		}
		p.jump = &jumpFlight{from: s.Self, landing: landing, frame: s.Frame, speed: 280, runup: back}
		return true
	}
	return false
}

func (p *Planner) jumpCommand(cmd quake.UserCmd) (quake.UserCmd, bool) {
	j := p.jump
	if j == nil {
		return cmd, false
	}
	s := p.World.Snapshot
	if s.Health <= 0 || s.Frame < j.frame || s.Frame-j.frame > 35 || quake.Distance(s.Self, j.from) > 256 {
		p.jump = nil
		p.routeKnown = false
		p.World.Command = CommandDecision{MoveSource: "none", AimSource: "none", Skill: "gap_jump", LimitReason: "jump_aborted"}
		return cmd, true
	}
	if j.phase < 2 {
		if !s.OnGround {
			p.jump = nil
			p.routeKnown = false
			p.World.Command = CommandDecision{MoveSource: "none", AimSource: "none", Skill: "gap_jump", LimitReason: "runup_lost_ground"}
			return cmd, true
		}
		if j.phase == 0 && quake.Horizontal(s.Self, j.runup) < 8 {
			j.phase = 1
		}
		if j.phase == 1 && quake.Horizontal(s.Self, j.from) <= 22 {
			j.phase = 2
		}
		if j.phase < 2 {
			target := j.runup
			reason := "jump_prepare"
			if j.phase == 1 {
				target = j.from
				reason = "jump_runup"
			}
			speed := 280.0
			if j.phase == 0 {
				speed = math.Min(speed, quake.Horizontal(s.Self, target)*10)
			}
			cmd = worldMove(cmd, s, target[0]-s.Self[0], target[1]-s.Self[1], speed, false)
			p.World.Command = CommandDecision{MoveSource: "gap_jump", AimSource: "route", Skill: "gap_jump", LimitReason: reason}
			return cmd, true
		}
	}
	if !s.OnGround {
		j.airborne = true
	}
	if j.airborne && s.OnGround {
		reason := "jump_landed"
		if quake.Horizontal(s.Self, j.landing) > 32 || math.Abs(s.Self[2]-j.landing[2]) > 18 {
			reason = "jump_missed"
		}
		p.jump = nil
		p.routeKnown = false
		p.World.Command = CommandDecision{MoveSource: "none", AimSource: "none", Skill: "gap_jump", LimitReason: reason}
		return cmd, true
	}
	dx, dy := j.landing[0]-s.Self[0], j.landing[1]-s.Self[1]
	cmd = worldMove(cmd, s, dx, dy, math.Min(j.speed, math.Hypot(dx, dy)*10), false)
	phase := "jump_flight"
	if !j.airborne {
		cmd.Up = 200
		phase = "jump_takeoff"
	}
	p.World.Command = CommandDecision{MoveSource: "gap_jump", AimSource: "route", Skill: "gap_jump", LimitReason: phase}
	return cmd, true
}
