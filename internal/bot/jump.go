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
	phase         int  // 0: retreat; 1: accelerate; 2: takeoff/flight; 3: brake and replan
	drop          bool // verified walk-off reach; never apply a jump impulse
}

type JumpTrace struct {
	From     quake.Vec3 `json:"from"`
	Landing  quake.Vec3 `json:"landing"`
	Runup    quake.Vec3 `json:"runup"`
	Speed    float64    `json:"speed"`
	Phase    int        `json:"phase"`
	Airborne bool       `json:"airborne"`
	Drop     bool       `json:"drop"`
}

func (j *jumpFlight) trace() *JumpTrace {
	return &JumpTrace{From: j.from, Landing: j.landing, Runup: j.runup, Speed: j.speed, Phase: j.phase, Airborne: j.airborne, Drop: j.drop}
}

func (p *Planner) planGapJump() bool {
	return p.planVerifiedJump(16)
}

func (p *Planner) planRampJump() bool {
	return p.planVerifiedJump(40)
}

func (p *Planner) planVerifiedJump(maxRise float64) bool {
	s := p.World.Snapshot
	if (p.World.Goal != "follow_teammate" && p.World.Goal != "regroup_after_respawn") || !s.OnGround || s.Health <= 0 || p.button != nil || p.elevator != nil {
		return false
	}
	// Prefer the nearest supported point along the remaining route, rather
	// than the endpoint of a walk-off reach, which may still be over a gap.
	route := p.World.Route
	for i, wp := range route {
		if wp.Kind == 11 {
			route = route[:i]
			break
		}
	}
	candidates := append([]quake.Waypoint(nil), route...)
	for _, wp := range route {
		if wp.Kind == 11 {
			break
		}
		if d := quake.Horizontal(s.Self, wp.Position); d < 48 || d > 180 {
			continue
		}
		xOffsets, yOffsets := []float64{0, -24, 24}, []float64{-24, 24, 0}
		if maxRise > 16 {
			yOffsets = append(yOffsets, -48, 48)
		}
		for _, dx := range xOffsets {
			for _, dy := range yOffsets {
				at := wp.Position
				at[0] += dx
				at[1] += dy
				candidates = append(candidates, quake.Waypoint{Position: at})
			}
		}
	}
	for i := 0; i+1 < len(route); i++ {
		a, b := route[i], route[i+1]
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
	brakeForAlignment := false
	for _, wp := range candidates {
		if wp.Kind == 11 {
			break
		}
		landing := wp.Position
		landing[2] += 0.125
		g := p.World.Geometry
		if maxRise > 16 {
			// AAS ramp samples can sit above the actual supporting BSP floor.
			// Probe the floor before applying the full-hull landing checks.
			if drop, ok := g.GroundDrop(landing, 32); ok && drop > 4 {
				landing[2] -= drop - 0.25
			}
		}
		d := quake.Horizontal(s.Self, landing)
		if d < 48 || d > 180 || landing[2]-s.Self[2] > maxRise || landing[2]-s.Self[2] < -64 {
			continue
		}
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
		// Validate both the continuous arc and the lower envelope produced by
		// 100 ms semi-implicit gravity steps (270 - gravity*dt/2 = 230).
		// Shorter commands fall between these two envelopes.
		impulses := []float64{270}
		if landing[2] < s.Self[2]-16 {
			impulses = append(impulses, 230)
		}
		prev, clear := s.Self, true
		for _, impulse := range impulses {
			flight := (impulse + math.Sqrt(impulse*impulse-1600*(landing[2]-s.Self[2]))) / 800
			prev = s.Self
			for i := 1; i <= 24; i++ {
				u := float64(i) / 24
				tm := flight * u
				at := quake.Vec3{s.Self[0] + (landing[0]-s.Self[0])*u, s.Self[1] + (landing[1]-s.Self[1])*u, s.Self[2] + impulse*tm - 400*tm*tm}
				if !g.PlayerMoveClear(prev, at) || g.DoorShotBlocked(s.Movers, prev, at) {
					clear = false
					break
				}
				prev = at
			}
			if !clear {
				break
			}
		}
		if !clear {
			continue
		}
		if maxRise > 16 && d <= 100 && speed <= 180 {
			if math.Hypot(s.SelfVelocity[0], s.SelfVelocity[1]) > 80 {
				// A standing arc is only valid after the approach momentum has
				// bled off; otherwise native air control carries us past it.
				continue
			}
			// This short ramp has no reliable ground run-up: the BSP probe
			// may approve retreat that native physics immediately loses.
			// Take off from the current grounded point along the checked arc.
			p.jump = &jumpFlight{from: s.Self, landing: landing, frame: s.Frame, speed: speed, phase: 2}
			return true
		}
		if landing[2] > s.Self[2]+8 {
			ux, uy := (landing[0]-s.Self[0])/d, (landing[1]-s.Self[1])/d
			along := s.SelfVelocity[0]*ux + s.SelfVelocity[1]*uy
			cross := math.Abs(s.SelfVelocity[0]*uy - s.SelfVelocity[1]*ux)
			if along >= 160 && along <= speed+20 && cross <= 50 {
				// The bot already has a grounded takeoff run toward the
				// landing. Excess momentum would carry it past the checked
				// arc, while lateral momentum would miss it; use a measured
				// run-up then.
				p.jump = &jumpFlight{from: s.Self, landing: landing, frame: s.Frame, speed: speed, phase: 2}
				return true
			}
			if along >= 160 && along <= speed+20 && cross > 50 {
				brakeForAlignment = true
			}
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
		// Lower landings need the calculated speed: full run speed would
		// carry the bot beyond the verified landing before it reaches the floor.
		launchSpeed := 280.0
		if maxRise > 16 {
			launchSpeed = speed
		} else if landing[2] < s.Self[2]-16 {
			launchSpeed = d / ((230 + math.Sqrt(230*230-1600*(landing[2]-s.Self[2]))) / 800)
		}
		p.jump = &jumpFlight{from: s.Self, landing: landing, frame: s.Frame, speed: launchSpeed, runup: back}
		return true
	}
	if brakeForAlignment {
		// There is a checked landing, but the current lateral momentum
		// cannot be corrected in the air and the retreat is unsafe. Let
		// ground friction reduce it, then replan from the new position.
		p.jump = &jumpFlight{from: s.Self, frame: s.Frame, phase: 3}
		return true
	}
	return false
}

func (p *Planner) jumpCommand(cmd quake.UserCmd) (quake.UserCmd, bool) {
	j := p.jump
	if j == nil {
		return cmd, false
	}
	defer func() { p.World.Jump = j.trace() }()
	s := p.World.Snapshot
	skill, prefix := "gap_jump", "jump"
	if j.drop {
		skill, prefix = "walk_off", "drop"
	}
	if s.Health <= 0 || s.Frame < j.frame || s.Frame-j.frame > 35 || quake.Distance(s.Self, j.from) > 256 {
		p.jump = nil
		p.routeKnown = false
		p.World.Command = CommandDecision{MoveSource: "none", AimSource: "none", Skill: skill, LimitReason: prefix + "_aborted"}
		return cmd, true
	}
	if j.drop && !j.airborne && s.OnGround && s.Frame-j.frame < 2 {
		// Shed the approach velocity before stepping off: air braking is weak.
		p.World.Command = CommandDecision{MoveSource: "none", Skill: skill, LimitReason: "drop_prepare"}
		return cmd, true
	}
	if j.phase == 3 {
		if !s.OnGround || math.Hypot(s.SelfVelocity[0], s.SelfVelocity[1]) <= 80 {
			p.jump = nil
			p.routeKnown = false
			p.World.Command = CommandDecision{MoveSource: "none", Skill: skill, LimitReason: "jump_braked"}
			return quake.UserCmd{Yaw: cmd.Yaw}, true
		}
		p.World.Command = CommandDecision{MoveSource: "none", Skill: skill, LimitReason: "jump_braking"}
		return quake.UserCmd{Yaw: cmd.Yaw}, true
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
		takeoffRadius := 22.0
		if j.speed < 200 || j.landing[2] < j.from[2]-16 {
			takeoffRadius = 8
		}
		if j.phase == 1 && quake.Horizontal(s.Self, j.from) <= takeoffRadius {
			j.phase = 2
		}
		if j.phase < 2 {
			target := j.runup
			reason := "jump_prepare"
			if j.phase == 1 {
				target = j.from
				reason = "jump_runup"
			}
			speed := j.speed
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
		reason := prefix + "_landed"
		if quake.Horizontal(s.Self, j.landing) > 32 || math.Abs(s.Self[2]-j.landing[2]) > 18 {
			reason = prefix + "_missed"
		}
		p.jump = nil
		p.routeKnown = false
		p.World.Command = CommandDecision{MoveSource: "none", AimSource: "none", Skill: skill, LimitReason: reason}
		return cmd, true
	}
	dx, dy := j.landing[0]-s.Self[0], j.landing[1]-s.Self[1]
	cmd = worldMove(cmd, s, dx, dy, math.Min(j.speed, math.Hypot(dx, dy)*10), false)
	phase := prefix + "_flight"
	if !j.airborne && !j.drop {
		cmd.Up = 200
		phase = "jump_takeoff"
	}
	p.World.Command = CommandDecision{MoveSource: skill, AimSource: "route", Skill: skill, LimitReason: phase}
	return cmd, true
}

// A blocked walking corner is not evidence of a gap. Only a nearby paired
// walk-off reach permits replacing its blocked entry with a verified jump.
func (p *Planner) blockedDropApproach() bool {
	r := p.World.Route
	if len(r) < 2 || r[0].Kind != 7 || r[1].Kind != 7 || r[0].ToArea != r[1].ToArea {
		return false
	}
	dz := p.World.Snapshot.Self[2] - r[1].Position[2]
	return dz >= 24 && dz <= 64 && quake.Horizontal(p.World.Snapshot.Self, r[0].Position) <= 128
}

// A walking reach can climb a ramp which the conservative BSP hull probe
// marks blocked. Only a nearby paired upward reach permits a verified jump.
func (p *Planner) blockedRiseApproach() bool {
	r := p.World.Route
	for i := 0; i+1 < len(r) && i < 6; i++ {
		if r[i].Kind != 2 || r[i+1].Kind != 2 || r[i].ToArea != r[i+1].ToArea {
			continue
		}
		rise := r[i+1].Position[2] - p.World.Snapshot.Self[2]
		if rise >= 16 && rise <= 40 && quake.Horizontal(p.World.Snapshot.Self, r[i].Position) <= 128 {
			return true
		}
	}
	return false
}
