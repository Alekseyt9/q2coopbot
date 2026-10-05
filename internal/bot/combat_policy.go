package bot

import (
	"fmt"
	"math"
	"time"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

type combatControl struct {
	history    policy.History
	mode       string
	provider   policy.Provider
	last       policy.Identity
	lastHealth int16
	life       int
}

func (c *Client) combatObservation(now time.Time) policy.Observation {
	s := c.planner.World.Snapshot
	id := policy.Identity{Map: s.Map, Connection: c.connection, Spawncount: c.spawncount, Actor: c.decoder.PlayerNumber, Frame: s.Frame}
	b := &c.combatControl
	previousID := b.last
	previousID.Life = 0
	if !policy.SameLife(previousID, id) || s.Frame < b.last.Frame {
		b.life = 1
	} else if s.Frame > b.last.Frame && b.lastHealth <= 0 && s.Health > 0 {
		b.life++
	}
	id.Life = b.life
	b.last, b.lastHealth = id, s.Health
	o := policy.Observe(s, id, c.previous)
	o.AgeMS = now.Sub(c.planner.World.Updated).Milliseconds()
	policy.EnrichEnvironment(&o, s, c.planner.World.Geometry)
	b.history.Enrich(&o)
	return o
}

// Direct mode does not call Planner.command on a valid policy-controlled frame.
// Rules own setup, death, noncombat travel and explicitly marked failures.
func (c *Client) combatCommand(o policy.Observation, now time.Time) (quake.UserCmd, quake.UserCmd, *policy.Selection, bool) {
	b := &c.combatControl
	sel := &policy.Selection{Mode: b.mode, Owner: "rules"}
	rules := func() (quake.UserCmd, quake.UserCmd, *policy.Selection, bool) {
		cmd := c.planner.command(c.previous)
		return cmd, c.planner.World.Command.proposedCommand, sel, false
	}
	if b.mode == "" || b.mode == "rules" {
		return rules()
	}
	if b.provider == nil {
		sel.Fallback = "provider_unavailable"
		return rules()
	}
	sel.ProviderVersion = b.provider.Version()
	if o.Health <= 0 || o.AgeMS < 0 || o.AgeMS > 300 || o.Identity.Frame != c.latestFrame {
		sel.Fallback = "dead_or_stale"
		return rules()
	}
	if c.idle || c.planner.testSetupHold || c.testCombatBarrier && !c.testCombatGo || c.testHoldPosition || c.testHoldPositionMap == o.Identity.Map || c.testTeleportSent && c.latestFrame-c.testTeleportSentFrame < 3 {
		sel.Fallback = "setup_or_harness_override"
		return rules()
	}
	if len(o.Enemies) == 0 {
		sel.Fallback = "system2_noncombat"
		return rules()
	}
	if o.Weapon != "Blaster" {
		sel.Fallback = "pilot_equip_not_ready"
		return rules()
	}
	if c.planner.World.Geometry == nil || c.planner.World.GeometryStatus != "ready" && c.planner.World.GeometryStatus != "available" {
		sel.Fallback = "geometry_unavailable"
		return rules()
	}
	start := time.Now()
	a, err := boundedDecision(b.provider, o)
	sel.Candidate = &a
	var proposed quake.UserCmd
	if err == nil {
		proposed, err = policy.Command(o, a, c.planner.World.Snapshot.DeltaAngles)
	}
	if err != nil {
		sel.Fallback = "invalid_provider: " + err.Error()
		sel.ElapsedUS = time.Since(start).Microseconds()
		return rules()
	}
	guarded, changes := c.planner.guardDirectCombat(c.planner.World.Snapshot, proposed)
	converted := policy.FromCommand(o, proposed, c.planner.World.Snapshot.DeltaAngles, "")
	if math.Abs(converted.PitchDelta-a.PitchDelta) > .01 {
		changes = append(changes, policy.Intervention{Component: "pitch", Reason: "protocol_pitch_limit"})
	}
	sel.CandidateCommand, sel.GuardedCommand, sel.Interventions = &proposed, &guarded, changes
	sel.ElapsedUS = time.Since(start).Microseconds()
	budget := 5 * time.Millisecond
	if remote, ok := b.provider.(*policy.Remote); ok && c.testSynchronous {
		budget = remote.DecisionBudget()
	}
	if sel.ElapsedUS > budget.Microseconds() {
		sel.Fallback = "inference_budget_exceeded"
		return rules()
	}
	if b.mode == "learned-shadow" {
		return rules()
	}
	sel.Owner = "provider"
	c.planner.World.Command = CommandDecision{MoveSource: "policy", AimSource: "policy", proposedCommand: proposed}
	for _, change := range changes {
		if change.Component == "movement" {
			c.planner.World.Command.MoveLimitReason = change.Reason
		} else {
			c.planner.World.Command.LimitReason = change.Reason
		}
	}
	return guarded, proposed, sel, true
}

func boundedDecision(p policy.Provider, o policy.Observation) (a policy.Action, err error) {
	defer func() {
		if v := recover(); v != nil {
			err = fmt.Errorf("provider panic: %v", v)
		}
	}()
	return p.Decide(o)
}

// The current pilot supports stock Blaster and dry ground/flat-ground jumps.
// Constraints stop components; they never aim, create a route or add firing.
func (p *Planner) guardDirectCombat(s quake.Snapshot, cmd quake.UserCmd) (quake.UserCmd, []policy.Intervention) {
	changes := []policy.Intervention{}
	stopMove := func(reason string) {
		cmd.Forward, cmd.Side, cmd.Up = 0, 0, 0
		changes = append(changes, policy.Intervention{Component: "movement", Reason: reason})
	}
	stopFire := func(reason string) {
		cmd.Buttons &^= 1
		changes = append(changes, policy.Intervention{Component: "attack", Reason: reason})
	}
	g := p.World.Geometry
	if g == nil || !g.HasCollision() {
		stopMove("bsp_unavailable")
		if cmd.Buttons&1 != 0 {
			stopFire("bsp_unavailable")
		}
		return cmd, changes
	}
	probe := cmd
	probe.Up = 0
	predicted := predictGroundStep(s, probe)
	if predicted != nil {
		dx, dy := predicted.Displacement[0], predicted.Displacement[1]
		step := math.Hypot(dx, dy)
		if hazard := g.GroundMoveHazardStep(p.Nav, s.Self, dx, dy, step); hazard != "" {
			stopMove(hazard)
		} else if hazard := g.DoorMoveHazard(s.Movers, s.Self, dx, dy); hazard != "" {
			stopMove(hazard)
		} else if g.HasStaticLethalLasers() && p.laserCommandUnsafe(s, probe) {
			stopMove("static_laser_hazard")
		}
		if cmd.Up > 0 {
			top := s.Self
			top[2] += 32
			if !g.PlayerMoveClear(s.Self, top) {
				cmd.Up = 0
				changes = append(changes, policy.Intervention{Component: "vertical", Reason: "jump_headroom"})
			}
		}
	} else if cmd.Forward != 0 || cmd.Side != 0 || cmd.Up != 0 {
		// Airborne/crouching dynamics require a checked trajectory guard before
		// the pilot permits new acceleration there; existing momentum is real.
		stopMove("unsupported_motion_guard")
	}
	if cmd.Buttons&1 != 0 {
		if s.Weapon != "Blaster" {
			stopFire("pilot_fixed_blaster")
		} else {
			yaw := float64(int16(uint16(cmd.Yaw)+uint16(s.DeltaAngles[1]))) * 2 * math.Pi / 65536
			pitch := float64(int16(uint16(cmd.Pitch)+uint16(s.DeltaAngles[0]))) * 2 * math.Pi / 65536
			from := s.EyePoint()
			to := from
			to[0] += 1024 * math.Cos(pitch) * math.Cos(yaw)
			to[1] += 1024 * math.Cos(pitch) * math.Sin(yaw)
			to[2] -= 1024 * math.Sin(pitch)
			trace := g.TraceProjectile(from, to)
			if trace.Valid {
				to = trace.End
			}
			recentPartner := s.LastTeammate != nil && s.TeammateAgeFrames != nil && *s.TeammateAgeFrames <= 10 && teammateBlocksShot(from, to, *s.LastTeammate)
			if s.Teammate != nil && teammateBlocksShot(from, to, *s.Teammate) || recentPartner || p.teammateEntersProjectile(s, from, to) {
				stopFire("friendly_line_of_fire")
			}
			before := cmd
			saved := p.World.Command
			cmd = p.guardBarrelShot(s, cmd)
			if before.Buttons != cmd.Buttons {
				changes = append(changes, policy.Intervention{Component: "attack", Reason: p.World.Command.LimitReason})
			}
			p.World.Command = saved
		}
	}
	return cmd, changes
}
