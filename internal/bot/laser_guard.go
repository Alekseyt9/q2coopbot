package bot

import (
	"math"

	"q2coopbot/internal/quake"
)

func laserStopEnd(start, velocity quake.Vec3, distance float64) quake.Vec3 {
	speed := math.Hypot(velocity[0], velocity[1])
	if speed > 0 {
		start[0] += velocity[0] / speed * distance
		start[1] += velocity[1] / speed * distance
	}
	return start
}

func (p *Planner) laserCommandUnsafe(s quake.Snapshot, cmd quake.UserCmd) bool {
	probe := cmd
	probe.Up = 0 // Treat jumping over a lethal beam as unverified.
	// The harness can execute this 50 ms planner command as a 100 ms game
	// command at 2x. Check the longer interval before it is transmitted.
	if probe.Msec < 100 {
		probe.Msec = 100
	}
	predicted := predictGroundStep(s, probe)
	if predicted == nil {
		return false
	}
	end := s.Self
	end[0] += predicted.Displacement[0]
	end[1] += predicted.Displacement[1]
	next := s
	next.Self, next.SelfVelocity = end, predicted.Velocity
	neutral := probe
	neutral.Forward, neutral.Side = 0, 0
	remaining := predictGroundStep(next, neutral)
	tail := end
	if remaining != nil {
		tail = laserStopEnd(end, predicted.Velocity, remaining.NeutralStopDistance)
	}
	off := p.laserOffOrigins(s.Frame)
	return p.World.Geometry.LaserMoveHazardExcept(s.Self, end, off) || p.World.Geometry.LaserMoveHazardExcept(end, tail, off)
}

// A laser can kill before the next snapshot. Check the issued command and its
// stopping tail, so a late neutral command does not coast through the beam.
func (p *Planner) limitLaserMovement(s quake.Snapshot, cmd quake.UserCmd) quake.UserCmd {
	if p.World.Geometry == nil || !p.World.Geometry.HasStaticLethalLasers() || !s.OnGround || s.Health <= 0 ||
		p.World.Geometry.LaserMoveHazardExcept(s.Self, s.Self, p.laserOffOrigins(s.Frame)) || !p.laserCommandUnsafe(s, cmd) {
		return cmd
	}
	cmd.Forward, cmd.Side, cmd.Up = 0, 0, 0
	p.World.Command.MoveSource = "none"
	p.World.Command.MoveLimitReason = "static_laser_hazard"
	p.World.Command.LimitReason = "static_laser_hazard"
	p.lastProgress = p.navigationNow(s.Frame)
	p.detourUntil = p.lastProgress
	p.failures = 0
	return cmd
}
