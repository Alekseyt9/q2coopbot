package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// Stock baseq2 barrels deal 150 damage in radius190. Larger margins account
// for bounds, bolt flight, barrel delay and observed player motion. Walls are
// not assumed to protect from splash. Custom damage and unseen barrels remain
// outside this observed-state check.
func (p *Planner) guardBarrelShot(s quake.Snapshot, cmd quake.UserCmd) quake.UserCmd {
	if s.Health <= 0 || s.Weapon != "Blaster" || cmd.Buttons&1 == 0 {
		return cmd
	}
	yaw := float64(int16(uint16(cmd.Yaw)+uint16(s.DeltaAngles[1]))) * 2 * math.Pi / 65536
	pitch := float64(int16(uint16(cmd.Pitch)+uint16(s.DeltaAngles[0]))) * 2 * math.Pi / 65536
	f := quake.Vec3{math.Cos(pitch) * math.Cos(yaw), math.Cos(pitch) * math.Sin(yaw), -math.Sin(pitch)}
	start := s.Self
	start[2] += s.EyePoint()[2] - s.Self[2] - 8
	end := start
	for i := range start {
		start[i] += 24 * f[i]
		end[i] = start[i] + 1000*f[i] // Include overshoot if the enemy dodges.
	}
	if p.World.Geometry != nil {
		tr := p.World.Geometry.TraceProjectile(start, end)
		if tr.Valid {
			end = tr.End
		}
	}
	for index, barrel := range s.Barrels {
		if !barrelRay(s.Self, start, barrel.Origin) && !barrelRay(start, end, barrel.Origin) {
			continue
		}
		queue := []int{index}
		seen := map[int]bool{index: true}
		for len(queue) > 0 {
			at := queue[0]
			queue = queue[1:]
			b := s.Barrels[at]
			future := s.Self
			for i := range future {
				future[i] += s.SelfVelocity[i]
			}
			if quake.Distance(s.Self, b.Origin) < 320 || quake.Distance(future, b.Origin) < 320 ||
				s.Teammate != nil && quake.Distance(*s.Teammate, b.Origin) < 320 {
				cmd.Buttons &^= 1
				p.World.Command.LimitReason = "barrel_blast_risk"
				return cmd
			}
			for other, candidate := range s.Barrels {
				if !seen[other] && quake.Distance(b.Origin, candidate.Origin) < 288 {
					seen[other] = true
					queue = append(queue, other)
				}
			}
		}
	}
	return cmd
}

// Conservative padded stock barrel box, including network quantization.
func barrelRay(from, to, at quake.Vec3) bool {
	lo, hi := 0., 1.
	for i := range from {
		min, max := at[i]-24, at[i]+24
		if i == 2 {
			min, max = at[i]-8, at[i]+48
		}
		d := to[i] - from[i]
		if math.Abs(d) < 1e-8 {
			if from[i] < min || from[i] > max {
				return false
			}
			continue
		}
		a, b := (min-from[i])/d, (max-from[i])/d
		if a > b {
			a, b = b, a
		}
		lo, hi = math.Max(lo, a), math.Min(hi, b)
		if lo > hi {
			return false
		}
	}
	return true
}
