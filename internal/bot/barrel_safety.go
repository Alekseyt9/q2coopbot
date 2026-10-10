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
	if s.Health <= 0 || (s.Weapon != "Blaster" && !directHitscanWeapon(s.Weapon)) || cmd.Buttons&1 == 0 {
		return cmd
	}
	yaw := float64(int16(uint16(cmd.Yaw)+uint16(s.DeltaAngles[1]))) * 2 * math.Pi / 65536
	pitch := float64(int16(uint16(cmd.Pitch)+uint16(s.DeltaAngles[0]))) * 2 * math.Pi / 65536
	f := quake.Vec3{math.Cos(pitch) * math.Cos(yaw), math.Cos(pitch) * math.Sin(yaw), -math.Sin(pitch)}
	start := s.Self
	start[2] += s.EyePoint()[2] - s.Self[2] - 8
	end := start
	rangeLimit := 1000.0
	if directHitscanWeapon(s.Weapon) {
		rangeLimit = 8192
	}
	if superShotgunWeapon(s.Weapon) {
		spread := math.Hypot(1000, 500)
		rangeLimit = math.Hypot(8192, spread) + math.Hypot(8192, 2*spread)
	}
	for i := range start {
		if !directHitscanWeapon(s.Weapon) {
			start[i] += 24 * f[i]
		}
		end[i] = start[i] + rangeLimit*f[i]
	}
	if p.World.Geometry != nil && !directHitscanWeapon(s.Weapon) {
		tr := p.World.Geometry.TraceProjectile(start, end)
		if tr.Valid {
			end = tr.End
		}
	}
	for index, barrel := range s.Barrels {
		intersects := barrelRay(s.Self, start, barrel.Origin) || barrelRay(start, end, barrel.Origin)
		if s.Weapon == "Blaster" {
			axis, starts, padding := boltLaunchPaths(s, cmd)
			for _, muzzle := range starts {
				tip := quake.Vec3{muzzle[0] + 1000*axis[0], muzzle[1] + 1000*axis[1], muzzle[2] + 1000*axis[2]}
				intersects = intersects || paddedBarrelRay(s.Self, muzzle, barrel.Origin, padding) || paddedBarrelRay(muzzle, tip, barrel.Origin, padding)
			}
		}
		if directHitscanWeapon(s.Weapon) {
			degrees := 18.0
			if superShotgunWeapon(s.Weapon) {
				degrees = 30
			}
			intersects = barrelRay(s.Self, start, barrel.Origin) || hitscanBarrelCone(start, end, barrel.Origin, degrees)
		}
		if !intersects {
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
	return paddedBarrelRay(from, to, at, 0)
}

func paddedBarrelRay(from, to, at quake.Vec3, padding float64) bool {
	lo, hi := 0., 1.
	for i := range from {
		min, max := at[i]-24, at[i]+24
		if i == 2 {
			min, max = at[i]-8, at[i]+48
		}
		min, max = min-padding, max+padding
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

// Stock Machinegun kick is bounded at 13.5 degrees; bullet spread and yaw
// kick fit within this conservative 18-degree cone. The barrel sphere encloses
// Stock Shotgun 1000/500 spread at range8192 and 2-degree kick also fit;
// the wider Machinegun envelope is intentionally retained for its pellets.
// its padded observed box; the extra 12 covers muzzle offset/quantization.
// This neither corrects aim nor assumes central-ray walls block spread rays.
func machinegunBarrelCone(from, to, at quake.Vec3) bool {
	return hitscanBarrelCone(from, to, at, 18)
}

func hitscanBarrelCone(from, to, at quake.Vec3, degrees float64) bool {
	length := quake.Distance(from, to)
	if length == 0 {
		return false
	}
	center := at
	center[2] += 20
	along, distance2 := 0.0, 0.0
	for i := range from {
		d := center[i] - from[i]
		along += d * (to[i] - from[i]) / length
		distance2 += d * d
	}
	radius := math.Sqrt(24*24+24*24+28*28) + 12
	if along < -radius || along > length+radius {
		return false
	}
	width := radius + math.Max(0, along)*math.Tan(degrees*math.Pi/180)
	return math.Max(0, distance2-along*along) <= width*width
}
