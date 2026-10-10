package bot

import (
	"math"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

// These execution limits do not choose a target, aim at it, or add firing.
// The original action/sample remains in the capture; every change is recorded.
func (p *Planner) guardLearnedCombatAim(o policy.Observation, s quake.Snapshot, a policy.Action, cmd quake.UserCmd) (quake.UserCmd, []policy.Intervention) {
	original := cmd
	var changes []policy.Intervention
	degrees := func(v int16) float64 { return float64(v) * 360 / 65536 }
	short := func(v float64) float64 { return math.Mod(v+540, 360) - 180 }
	angles := [2]float64{degrees(o.ViewAngles[0]), degrees(o.ViewAngles[1])}
	wanted := [2]float64{degrees(int16(uint16(cmd.Pitch) + uint16(s.DeltaAngles[0]))), degrees(int16(uint16(cmd.Yaw) + uint16(s.DeltaAngles[1])))}
	var target *quake.Object
	for _, e := range o.Enemies {
		if e.ID != a.TargetEntity || a.TargetTrack != 0 && (e.Track == nil || *e.Track != a.TargetTrack) {
			continue
		}
		for i := range s.Enemies {
			if s.Enemies[i].ID == e.ID {
				target = &s.Enemies[i]
				break
			}
		}
	}
	for axis, limit := range [2]float64{18, 36} {
		delta := short(wanted[axis] - angles[axis])
		clipped := math.Max(-limit, math.Min(limit, delta))
		reason := ""
		if math.Abs(clipped-delta) > .01 {
			reason = "learned_turn_rate"
		}
		if target != nil {
			point, eye := target.AimPoint(), s.EyePoint()
			dx, dy, dz := point[0]-eye[0], point[1]-eye[1], point[2]-eye[2]
			bearing := math.Atan2(dy, dx) * 180 / math.Pi
			if axis == 0 {
				bearing = -math.Atan2(dz, math.Hypot(dx, dy)) * 180 / math.Pi
			}
			before := math.Abs(short(angles[axis] - bearing))
			after := math.Abs(short(angles[axis] + clipped - bearing))
			if after > math.Max(10, before+2) {
				clipped, reason = 0, "learned_turn_away_from_target"
			}
		}
		wanted[axis] = angles[axis] + clipped
		if reason != "" {
			component := "yaw"
			if axis == 0 {
				component = "pitch"
			}
			changes = append(changes, policy.Intervention{Component: component, Reason: reason})
		}
	}
	cmd.Pitch = int16(int(math.Round(wanted[0]*65536/360))) - s.DeltaAngles[0]
	cmd.Yaw = int16(int(math.Round(wanted[1]*65536/360))) - s.DeltaAngles[1]
	if cmd.Yaw != original.Yaw && (cmd.Forward != 0 || cmd.Side != 0) {
		// Preserve the requested world direction when the view rotation is clipped.
		diff := degrees(int16(uint16(original.Yaw)-uint16(cmd.Yaw))) * math.Pi / 180
		f, side := float64(cmd.Forward), float64(cmd.Side)
		f, side = f*math.Cos(diff)+side*math.Sin(diff), -f*math.Sin(diff)+side*math.Cos(diff)
		scale := math.Max(1, math.Max(math.Abs(f), math.Abs(side))/400)
		cmd.Forward, cmd.Side = int16(math.Round(f/scale)), int16(math.Round(side/scale))
		changes = append(changes, policy.Intervention{Component: "movement", Reason: "learned_view_frame_adjustment"})
	}
	if cmd.Buttons&1 != 0 && !p.learnedShotHasTarget(o, s, cmd) {
		cmd.Buttons &^= 1
		changes = append(changes, policy.Intervention{Component: "attack", Reason: "learned_shot_off_target_or_blocked"})
	}
	return cmd, changes
}

func (p *Planner) learnedShotHasTarget(o policy.Observation, s quake.Snapshot, cmd quake.UserCmd) bool {
	if p.World.Geometry == nil || !p.World.Geometry.HasCollision() {
		return false
	}
	yaw := float64(int16(uint16(cmd.Yaw)+uint16(s.DeltaAngles[1]))) * 2 * math.Pi / 65536
	pitch := float64(int16(uint16(cmd.Pitch)+uint16(s.DeltaAngles[0]))) * 2 * math.Pi / 65536
	from := s.EyePoint()
	to := quake.Vec3{from[0] + 8192*math.Cos(pitch)*math.Cos(yaw), from[1] + 8192*math.Cos(pitch)*math.Sin(yaw), from[2] - 8192*math.Sin(pitch)}
	tr := p.World.Geometry.TraceProjectile(from, to)
	if !tr.Valid {
		return false
	}
	for _, e := range o.Enemies {
		if e.ClearShot != nil && !*e.ClearShot || e.Solid == nil || *e.Solid == 0 || *e.Solid == 31 {
			continue
		}
		solid := *e.Solid
		radius := float64(solid&31) * 8
		bottom, top := -float64((solid>>5)&31)*8, float64((solid>>10)&63)*8-32
		if radius == 0 || top <= bottom {
			continue
		}
		at := quake.Vec3{o.Position[0] + e.Relative[0], o.Position[1] + e.Relative[1], o.Position[2] + e.Relative[2]}
		// Allow muzzle parallax, stock spread, and observed projectile lead.
		pad := 8.0
		lo, hi := quake.Vec3{at[0] - radius - pad, at[1] - radius - pad, at[2] + bottom - pad}, quake.Vec3{at[0] + radius + pad, at[1] + radius + pad, at[2] + top + pad}
		if s.Weapon == "Blaster" && e.Velocity != nil {
			flight := math.Min(1.5, quake.Distance(from, at)/1000)
			for axis, v := range *e.Velocity {
				shift := v * flight
				lo[axis] += math.Min(0, shift)
				hi[axis] += math.Max(0, shift)
			}
		}
		if fraction, ok := learnedRayBox(from, to, lo, hi); ok && fraction <= tr.Fraction+4.0/8192 {
			return true
		}
	}
	return false
}

func learnedRayBox(from, to, min, max quake.Vec3) (float64, bool) {
	enter, leave := 0., 1.
	for axis := 0; axis < 3; axis++ {
		d := to[axis] - from[axis]
		if math.Abs(d) < 1e-8 {
			if from[axis] < min[axis] || from[axis] > max[axis] {
				return 0, false
			}
			continue
		}
		a, b := (min[axis]-from[axis])/d, (max[axis]-from[axis])/d
		if a > b {
			a, b = b, a
		}
		enter, leave = math.Max(enter, a), math.Min(leave, b)
		if enter > leave {
			return 0, false
		}
	}
	return enter, true
}
