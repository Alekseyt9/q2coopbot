package learningenv

import (
	"fmt"
	"math"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

// Costs concern executed commands, not inferred native misses. Missing bounds,
// occluded targets, new targets, and projectile lead are not labelled failures.
func (c RewardConfig) actionQualityCosts(s *Step) (map[string]float64, error) {
	r := map[string]float64{"off_target_attack": 0, "turn_away": 0, "stalled_movement": 0}
	o, a := s.Observation, s.AppliedAction
	pitch, yaw := float64(o.ViewAngles[0])*math.Pi/32768, float64(o.ViewAngles[1])*math.Pi/32768
	if c.AimKickAngles && o.KickAngles != nil {
		pitch += (*o.KickAngles)[0] * math.Pi / 180
		yaw += (*o.KickAngles)[1] * math.Pi / 180
	}
	direction := func(p, y float64) quake.Vec3 {
		return quake.Vec3{math.Cos(p) * math.Cos(y), math.Cos(p) * math.Sin(y), -math.Sin(p)}
	}
	before := direction(pitch, yaw)
	after := direction(pitch+a.PitchDelta*math.Pi/180, yaw+a.YawDelta*math.Pi/180)
	errorDegrees := func(v, target quake.Vec3) float64 {
		return math.Acos(math.Max(-1, math.Min(1, v[0]*target[0]+v[1]*target[1]+v[2]*target[2]))) * 180 / math.Pi
	}
	known, uncertain, aligned := 0, false, false
	for _, e := range o.Enemies {
		bearing, available, err := policy.ObservedAimDirection(o, e)
		if err != nil {
			return nil, err
		}
		if !available {
			uncertain = true
			continue
		}
		known++
		radius := float64(*e.Solid&31)*8 + 8
		tolerance := 8 + math.Atan2(radius, math.Max(1, e.Distance))*180/math.Pi
		if o.Weapon == "Blaster" && e.Velocity != nil {
			speed := math.Hypot(math.Hypot((*e.Velocity)[0], (*e.Velocity)[1]), (*e.Velocity)[2])
			tolerance += math.Atan2(speed*math.Min(1.5, e.Distance/1000), math.Max(1, e.Distance)) * 180 / math.Pi
		}
		currentError, newError := errorDegrees(before, bearing), errorDegrees(after, bearing)
		if math.IsNaN(newError) || math.IsNaN(tolerance) || math.IsInf(tolerance, 0) {
			return nil, fmt.Errorf("invalid action geometry")
		}
		aligned = aligned || newError <= tolerance
		if e.ID == s.Action.TargetEntity && e.Track != nil && *e.Track == s.Action.TargetTrack &&
			o.PreviousTarget != nil && o.PreviousTarget.Entity == e.ID && o.PreviousTarget.Track == *e.Track &&
			math.Abs(a.YawDelta)+math.Abs(a.PitchDelta) >= 10 && newError > currentError+10 {
			for _, next := range s.Next.Enemies {
				if next.ID == e.ID && next.Track != nil && *next.Track == *e.Track && s.Next.Health > 0 {
					r["turn_away"] = c.TurnAway
				}
			}
		}
	}
	// This experiment calibrates direct aim only for the two training loadouts.
	// Splash, ricochets and other weapons need their own shot geometry.
	directWeapon := o.Weapon == "Blaster" || o.Weapon == "Machinegun" || o.Weapon == "models/weapons/v_machn/tris.md2"
	if directWeapon && a.Attack && known > 0 && !uncertain && !aligned {
		r["off_target_attack"] = c.OffTargetAttack
	}
	previous := o.PreviousCommand
	if a.Vertical == "release" && o.OnGround && s.Next.OnGround && !o.Ducked && !s.Next.Ducked &&
		math.Hypot(a.Forward, a.Side) >= .25 && math.Hypot(float64(previous.Forward), float64(previous.Side)) >= 100 &&
		previous.Up == 0 && quake.Distance(o.Position, s.Next.Position) < 1 &&
		math.Hypot(s.Next.Velocity[0], s.Next.Velocity[1]) < 10 {
		r["stalled_movement"] = c.StalledMovement
	}
	return r, nil
}
