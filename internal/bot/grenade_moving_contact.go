package bot

import (
	"fmt"
	"math"
	"q2coopbot/internal/quake"
)

// Isolated walking misc_insane fixture only. Native edicts run in ID order;
// a target with a higher ID moves after the grenade and its contact knockback.
// Observed target positions are inputs, not a forecast fitted to the explosion.
func CheckGrenadeMovingContact(rows []quake.Snapshot, g *quake.MapInfo) (GrenadeCalibration, error) {
	r := GrenadeCalibration{Scope: "observed_moving_damageable_contact_native", Warmup: 5}
	if !g.HasCollision() {
		return r, fmt.Errorf("missing geometry")
	}
	byFrame := map[int]quake.Snapshot{}
	var seen []quake.Snapshot
	for _, s := range rows {
		if _, ok := byFrame[s.Frame]; ok {
			return r, fmt.Errorf("duplicate frame")
		}
		byFrame[s.Frame] = s
		if len(s.Projectiles) == 0 {
			continue
		}
		if len(s.Projectiles) != 1 || s.Projectiles[0].Class != "hand_grenade" || s.Projectiles[0].ID <= 0 {
			return r, fmt.Errorf("ambiguous projectile")
		}
		if r.Entity == 0 {
			r.Entity = s.Projectiles[0].ID
		}
		if s.Projectiles[0].ID != r.Entity || len(seen) > 0 && (s.Frame != seen[len(seen)-1].Frame+1 || s.Map != seen[0].Map || s.Gravity != seen[0].Gravity) {
			return r, fmt.Errorf("projectile observation discontinuity")
		}
		for _, v := range s.Projectiles[0].Origin {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return r, fmt.Errorf("invalid projectile coordinate")
			}
		}
		seen = append(seen, s)
	}
	if len(seen) < 5 || seen[0].Gravity <= 0 {
		return r, fmt.Errorf("insufficient free flight")
	}
	a, b := seen[0], seen[4]
	r.InitialFrame = b.Frame
	r.VelocityQuantization = .3125
	for i := range r.Velocity {
		r.Velocity[i] = (b.Projectiles[0].Origin[i] - a.Projectiles[0].Origin[i]) / .4
	}
	r.Velocity[2] -= float64(b.Gravity) * .15
	launch, err := checkGrenadeLaunch(rows, r, true)
	r.Launch = &launch
	if err != nil {
		return r, err
	}
	if len(b.Enemies) != 1 || b.Enemies[0].Class != "monster_insane" || b.Enemies[0].Solid != 8290 {
		return r, fmt.Errorf("walking target missing or wrong bounds")
	}
	target := b.Enemies[0]
	// The fixture uses standing bounds from SP_misc_insane, without crucified.
	body := func(e quake.Object) grenadeBody {
		return grenadeBody{e.Origin, quake.Vec3{-16, -16, -24}, quake.Vec3{16, 16, 32}}
	}
	targetAt := func(frame int) (quake.Object, bool) {
		s, ok := byFrame[frame]
		if !ok || s.Map != b.Map || s.Gravity != b.Gravity || len(s.Enemies) != 1 {
			return quake.Object{}, false
		}
		e := s.Enemies[0]
		return e, e.ID == target.ID && e.Class == target.Class && e.Solid == target.Solid
	}
	bodyAt := func(frame int) (quake.Object, bool) {
		if target.ID > r.Entity {
			frame--
		}
		return targetAt(frame)
	}
	minY, maxY := target.Origin[1], target.Origin[1]
	for _, s := range seen {
		e, ok := targetAt(s.Frame)
		if !ok {
			return r, fmt.Errorf("target observation changed")
		}
		minY = math.Min(minY, e.Origin[1])
		maxY = math.Max(maxY, e.Origin[1])
		if s.Frame > a.Frame && s.Frame <= b.Frame {
			prev := byFrame[s.Frame-1].Projectiles[0].Origin
			tr := g.TraceProjectile(prev, s.Projectiles[0].Origin)
			collisionTarget, ok := bodyAt(s.Frame)
			if !ok {
				return r, fmt.Errorf("warmup target input missing")
			}
			_, hit := grenadeBodyHit(prev, s.Projectiles[0].Origin, body(collisionTarget))
			if !tr.Valid || tr.StartSolid || tr.Fraction != 1 || hit {
				return r, fmt.Errorf("warmup collision")
			}
		}
	}
	if maxY-minY < 2 {
		return r, fmt.Errorf("target did not move during flight")
	}
	missing := false
	tick := 0
	f := simulateGrenadeWithBodies(b.Projectiles[0].Origin, r.Velocity, float64(launch.ExpectedExplosionFrame-b.Frame)*.1, float64(b.Gravity), g.TraceProjectile, func(t int) []grenadeBody {
		e, ok := bodyAt(b.Frame + t)
		if !ok {
			missing = true
			return nil
		}
		return []grenadeBody{body(e)}
	}, func(point quake.Vec3, bounces int) {
		tick++
		s, ok := byFrame[b.Frame+tick]
		if !ok || len(s.Projectiles) != 1 || s.Projectiles[0].ID != r.Entity {
			missing = true
			return
		}
		distance := quake.Distance(point, s.Projectiles[0].Origin)
		r.MaxError = math.Max(r.MaxError, distance)
		r.Points = append(r.Points, GrenadeComparison{s.Frame, point, s.Projectiles[0].Origin, distance, bounces})
		r.Compared++
		if bounces > 0 {
			r.BounceFrames++
		}
	})
	contactFrame := b.Frame + int(math.Round(f.Seconds*10))
	if missing || f.Event != "damageable_contact" || contactFrame != launch.ExplosionFrame || r.MaxError > 2 {
		return r, fmt.Errorf("native moving contact tick/path mismatch")
	}
	// Native walk animations include zero-distance frames. Keep their actual
	// pauses as inputs; require recent movement, not movement on every tick.
	after, ok := bodyAt(contactFrame)
	recentMovement := 0.0
	for frame := max(a.Frame, contactFrame-4); frame < contactFrame; frame++ {
		before, valid := bodyAt(frame)
		if !valid {
			return r, fmt.Errorf("recent target observation missing")
		}
		recentMovement = math.Max(recentMovement, quake.Distance(before.Origin, after.Origin))
	}
	if !ok || recentMovement < 2 {
		return r, fmt.Errorf("target lacks recent native movement")
	}
	visual := f.End
	for i := range visual {
		visual[i] -= .02 * f.impactVelocity[i]
	}
	distance := quake.Distance(visual, launch.ExplosionPosition)
	r.MaxError = math.Max(r.MaxError, distance)
	r.Points = append(r.Points, GrenadeComparison{contactFrame, visual, launch.ExplosionPosition, distance, f.Bounces})
	if distance > 2 {
		return r, fmt.Errorf("native moving explosion position mismatch: %.3f", distance)
	}
	return r, nil
}
