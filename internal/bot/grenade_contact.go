package bot

import (
	"fmt"
	"math"
	"q2coopbot/internal/quake"
)

// Stationary native damageable target, isolated from all other sources of fire.
// This proves contact physics, not the tactics of fighting an active enemy.
func CheckGrenadeContact(rows []quake.Snapshot, g *quake.MapInfo) (GrenadeCalibration, error) {
	r := GrenadeCalibration{Scope: "stationary_damageable_contact_native", Warmup: 5}
	if !g.HasCollision() {
		return r, fmt.Errorf("missing geometry")
	}
	var seen []quake.Snapshot
	for _, s := range rows {
		if len(s.Projectiles) == 0 {
			continue
		}
		if len(s.Projectiles) != 1 || s.Projectiles[0].Class != "hand_grenade" {
			return r, fmt.Errorf("ambiguous projectile")
		}
		if r.Entity == 0 {
			r.Entity = s.Projectiles[0].ID
		}
		if s.Projectiles[0].ID != r.Entity || len(seen) > 0 && s.Frame != seen[len(seen)-1].Frame+1 {
			return r, fmt.Errorf("projectile observation discontinuity")
		}
		seen = append(seen, s)
	}
	if len(seen) != 5 {
		return r, fmt.Errorf("contact fixture requires five free-flight sightings, got%d", len(seen))
	}
	a, b := seen[0], seen[4]
	if a.Gravity <= 0 || b.Gravity != a.Gravity {
		return r, fmt.Errorf("unknown gravity")
	}
	for i := 1; i < 5; i++ {
		tr := g.TraceProjectile(seen[i-1].Projectiles[0].Origin, seen[i].Projectiles[0].Origin)
		if !tr.Valid || tr.StartSolid || tr.Fraction != 1 {
			return r, fmt.Errorf("warmup collision")
		}
	}
	r.InitialFrame = b.Frame
	r.VelocityQuantization = 0.3125
	for i := range r.Velocity {
		r.Velocity[i] = (b.Projectiles[0].Origin[i] - a.Projectiles[0].Origin[i]) / 0.4
	}
	r.Velocity[2] -= float64(b.Gravity) * 0.15
	launch, err := checkGrenadeLaunch(rows, r, true)
	r.Launch = &launch
	if err != nil {
		return r, err
	}
	if len(b.Enemies) != 2 {
		return r, fmt.Errorf("native contact targets missing")
	}
	bodies := []grenadeBody{}
	for _, target := range b.Enemies {
		if target.Class != "monster_insane" || target.Solid != 8290 {
			return r, fmt.Errorf("unexpected hanging target bounds")
		}
		for _, s := range seen {
			found := false
			for _, e := range s.Enemies {
				if e.ID == target.ID && e.Class == target.Class && e.Origin == target.Origin && e.Solid == target.Solid {
					found = true
				}
			}
			if !found || len(s.Enemies) != 2 {
				return r, fmt.Errorf("contact target moved or changed")
			}
		}
		// Exact hanging-insane bounds from SP_misc_insane, fixture-specific.
		// SP_misc_insane links its initial standing box before changing the
		// hanging bounds. Packed solid8290 is not the exact contact shape.
		bodies = append(bodies, grenadeBody{target.Origin, quake.Vec3{-16, 0, 0}, quake.Vec3{16, 8, 32}})
	}
	f := grenadeFlight(b.Projectiles[0].Origin, r.Velocity, launch.Timer, float64(b.Gravity), g.TraceProjectile, bodies)
	if f.Event != "damageable_contact" || f.Bounces != 0 || b.Frame+int(math.Round(f.Seconds*10)) != launch.ExplosionFrame {
		return r, fmt.Errorf("native event does not match predicted contact tick")
	}
	visual := f.End
	for i := range visual {
		v := r.Velocity[i]
		if i == 2 {
			v -= float64(b.Gravity) * f.Seconds
		}
		visual[i] -= 0.02 * v
	}
	r.MaxError = quake.Distance(visual, launch.ExplosionPosition)
	r.Points = []GrenadeComparison{{launch.ExplosionFrame, visual, launch.ExplosionPosition, r.MaxError, 0}}
	if r.MaxError > 2 {
		return r, fmt.Errorf("native contact explosion position mismatch: %.3f", r.MaxError)
	}
	return r, nil
}
