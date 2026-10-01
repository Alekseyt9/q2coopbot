package bot

import (
	"fmt"
	"math"
	"q2coopbot/internal/quake"
)

type GrenadeComparison struct {
	Frame               int `json:"frame"`
	Predicted, Observed quake.Vec3
	Error               float64 `json:"error"`
	Bounces             int     `json:"bounces"`
}
type GrenadeCalibration struct {
	BounceEnvelope         *GrenadeBounceEnvelope `json:"bounce_envelope,omitempty"`
	Launch                 *GrenadeLaunchCheck    `json:"launch,omitempty"`
	Scope                  string                 `json:"scope"`
	Entity                 int                    `json:"entity"`
	InitialFrame           int                    `json:"initial_frame"`
	Velocity               quake.Vec3             `json:"inferred_velocity"`
	Warmup                 int                    `json:"warmup_sightings"`
	VelocityQuantization   float64                `json:"velocity_quantization_radius"`
	Compared, BounceFrames int
	MaxError               float64             `json:"max_error"`
	Points                 []GrenadeComparison `json:"points"`
}

// Calibrate an isolated, single-grenade fixture. Infer velocity once from the
// first five consecutive free-flight sightings; subsequent points are held out. Never
// refit after a bounce or use disappearance as evidence of fuse/contact.
func CalibrateGrenade(rows []quake.Snapshot, g *quake.MapInfo) (GrenadeCalibration, error) {
	r := GrenadeCalibration{Scope: "observed_initial_velocity_static_path_only"}
	if !g.HasCollision() {
		return r, fmt.Errorf("missing BSP collision")
	}
	var sightings []struct {
		frame   int
		point   quake.Vec3
		gravity int16
	}
	mapName := ""
	for _, s := range rows {
		if len(s.Projectiles) > 1 {
			return r, fmt.Errorf("ambiguous multiple projectiles")
		}
		if len(s.Projectiles) == 0 {
			continue
		}
		o := s.Projectiles[0]
		if o.Class != "hand_grenade" {
			continue
		}
		if o.ID <= 0 {
			return r, fmt.Errorf("invalid projectile identity")
		}
		for _, v := range o.Origin {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return r, fmt.Errorf("invalid projectile coordinate")
			}
		}
		if mapName == "" {
			mapName = s.Map
		}
		if s.Map != mapName {
			return r, fmt.Errorf("map changed during observation")
		}
		if r.Entity == 0 {
			r.Entity = o.ID
		}
		if o.ID != r.Entity {
			return r, fmt.Errorf("projectile identity changed")
		}
		if len(sightings) > 0 && s.Frame != sightings[len(sightings)-1].frame+1 {
			return r, fmt.Errorf("projectile observation gap or duplicate")
		}
		sightings = append(sightings, struct {
			frame   int
			point   quake.Vec3
			gravity int16
		}{s.Frame, o.Origin, s.Gravity})
	}
	const warmup = 5
	if len(sightings) < warmup+8 {
		return r, fmt.Errorf("insufficient consecutive sightings: %d", len(sightings))
	}
	a, b := sightings[0], sightings[warmup-1]
	if a.gravity <= 0 || a.gravity != b.gravity || quake.Distance(a.point, b.point) < 4 {
		return r, fmt.Errorf("initial velocity cannot be inferred")
	}
	for i := 1; i < warmup; i++ {
		tr := g.TraceProjectile(sightings[i-1].point, sightings[i].point)
		if !tr.Valid || tr.StartSolid || tr.Fraction != 1 || sightings[i].gravity != a.gravity {
			return r, fmt.Errorf("warmup includes collision or gravity change")
		}
	}
	r.InitialFrame = b.frame
	r.Warmup = warmup
	r.VelocityQuantization = 0.125 / (0.1 * float64(warmup-1))
	for i := range r.Velocity {
		r.Velocity[i] = (b.point[i] - a.point[i]) / (0.1 * float64(warmup-1))
	}
	// Average vertical velocity lies halfway through the four gravity updates.
	r.Velocity[2] -= float64(b.gravity) * 0.1 * float64(warmup-2) / 2
	path := []struct {
		point   quake.Vec3
		bounces int
	}{}
	simulateGrenade(b.point, r.Velocity, 3.2, float64(b.gravity), g.TraceProjectile, nil, func(p quake.Vec3, n int) {
		path = append(path, struct {
			point   quake.Vec3
			bounces int
		}{p, n})
	})
	for i, s := range sightings[warmup:] {
		if i >= len(path) || s.gravity != b.gravity {
			return r, fmt.Errorf("unmodeled path or gravity change")
		}
		v := path[i]
		e := quake.Distance(v.point, s.point)
		r.Points = append(r.Points, GrenadeComparison{s.frame, v.point, s.point, e, v.bounces})
		r.MaxError = max(r.MaxError, e)
		r.Compared++
		if v.bounces > 0 {
			r.BounceFrames++
		}
	}
	if r.Compared < 8 || r.BounceFrames < 3 || r.MaxError > 4 {
		return r, fmt.Errorf("calibration rejected: compared%d bounced%d max_error%.3f", r.Compared, r.BounceFrames, r.MaxError)
	}
	launch, err := checkGrenadeLaunch(rows, r)
	r.Launch = &launch
	if err != nil {
		return r, err
	}
	if err := checkGrenadeBounceReach(&r, float64(b.gravity)); err != nil {
		return r, err
	}
	return r, nil
}
