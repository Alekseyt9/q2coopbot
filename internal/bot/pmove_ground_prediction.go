package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// Diagnostic only: stock dry-ground friction/acceleration and step-slide.
// Not a movement permission: dynamic contacts, ladders, server physics changes
// and PM_SnapPosition remain outside this model. Actor features are unchanged.
type PmoveGroundPrediction struct {
	Position         quake.Vec3             `json:"position"`
	Velocity         quake.Vec3             `json:"velocity"`
	Segments         [][2]quake.Vec3        `json:"segments"`
	Contacts         []quake.Vec3           `json:"contacts"`
	Model            string                 `json:"model"`
	InitialGrounded  bool                   `json:"initial_grounded"`
	SupportAmbiguous bool                   `json:"support_ambiguous"`
	Alternative      *PmoveGroundPrediction `json:"alternative,omitempty"`
}

func predictPmoveGround(s quake.Snapshot, cmd quake.UserCmd, g *quake.MapInfo) *PmoveGroundPrediction {
	if !s.PMStateKnown || s.PMType != 0 || s.PMTime != 0 || s.PMFlags&(8|16|32) != 0 ||
		!s.OnGround || s.Health <= 0 || cmd.Up > 0 || cmd.Msec == 0 || cmd.Msec > 100 ||
		s.Gravity <= 0 || g.GroundFrictionStatus(s.Self) != "dry_flat" || s.SelfVelocity[2] > 180 {
		return nil
	}
	v := s.SelfVelocity
	for _, value := range v {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil
		}
	}
	ducked := s.Ducked
	if cmd.Up < 0 {
		ducked = true
	} else if ducked {
		standing := g.TracePlayerWithEntities(s.Self, s.Self, false, s.Movers, s.CollisionBodies)
		if !standing.Valid {
			return nil
		}
		if !standing.AllSolid && !standing.StartSolid {
			ducked = false
		}
	}
	probe := s.Self
	probe[2] -= .25
	ground := g.TracePlayerWithEntities(s.Self, probe, ducked, s.Movers, s.CollisionBodies)
	if !ground.Valid || ground.AllSolid || ground.StartSolid {
		// A packet-quantized actor can appear to overlap while its server pose
		// permits escape (or vice versa). Mark applicability unknown instead
		// of assuming the native startsolid state from this single pose.
		return nil
	}
	// At a quarter-unit support boundary, the broad-phase/quantized contact
	// can disagree with the previous ground flag. Preserve both trajectories;
	// the caller must check all supported alternatives rather than guess.
	ambiguous := ground.Normal[2] < .7 || ground.Fraction+.03125/.25 >= 1-1e-8
	evaluate := func(grounded bool) *PmoveGroundPrediction {
		v := s.SelfVelocity
		dt := float64(cmd.Msec) / 1000
		speed := math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
		// Native PM_Friction uses total 3D speed before PM_AirMove zeroes Z.
		if speed < 1 {
			v[0], v[1] = 0, 0
		} else if grounded {
			scale := math.Max(0, speed-math.Max(speed, 100)*6*dt) / speed
			for axis := range v {
				v[axis] *= scale
			}
		}
		if grounded {
			v[2] = 0
		}
		yaw := float64(int16(uint16(cmd.Yaw)+uint16(s.DeltaAngles[1]))) * 2 * math.Pi / 65536
		pitch := float64(int16(uint16(cmd.Pitch)+uint16(s.DeltaAngles[0]))) * 2 * math.Pi / 65536
		pitch = math.Max(-89*math.Pi/180, math.Min(89*math.Pi/180, pitch)) / 3
		wish := quake.Vec3{math.Cos(pitch)*math.Cos(yaw)*float64(cmd.Forward) + math.Sin(yaw)*float64(cmd.Side),
			math.Cos(pitch)*math.Sin(yaw)*float64(cmd.Forward) - math.Cos(yaw)*float64(cmd.Side), 0}
		wishSpeed := math.Hypot(wish[0], wish[1])
		if wishSpeed > 0 {
			x, y := wish[0]/wishSpeed, wish[1]/wishSpeed
			limit := 300.0
			if ducked {
				limit = 100
			}
			wishSpeed = math.Min(wishSpeed, limit)
			add := wishSpeed - (v[0]*x + v[1]*y)
			if add > 0 {
				acceleration := 1.0
				if grounded {
					acceleration = 10
				}
				accel := math.Min(add, acceleration*dt*wishSpeed)
				v[0] += x * accel
				v[1] += y * accel
			}
		}
		if !grounded {
			v[2] -= float64(s.Gravity) * dt
		}
		slide, ok := g.StepSlidePlayerWithEntities(s.Self, v, dt, ducked, s.Movers, s.CollisionBodies)
		if !ok {
			return nil
		}
		return &PmoveGroundPrediction{Position: slide.Position, Velocity: slide.Velocity,
			Segments: slide.Segments, Contacts: slide.Contacts, Model: "baseq2_dry_ground_step_slide_observed_entities_diagnostic_v4", InitialGrounded: grounded}
	}
	primary := evaluate(true)
	if primary != nil && ambiguous {
		primary.SupportAmbiguous = true
		primary.Alternative = evaluate(false)
	}
	return primary

}
