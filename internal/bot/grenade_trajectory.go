package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// A diagnostic candidate, never authorization to arm. Random samples are not a
// bound on all trajectories; server gravity, hand and fuse need confirmation.
type GrenadePrediction struct {
	Scope   string          `json:"scope"`
	Reason  string          `json:"reason"`
	Samples []GrenadeFlight `json:"samples,omitempty"`
}
type GrenadeFlight struct {
	End     quake.Vec3 `json:"end"`
	Seconds float64    `json:"seconds"`
	Bounces int        `json:"bounces"`
	Event   string     `json:"event"`
	Risk    string     `json:"risk"`
}
type grenadeBody struct{ origin, mins, maxs quake.Vec3 }

// Baseq2 SV_Physics_Toss: think before motion, gravity before sweep, one
// collision per 100ms tick (unused tick time is discarded), overbounce1.5.
func grenadeFlight(start, velocity quake.Vec3, fuse, gravity float64, trace func(quake.Vec3, quake.Vec3) quake.PointTrace, bodies []grenadeBody) GrenadeFlight {
	r := GrenadeFlight{End: start, Event: "invalid_input"}
	for _, v := range []float64{fuse, gravity, start[0], start[1], start[2], velocity[0], velocity[1], velocity[2]} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return r
		}
	}
	if fuse <= 0 || fuse > 3.2 || gravity <= 0 || gravity > 2000 || trace == nil {
		return r
	}
	settled := false
	for tick := 1; tick <= 33; tick++ {
		r.Seconds = float64(tick) * 0.1
		if r.Seconds+1e-8 >= fuse {
			r.Event = "fuse"
			return r
		}
		if settled {
			continue
		}
		velocity[2] -= gravity * 0.1
		to := r.End
		for i := range to {
			velocity[i] = math.Max(-2000, math.Min(2000, velocity[i]))
			to[i] += velocity[i] * 0.1
		}
		tr := trace(r.End, to)
		if !tr.Valid || tr.StartSolid {
			r.Event = "unknown_collision"
			return r
		}
		first := tr.Fraction
		contact := false
		for _, b := range bodies {
			if f, ok := grenadeBodyHit(r.End, to, b); ok && f <= first {
				first = f
				contact = true
			}
		}
		if contact {
			for i := range to {
				r.End[i] += (to[i] - r.End[i]) * first
			}
			r.Event = "damageable_contact"
			return r
		}
		r.End = tr.End
		if tr.Fraction == 1 {
			continue
		}
		if tr.Sky {
			r.Event = "sky"
			return r
		}
		dot := 0.0
		for i := range velocity {
			dot += velocity[i] * tr.Normal[i]
		}
		for i := range velocity {
			velocity[i] -= 1.5 * dot * tr.Normal[i]
			if math.Abs(velocity[i]) < 0.1 {
				velocity[i] = 0
			}
		}
		r.Bounces++
		if tr.Normal[2] > 0.7 && velocity[2] < 60 {
			settled = true
			velocity = quake.Vec3{}
		}
	}
	r.Event = "unknown_fuse"
	return r
}

func grenadeBodyHit(from, to quake.Vec3, b grenadeBody) (float64, bool) {
	enter, leave := 0.0, 1.0
	for i := range from {
		lo, hi := b.origin[i]+b.mins[i], b.origin[i]+b.maxs[i]
		d := to[i] - from[i]
		if math.Abs(d) < 1e-9 {
			if from[i] < lo || from[i] > hi {
				return 0, false
			}
			continue
		}
		a, z := (lo-from[i])/d, (hi-from[i])/d
		if a > z {
			a, z = z, a
		}
		enter = math.Max(enter, a)
		leave = math.Min(leave, z)
		if enter > leave {
			return 0, false
		}
	}
	return enter, true
}

func (p *Planner) predictGrenadeCandidate(s quake.Snapshot) *GrenadePrediction {
	r := &GrenadePrediction{Scope: "nominal_right_hand_timer3_diagnostic", Reason: "not_authorized"}
	g := p.World.Geometry
	if !g.HasCollision() || s.Gravity <= 0 {
		r.Reason = "unknown_geometry_or_gravity"
		return r
	}
	// Only idle candidates: an observed gunframe alone does not reveal the
	// remaining fuse of an externally primed grenade.
	if s.GunFrame < 16 {
		r.Reason = "armed_fuse_unknown"
		return r
	}
	if len(s.Obstacles) > len(s.Enemies) {
		r.Reason = "dynamic_collision_unmodeled"
		return r
	}
	trace := func(from, to quake.Vec3) quake.PointTrace {
		tr := g.TraceProjectile(from, to)
		for _, m := range s.Movers {
			model, ok := g.Model(m.Model)
			if !ok {
				return quake.PointTrace{}
			}
			// Observed bounds only reject; they never certify a mover bounce.
			if f, hit := grenadeBodyHit(from, to, grenadeBody{m.Origin, model.Min, model.Max}); hit && f <= tr.Fraction {
				return quake.PointTrace{}
			}
		}
		return tr
	}
	bodies := []grenadeBody{}
	for _, e := range s.Enemies {
		if e.Solid == 0 || e.Solid == 31 {
			r.Reason = "unknown_enemy_bounds"
			return r
		}
		x := float64(e.Solid&31) * 8
		down := float64((e.Solid>>5)&31) * 8
		up := float64((e.Solid>>10)&63)*8 - 32
		bodies = append(bodies, grenadeBody{e.Origin, quake.Vec3{-x, -x, -down}, quake.Vec3{x, x, up}})
	}
	if s.Teammate != nil {
		bodies = append(bodies, grenadeBody{*s.Teammate, quake.Vec3{-16, -16, -24}, quake.Vec3{16, 16, 32}})
	}
	pitch, yaw := float64(s.ViewAngles[0])*2*math.Pi/65536, float64(s.ViewAngles[1])*2*math.Pi/65536
	sp, cp := math.Sincos(pitch)
	sy, cy := math.Sincos(yaw)
	fwd, right, up := quake.Vec3{cp * cy, cp * sy, -sp}, quake.Vec3{sy, -cy, 0}, quake.Vec3{sp * cy, sp * sy, cp}
	start := s.Self
	viewheight := 22.0
	if s.Ducked {
		viewheight = -2
	}
	for i := range start {
		start[i] += 8*fwd[i] + 8*right[i]
	}
	start[2] += viewheight - 8
	for _, uj := range []float64{-10, 0, 10} {
		for _, rj := range []float64{-10, 0, 10} {
			v := quake.Vec3{}
			for i := range v {
				v[i] = 400*fwd[i] + (200+uj)*up[i] + rj*right[i]
			}
			flight := grenadeFlight(start, v, 3, float64(s.Gravity), trace, bodies)
			flight.Risk = "unproven"
			if flight.Event == "fuse" || flight.Event == "damageable_contact" {
				if quake.Distance(flight.End, s.Self) <= 197+quake.Distance(quake.Vec3{}, s.SelfVelocity)*flight.Seconds {
					flight.Risk = "self_reachable_blast"
				}
				if s.Teammate != nil && quake.Distance(flight.End, *s.Teammate) <= 197+400*flight.Seconds {
					flight.Risk = "teammate_reachable_blast"
				}
				if s.Teammate == nil && s.LastTeammate != nil {
					flight.Risk = "unseen_teammate"
				}
			}
			r.Samples = append(r.Samples, flight)
		}
	}
	return r
}
