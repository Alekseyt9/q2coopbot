package bot

import "q2coopbot/internal/quake"

type GrenadeGeometryEnvelope struct {
	BounceEnvelope      *GrenadeBounceEnvelope `json:"bounce_envelope,omitempty"`
	Scope               string                 `json:"scope"`
	ClearSeconds        float64                `json:"static_clear_seconds"`
	StopSeconds         float64                `json:"stop_seconds,omitempty"`
	Reason              string                 `json:"reason"`
	Authorized          bool                   `json:"authorized"`
	PostBounceCertified bool                   `json:"post_bounce_certified"`
}

// A certificate only for a static, unobstructed prefix of nominal free flight.
// On any possible collision, stop: corners/sampled normals cannot certify a
// continuous post-bounce envelope. Observed mover bounds also reject overlap.
func grenadeGeometryEnvelope(s quake.Snapshot, start, fwd, right, up quake.Vec3, g *quake.MapInfo) *GrenadeGeometryEnvelope {
	r := &GrenadeGeometryEnvelope{Scope: "static_free_flight_prefix_only", Reason: "unknown_geometry"}
	if !g.HasCollision() || s.Gravity <= 0 {
		return r
	}
	for tick := 1; tick <= 5; tick++ {
		segment := grenadeBallisticSegment(start, fwd, right, up, float64(s.Gravity), tick)
		clear, valid := g.ProjectileBoxClear(segment.lo, segment.hi)
		if !valid {
			return r
		}
		if !clear {
			r.BounceEnvelope = grenadeBounceReach(segment, tick, 5, float64(s.Gravity))
			r.StopSeconds = float64(tick) * .1
			r.Reason = "possible_static_bounce"
			return r
		}
		for _, m := range s.Movers {
			model, ok := g.Model(m.Model)
			if !ok {
				r.Reason = "unknown_mover_bounds"
				r.StopSeconds = float64(tick) * .1
				return r
			}
			bounds := grenadeBox{}
			for i := range bounds.lo {
				bounds.lo[i] = model.Min[i] + m.Origin[i]
				bounds.hi[i] = model.Max[i] + m.Origin[i]
			}
			if _, ok := grenadeBoxIntersection(segment, bounds); ok {
				r.Reason = "observed_mover_overlap"
				r.StopSeconds = float64(tick) * .1
				return r
			}
		}
		r.ClearSeconds = float64(tick) * .1
	}
	r.Reason = "free_prefix_only"
	return r
}
