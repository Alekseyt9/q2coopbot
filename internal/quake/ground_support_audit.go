package quake

import "math"

// GroundSupportAudit describes recorded geometry, never movement permission.
type GroundSupportAudit struct {
	Status         string               `json:"status"`
	FrictionStatus string               `json:"friction_status"`
	Points         []GroundSupportPoint `json:"points"`
	Hulls          []GroundSupportHull  `json:"hulls"`
}

type GroundSupportPoint struct {
	Offset       Vec3    `json:"offset"`
	Inset        float64 `json:"inset"`
	Known        bool    `json:"known"`
	Drop         float64 `json:"drop"`
	Normal       Vec3    `json:"normal"`
	SurfaceKnown bool    `json:"surface_known"`
	SurfaceFlags uint32  `json:"surface_flags"`
}

type GroundSupportHull struct {
	Source string      `json:"source"`
	Down   float64     `json:"down"`
	Trace  PlayerTrace `json:"trace"`
}

// InspectGroundSupport compares center/corner point support with short hull
// sweeps. Observed entities can differ from server entities or move afterward;
// even a valid contact says nothing about the requested future trajectory.
func (m *MapInfo) InspectGroundSupport(origin Vec3, ducked bool, movers []Mover, bodies []Object) GroundSupportAudit {
	r := GroundSupportAudit{Status: "unknown_geometry", FrictionStatus: m.GroundFrictionStatus(origin)}
	if m == nil || !m.MovementComplete() {
		return r
	}
	for _, v := range origin {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			r.Status = "invalid_origin"
			return r
		}
	}
	r.Status = "recorded_geometry_only"
	for _, offset := range []Vec3{{}, {-16, -16, 0}, {-16, 16, 0}, {16, -16, 0}, {16, 16, 0}} {
		p := Vec3{origin[0] + offset[0], origin[1] + offset[1], origin[2]}
		drop, normal, surface, known := m.collision.groundContactInset(p, 64, 0)
		if !known {
			drop = 0 // Unknown is carried explicitly, never serialize +Inf.
		}
		r.Points = append(r.Points, GroundSupportPoint{Offset: offset, Known: known,
			Drop: drop, Normal: normal, SurfaceKnown: surface.known, SurfaceFlags: surface.flags})
	}
	drop, normal, surface, known := m.collision.groundContactInset(origin, 64, .125)
	if !known {
		drop = 0
	}
	r.Points = append(r.Points, GroundSupportPoint{Inset: .125, Known: known,
		Drop: drop, Normal: normal, SurfaceKnown: surface.known, SurfaceFlags: surface.flags})
	for _, down := range []float64{.25, 1, 18.5, 64} {
		to := origin
		to[2] -= down
		r.Hulls = append(r.Hulls,
			GroundSupportHull{"static", down, m.TracePlayer(origin, to, ducked)},
			GroundSupportHull{"with_movers", down, m.TracePlayerWithMovers(origin, to, ducked, movers)},
			GroundSupportHull{"with_observed_bodies", down, m.TracePlayerWithEntities(origin, to, ducked, movers, bodies)})
	}
	return r
}
