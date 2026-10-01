package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

type GrenadeContactRange struct {
	Entity        int        `json:"entity"`
	Kind          string     `json:"kind"`
	Seconds       float64    `json:"seconds"`
	Min           quake.Vec3 `json:"min"`
	Max           quake.Vec3 `json:"max"`
	SelfBlast     bool       `json:"self_blast"`
	TeammateBlast bool       `json:"teammate_blast"`
}

type GrenadeEnvelope struct {
	Geometry          *GrenadeGeometryEnvelope `json:"geometry,omitempty"`
	Frame             int                      `json:"frame"`
	Scope             string                   `json:"scope"`
	Status            string                   `json:"status"`
	Authorized        bool                     `json:"authorized"`
	Horizon           float64                  `json:"horizon_seconds"`
	AssumedAxisSpeed  float64                  `json:"assumed_axis_speed"`
	GeometryCertified bool                     `json:"geometry_certified"`
	Contacts          []GrenadeContactRange    `json:"contacts,omitempty"`
}

func grenadeBallisticSegment(start, fwd, right, up quake.Vec3, gravity float64, tick int) grenadeBox {
	previous := grenadeBallisticBox(start, fwd, right, up, gravity, tick-1)
	current := grenadeBallisticBox(start, fwd, right, up, gravity, tick)
	for i := range current.lo {
		current.lo[i] = math.Min(previous.lo[i], current.lo[i])
		current.hi[i] = math.Max(previous.hi[i], current.hi[i])
	}
	return current
}

type grenadeBox struct{ lo, hi quake.Vec3 }

// Bound the continuous native up/right jitter of a nominal timer3/speed400
// free flight, including segment interiors and snapshot rounding. Geometry,
// bounce uncertainty and later flight are not certified. This can only reject;
// neither no contact nor the assumed body speed is authorization to throw.
func grenadeEarlyEnvelope(s quake.Snapshot, start, fwd, right, up quake.Vec3) *GrenadeEnvelope {
	r := &GrenadeEnvelope{Frame: s.Frame, Scope: "free_flight_500ms_reject_only", Status: "unproven", Horizon: .5, AssumedAxisSpeed: 400}
	type actor struct {
		id   int
		kind string
		body grenadeBody
	}
	actors := []actor{}
	for _, e := range s.Enemies {
		if e.Solid == 0 || e.Solid == 31 {
			continue
		}
		x := float64(e.Solid&31)*8 + 8
		down := float64((e.Solid>>5)&31)*8 + 8
		top := float64((e.Solid>>10)&63)*8 - 32 + 8
		actors = append(actors, actor{e.ID, "target", grenadeBody{e.Origin, quake.Vec3{-x, -x, -down}, quake.Vec3{x, x, top}}})
	}
	if s.Teammate != nil {
		actors = append(actors, actor{s.TeammateEntity, "teammate", grenadeBody{*s.Teammate, quake.Vec3{-24, -24, -32}, quake.Vec3{24, 24, 40}}})
	}
	previous := grenadeBallisticBox(start, fwd, right, up, float64(s.Gravity), 0)
	for tick := 1; tick <= 5; tick++ {
		seconds := float64(tick) * .1
		current := grenadeBallisticBox(start, fwd, right, up, float64(s.Gravity), tick)
		segment := current
		for i := range segment.lo {
			segment.lo[i] = math.Min(previous.lo[i], current.lo[i])
			segment.hi[i] = math.Max(previous.hi[i], current.hi[i])
		}
		for _, a := range actors {
			reach := r.AssumedAxisSpeed * seconds
			body := grenadeBox{}
			for i := range body.lo {
				body.lo[i] = a.body.origin[i] + a.body.mins[i] - reach
				body.hi[i] = a.body.origin[i] + a.body.maxs[i] + reach
			}
			contact, ok := grenadeBoxIntersection(segment, body)
			if !ok {
				continue
			}
			hit := GrenadeContactRange{Entity: a.id, Kind: a.kind, Seconds: seconds, Min: contact.lo, Max: contact.hi}
			// Euclidean reach is expanded to match the assumed per-axis speed.
			blastReach := 197 + math.Sqrt(3)*reach
			hit.SelfBlast = grenadeBoxDistance(s.Self, contact) <= blastReach
			hit.TeammateBlast = s.Teammate != nil && grenadeBoxDistance(*s.Teammate, contact) <= blastReach
			if hit.SelfBlast || hit.TeammateBlast {
				r.Status = "reject_early_blast"
			}
			r.Contacts = append(r.Contacts, hit)
		}
		previous = current
	}
	if s.Teammate == nil && s.LastTeammate != nil {
		r.Status = "reject_unseen_teammate"
	}
	return r
}

func grenadeBallisticBox(start, fwd, right, up quake.Vec3, gravity float64, tick int) grenadeBox {
	t := float64(tick) * .1
	b := grenadeBox{}
	for i := range start {
		center := start[i] + t*(400*fwd[i]+200*up[i])
		if i == 2 {
			center -= gravity * .01 * float64(tick*(tick+1)) / 2
		}
		radius := 10*t*(math.Abs(up[i])+math.Abs(right[i])) + .125
		b.lo[i] = center - radius
		b.hi[i] = center + radius
	}
	return b
}

func grenadeBoxIntersection(a, b grenadeBox) (grenadeBox, bool) {
	c := grenadeBox{}
	for i := range c.lo {
		c.lo[i] = math.Max(a.lo[i], b.lo[i])
		c.hi[i] = math.Min(a.hi[i], b.hi[i])
		if c.lo[i] > c.hi[i] {
			return c, false
		}
	}
	return c, true
}

func grenadeBoxDistance(point quake.Vec3, b grenadeBox) float64 {
	squared := 0.0
	for i := range point {
		d := math.Max(b.lo[i]-point[i], math.Max(0, point[i]-b.hi[i]))
		squared += d * d
	}
	return math.Sqrt(squared)
}
