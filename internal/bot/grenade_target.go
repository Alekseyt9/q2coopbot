package bot

import "q2coopbot/internal/quake"

type GrenadeTargetMotion struct {
	Frame    int        `json:"frame"`
	Entity   int        `json:"entity"`
	Class    string     `json:"class"`
	Velocity quake.Vec3 `json:"velocity"`
}

type GrenadeTargetContact struct {
	Entity  int        `json:"entity"`
	Seconds float64    `json:"seconds"`
	Point   quake.Vec3 `json:"point"`
}

func (p *Planner) grenadeTargetMotions(s quake.Snapshot) []GrenadeTargetMotion {
	var motions []GrenadeTargetMotion
	for _, e := range s.Enemies {
		m, ok := p.enemyMotion[e.ID]
		if ok && m.frame == s.Frame && m.class == e.Class && m.velocityKnown && m.stable && quake.Distance(m.velocity, quake.Vec3{}) >= 10 {
			motions = append(motions, GrenadeTargetMotion{s.Frame, e.ID, e.Class, m.velocity})
		}
	}
	return motions
}

// The packed solid is approximate (some monsters change bounds after linking).
// Add eight units and use this only as an early-explosion hazard hypothesis.
func grenadeTargetContact(start, velocity quake.Vec3, gravity float64, trace func(quake.Vec3, quake.Vec3) quake.PointTrace, enemies []quake.Object, motions []GrenadeTargetMotion) *GrenadeTargetContact {
	var first *GrenadeTargetContact
	for _, m := range motions {
		for _, e := range enemies {
			if e.ID != m.Entity || e.Class != m.Class || e.Solid == 0 || e.Solid == 31 {
				continue
			}
			x := float64(e.Solid&31)*8 + 8
			down := float64((e.Solid>>5)&31)*8 + 8
			up := float64((e.Solid>>10)&63)*8 - 32 + 8
			seconds, point := grenadeMovingBodyContact(start, velocity, gravity, trace, grenadeBody{e.Origin, quake.Vec3{-x, -x, -down}, quake.Vec3{x, x, up}}, m.Velocity)
			if seconds > 0 && (first == nil || seconds < first.Seconds) {
				first = &GrenadeTargetContact{e.ID, seconds, point}
			}
		}
	}
	return first
}
