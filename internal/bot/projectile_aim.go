package bot

import (
	"math"
	"q2coopbot/internal/quake"
	"strings"
)

type enemyMotion struct {
	origin, velocity quake.Vec3
	class            string
	frame            int
	stable           bool
	velocityKnown    bool
	velocityChange   float64
	steadyFrames     int
}

func projectileFixtureWeaponReady(fixture, weapon string) bool {
	return fixture == "projectile_blaster" && weapon == "Blaster" ||
		fixture == "projectile_hyper" && strings.Contains(weapon, "/v_hyperb/")
}

func (p *Planner) observeEnemyMotion(s quake.Snapshot) {
	next := make(map[int]enemyMotion, len(s.Enemies))
	for _, e := range s.Enemies {
		m := enemyMotion{origin: e.Origin, class: e.Class, frame: s.Frame}
		old, ok := p.enemyMotion[e.ID]
		if ok && p.World.Snapshot.Map == s.Map && s.Health > 0 && p.World.Snapshot.Health > 0 && old.class == e.Class && old.frame+1 == s.Frame {
			v := quake.Vec3{(e.Origin[0] - old.origin[0]) * 10, (e.Origin[1] - old.origin[1]) * 10, 0}
			if math.Hypot(v[0], v[1]) <= 400 && math.Abs(e.Origin[2]-old.origin[2]) <= 2 {
				m.velocity = v
				m.velocityKnown = true
				m.velocityChange = math.Hypot(v[0]-old.velocity[0], v[1]-old.velocity[1])
				m.stable = old.velocityKnown && m.velocityChange <= 80 && v[0]*old.velocity[0]+v[1]*old.velocity[1] >= 0
				m.steadyFrames = 1
				// Allow snapshot quantization, but restart the history on a turn
				// or acceleration instead of extrapolating a newly chosen course.
				if old.velocityKnown && m.velocityChange <= 8 {
					m.steadyFrames = min(old.steadyFrames+1, 8)
				}
			}
		}
		next[e.ID] = m
	}
	p.enemyMotion = next
}

// Both player blaster variants use speed 1000 in baseq2's Blaster_Fire.
// A bounded horizontal intercept is used only for stable, consecutive sightings.
func (p *Planner) projectileAim(s quake.Snapshot, e quake.Object) (quake.Vec3, float64) {
	point := e.AimPoint()
	if p.TestDisableProjectileLead {
		return point, 0
	}
	weapon := strings.ToLower(s.Weapon)
	if weapon != "blaster" && !strings.Contains(weapon, "/v_hyperb/") {
		return point, 0
	}
	m, ok := p.enemyMotion[e.ID]
	if !ok || !m.stable || m.frame != s.Frame || m.class != e.Class || math.Hypot(m.velocity[0], m.velocity[1]) < 10 {
		return point, 0
	}
	t, ok := interceptTime(s.EyePoint(), point, m.velocity, 1000)
	if !ok || t > 0.75 || math.Hypot(m.velocity[0], m.velocity[1])*t > 128 {
		return point, 0
	}
	// The same frame-to-frame speed change is much more uncertain at a long
	// flight time. Reject lead if that acceleration would add over eight units
	// of displacement. This is a conservative heuristic, not an error guarantee.
	if !projectileMotionReliable(m.velocityChange, t) {
		return point, 0
	}
	if !projectileHistoryReliable(m.steadyFrames, t) {
		return point, 0
	}
	lead := point
	lead[0] += m.velocity[0] * t
	lead[1] += m.velocity[1] * t
	g := p.World.Geometry
	if !g.HasCollision() || !g.ClearShot(point, lead) || !g.ClearShot(s.EyePoint(), lead) || g.DoorShotBlocked(s.Movers, point, lead) || g.DoorShotBlocked(s.Movers, s.EyePoint(), lead) {
		return point, 0
	}
	return lead, t
}

func projectileMotionReliable(velocityChange, flight float64) bool {
	return 0.5*(velocityChange*10)*flight*flight <= 8
}

func projectileHistoryReliable(steadyFrames int, flight float64) bool {
	// Stable observations already cover two intervals, with their acceleration
	// checked separately. Longer flights need a correspondingly steady course.
	return float64(max(2, steadyFrames))*0.1 >= flight
}

func interceptTime(from, to, velocity quake.Vec3, speed float64) (float64, bool) {
	r := quake.Vec3{to[0] - from[0], to[1] - from[1], to[2] - from[2]}
	a, b, c := -speed*speed, 0.0, 0.0
	for i := 0; i < 3; i++ {
		a += velocity[i] * velocity[i]
		b += 2 * r[i] * velocity[i]
		c += r[i] * r[i]
	}
	if speed <= 0 || c <= 0 {
		return 0, false
	}
	if math.Abs(a) < 1e-9 {
		if b >= 0 {
			return 0, false
		}
		return -c / b, true
	}
	d := b*b - 4*a*c
	if d < 0 {
		return 0, false
	}
	t1, t2 := (-b-math.Sqrt(d))/(2*a), (-b+math.Sqrt(d))/(2*a)
	t := math.Inf(1)
	if t1 > 0 {
		t = t1
	}
	if t2 > 0 && t2 < t {
		t = t2
	}
	return t, !math.IsInf(t, 1) && !math.IsNaN(t)
}
