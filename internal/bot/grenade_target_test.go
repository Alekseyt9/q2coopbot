package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestGrenadeMovingTargetContact(t *testing.T) {
	e := quake.Object{ID: 9, Class: "monster_insane", Origin: quake.Vec3{80, 60, 0}, Solid: 8290}
	m := GrenadeTargetMotion{Frame: 20, Entity: 9, Class: e.Class, Velocity: quake.Vec3{0, -200, 0}}
	hit := grenadeTargetContact(quake.Vec3{}, quake.Vec3{400, 0, 200}, 800, emptyGrenadeTrace, []quake.Object{e}, []GrenadeTargetMotion{m})
	if hit == nil || hit.Entity != 9 || hit.Seconds > .3 || hit.Point[0] < 40 || hit.Point[0] > 104 {
		t.Fatal(hit)
	}
	m.Velocity[1] = 200
	if hit := grenadeTargetContact(quake.Vec3{}, quake.Vec3{400, 0, 200}, 800, emptyGrenadeTrace, []quake.Object{e}, []GrenadeTargetMotion{m}); hit != nil {
		t.Fatal("Departing target", hit)
	}
	m.Velocity[1] = -200
	if hit := grenadeTargetContact(quake.Vec3{}, quake.Vec3{400, 0, 200}, 800, func(_, _ quake.Vec3) quake.PointTrace { return quake.PointTrace{} }, []quake.Object{e}, []GrenadeTargetMotion{m}); hit != nil {
		t.Fatal("Unknown geometry", hit)
	}
	wall := func(from, to quake.Vec3) quake.PointTrace {
		if to[0] > 20 {
			f := (20 - from[0]) / (to[0] - from[0])
			end := from
			for i := range end {
				end[i] += f * (to[i] - from[i])
			}
			return quake.PointTrace{Valid: true, Fraction: f, End: end, Normal: quake.Vec3{-1, 0, 0}}
		}
		return emptyGrenadeTrace(from, to)
	}
	if hit := grenadeTargetContact(quake.Vec3{}, quake.Vec3{400, 0, 200}, 800, wall, []quake.Object{e}, []GrenadeTargetMotion{m}); hit != nil {
		t.Fatal("Target behind wall", hit)
	}
}

func TestGrenadeTargetMotionRequiresCurrentHistory(t *testing.T) {
	s := quake.Snapshot{Frame: 20, Enemies: []quake.Object{{ID: 9, Class: "monster_insane"}}}
	valid := enemyMotion{frame: 20, class: "monster_insane", stable: true, velocityKnown: true, velocity: quake.Vec3{0, 100, 0}}
	p := &Planner{enemyMotion: map[int]enemyMotion{9: valid}}
	if len(p.grenadeTargetMotions(s)) != 1 {
		t.Fatal("Missing current motion")
	}
	for _, change := range []func(*enemyMotion){func(m *enemyMotion) { m.frame-- }, func(m *enemyMotion) { m.class = "monster_tank" }, func(m *enemyMotion) { m.stable = false }, func(m *enemyMotion) { m.velocityKnown = false }, func(m *enemyMotion) { m.velocity = quake.Vec3{} }} {
		m := valid
		change(&m)
		p.enemyMotion[9] = m
		if len(p.grenadeTargetMotions(s)) != 0 {
			t.Fatal("Stale/unknown/turning motion accepted", m)
		}
	}
}
