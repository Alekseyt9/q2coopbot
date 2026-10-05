package policy

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestEnvironmentUnavailableUsesMasks(t *testing.T) {
	s := quake.Snapshot{Frame: 1, Health: 100, Pickups: []quake.Object{{ID: 1, Class: "item_health"}}}
	o := Observe(s, Identity{Frame: 1}, quake.UserCmd{})
	EnrichEnvironment(&o, s, nil)
	if o.Geometry != nil || o.Pickups != nil || o.Projectiles != nil || o.Movers != nil {
		t.Fatal("invented geometry/object observations", o)
	}
}

func TestEnvironmentProbesAndObjectsUseActualBSPVisibility(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	s := quake.Snapshot{Frame: 10, Health: 100, OnGround: true, Self: quake.Vec3{32, -224, 24.125},
		Projectiles: []quake.Object{{ID: 1, Class: "blaster_bolt", Origin: quake.Vec3{64, -224, 32}}, {ID: 2, Class: "grenade", Origin: quake.Vec3{32, -224, -100}}},
		Pickups:     []quake.Object{{ID: 3, Class: "item_health", HealthAmount: 25, Origin: quake.Vec3{100, -224, 24.125}}, {ID: 4, Class: "item_health", Origin: quake.Vec3{32, -224, -100}}},
		Barrels:     []quake.Object{{ID: 5, Class: "misc_explobox", Origin: quake.Vec3{64, -224, 24.125}, Solid: 7266}},
		Beams:       []quake.BeamObservation{{ID: 6, Frame: 10, Origin: quake.Vec3{64, -224, 32}, End: quake.Vec3{80, -224, 32}}}}
	o := Observe(s, Identity{Frame: 10}, quake.UserCmd{})
	EnrichEnvironment(&o, s, &g)
	if o.Geometry == nil || len(o.Geometry.Probes) != 8 || o.Geometry.GroundDrop == nil {
		t.Fatal("missing geometry", o.Geometry)
	}
	if o.Projectiles == nil || len(*o.Projectiles) != 1 || (*o.Projectiles)[0].ID != 1 || o.Pickups == nil || len(*o.Pickups) != 1 || *(*o.Pickups)[0].HealthAmount != 25 || o.Props == nil || len(*o.Props) != 1 {
		t.Fatal("hidden or unavailable objects exported", o)
	}
	blocked := false
	for _, p := range o.Geometry.Probes {
		if p.RayDistance < 256 {
			blocked = true
		}
		if p.RayDistance < 0 || p.RayDistance > 256 {
			t.Fatal(p)
		}
	}
	if !blocked {
		t.Fatal("walls absent from probes")
	}
	if o.Beams == nil || len(*o.Beams) != 1 || (*o.Beams)[0].Direction == nil || (*o.Beams)[0].Direction[0] != 1 {
		t.Fatal("beam direction unavailable", o.Beams)
	}
}

func TestProjectileVelocityDirectionResetAndHistoryIsolation(t *testing.T) {
	h := History{}
	a := historyObservation(10)
	p := []Enemy{{ID: 90, Class: "rocket", Relative: quake.Vec3{100, 0, 0}}}
	a.Projectiles = &p
	h.Enrich(&a)
	if p[0].Velocity != nil {
		t.Fatal("invented first projectile velocity")
	}
	b := historyObservation(11)
	q := []Enemy{{ID: 90, Class: "rocket", Relative: quake.Vec3{198, 0, 0}}}
	b.Projectiles = &q
	h.Enrich(&b)
	if q[0].Velocity == nil || q[0].Velocity[0] != 1000 || q[0].MotionDirection == nil || q[0].MotionDirection[0] != 1 {
		t.Fatal("wrong game-time projectile velocity", q)
	}
	*q[0].MotionDirection = quake.Vec3{0, 1, 0}
	c := historyObservation(12)
	r := []Enemy{{ID: 90, Class: "rocket", Relative: quake.Vec3{296, 0, 0}}}
	c.Projectiles = &r
	h.Enrich(&c)
	if r[0].MotionDirection[0] != 1 || (*c.History[1].Projectiles)[0].MotionDirection[0] != 1 {
		t.Fatal("provider mutation poisoned projectile history")
	}
	d := historyObservation(13)
	empty := []Enemy{}
	d.Projectiles = &empty
	h.Enrich(&d)
	e := historyObservation(14)
	r = append([]Enemy{}, r...)
	e.Projectiles = &r
	h.Enrich(&e)
	if r[0].Velocity != nil || r[0].MotionDirection != nil {
		t.Fatal("projectile velocity crossed occlusion")
	}
}
