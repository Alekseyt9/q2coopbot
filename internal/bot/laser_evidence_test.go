package bot

import (
	"os"
	"testing"

	"q2coopbot/internal/quake"
)

func TestLaserRemovalRequiresStableViewAndExpires(t *testing.T) {
	root := "../../workspace/runtime/q2go/baseq2"
	if _, err := os.Stat(root + "/maps/base3.bsp"); err != nil {
		t.Skip("local base3 assets unavailable")
	}
	geometry, err := quake.LoadMap(root, "base3")
	if err != nil {
		t.Fatal(err)
	}
	origins := geometry.StaticLethalLaserOrigins()
	if len(origins) != 2 {
		t.Fatalf("want two emitters, got %d", len(origins))
	}
	p := &Planner{World: World{Map: "base3", Geometry: &geometry}}
	self := quake.Vec3{-730, -455, -279.875}
	active := quake.Snapshot{Map: "base3", Frame: 130, Self: self,
		Beams: []quake.BeamObservation{{ID: 109, Origin: origins[0]}, {ID: 110, Origin: origins[1]}}}
	p.observeLasers(quake.Snapshot{}, active)
	if len(p.laserOffOrigins(130)) != 0 {
		t.Fatal("active beam considered off")
	}
	removed := quake.Snapshot{Map: "base3", Frame: 131, Self: self, RemovedEntities: []int{109, 110}}
	p.observeLasers(active, removed)
	if len(p.laserOffOrigins(131)) != 0 {
		t.Fatal("one removal frame is insufficient")
	}
	confirmed := quake.Snapshot{Map: "base3", Frame: 132, Self: self}
	p.observeLasers(removed, confirmed)
	if len(p.laserOffOrigins(132)) != 2 {
		t.Fatal("stable removal was not recognized")
	}
	from, to := quake.Vec3{-730, -455, -279.875}, quake.Vec3{-730, -410, -279.875}
	if !geometry.LaserMoveHazard(from, to) || geometry.LaserMoveHazardExcept(from, to, p.laserOffOrigins(132)) {
		t.Fatal("off evidence did not release static laser guard")
	}
	active.Frame = 133
	p.observeLasers(confirmed, active)
	if len(p.laserOffOrigins(133)) != 0 {
		t.Fatal("reappearing beam did not revoke off evidence")
	}
	removed.Frame, confirmed.Frame = 134, 135
	p.observeLasers(active, removed)
	p.observeLasers(removed, confirmed)
	gap := quake.Snapshot{Map: "base3", Frame: 137, Self: self}
	p.observeLasers(confirmed, gap)
	if len(p.laserOffOrigins(137)) != 0 {
		t.Fatal("frame gap must revoke off evidence immediately")
	}
	if len(p.laserOffOrigins(142)) != 0 {
		t.Fatal("stale off evidence survived")
	}
}

func TestLaserDisappearanceWithoutRemovalStaysHazardous(t *testing.T) {
	root := "../../workspace/runtime/q2go/baseq2"
	if _, err := os.Stat(root + "/maps/base3.bsp"); err != nil {
		t.Skip("local base3 assets unavailable")
	}
	geometry, err := quake.LoadMap(root, "base3")
	if err != nil {
		t.Fatal(err)
	}
	origin := geometry.StaticLethalLaserOrigins()[0]
	base := quake.Snapshot{Map: "base3", Frame: 130, Self: quake.Vec3{-730, -455, -280},
		Beams: []quake.BeamObservation{{ID: 109, Origin: origin}}}
	for _, tc := range []struct {
		name   string
		change func(*quake.Snapshot)
	}{
		{"no_remove", func(*quake.Snapshot) {}},
		{"moving_view", func(s *quake.Snapshot) { s.RemovedEntities = []int{109}; s.Self[0] += 20 }},
		{"suppressed", func(s *quake.Snapshot) { s.RemovedEntities = []int{109}; s.Suppressed = 1 }},
		{"mover_changed", func(s *quake.Snapshot) {
			s.RemovedEntities = []int{109}
			s.Movers = []quake.Mover{{ID: 7, Model: 2, Origin: s.Self}}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := &Planner{World: World{Map: "base3", Geometry: &geometry}}
			p.observeLasers(quake.Snapshot{}, base)
			missing := quake.Snapshot{Map: "base3", Frame: 131, DeltaFrame: 130, Self: base.Self}
			tc.change(&missing)
			p.observeLasers(base, missing)
			stillMissing := quake.Snapshot{Map: "base3", Frame: 132, Self: missing.Self}
			p.observeLasers(missing, stillMissing)
			if len(p.laserOffOrigins(132)) != 0 {
				t.Fatal("ambiguous disappearance released guard")
			}
		})
	}
}

func TestFullFrameOmissionCanConfirmLaserOff(t *testing.T) {
	root := "../../workspace/runtime/q2go/baseq2"
	if _, err := os.Stat(root + "/maps/base3.bsp"); err != nil {
		t.Skip("local base3 assets unavailable")
	}
	geometry, err := quake.LoadMap(root, "base3")
	if err != nil {
		t.Fatal(err)
	}
	origin := geometry.StaticLethalLaserOrigins()[0]
	p := &Planner{World: World{Map: "base3", Geometry: &geometry}}
	base := quake.Snapshot{Map: "base3", Frame: 130, Self: quake.Vec3{-730, -455, -280}, Beams: []quake.BeamObservation{{ID: 109, Origin: origin}}}
	missing := quake.Snapshot{Map: "base3", Frame: 131, DeltaFrame: -1, Self: base.Self} // Complete entity list in protocol 34.
	confirmed := quake.Snapshot{Map: "base3", Frame: 132, DeltaFrame: -1, Self: base.Self}
	p.observeLasers(quake.Snapshot{}, base)
	p.observeLasers(base, missing)
	if len(p.laserOffOrigins(131)) != 0 {
		t.Fatal("one full-frame omission is insufficient")
	}
	p.observeLasers(missing, confirmed)
	if len(p.laserOffOrigins(132)) != 1 {
		t.Fatal("two stable complete observations should confirm omission")
	}
}
