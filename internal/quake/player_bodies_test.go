package quake

import "testing"

func TestProtocolSolidBoundsAndBodySweep(t *testing.T) {
	body := Object{ID: 8, Origin: Vec3{40, 0, 24}, Solid: 2 | (3 << 5) | (7 << 10)}
	lo, hi, ok := body.SolidBounds()
	if !ok || lo != (Vec3{-16, -16, -24}) || hi != (Vec3{16, 16, 24}) {
		t.Fatal("wrong protocol hull", lo, hi, ok)
	}
	m := playerTraceBoxes([2]Vec3{{-200, -200, -20}, {200, 200, 0}})
	from, to := Vec3{0, 0, 24.125}, Vec3{40, 0, 24.125}
	tr := m.TracePlayerWithEntities(from, to, false, nil, []Object{body})
	if !tr.Valid || tr.Fraction >= 1 || tr.End[0] >= 8 || tr.Normal != (Vec3{-1, 0, 0}) {
		t.Fatal("body ignored", tr)
	}
	body.Solid = 0
	if tr = m.TracePlayerWithEntities(from, to, false, nil, []Object{body}); tr.Fraction != 1 {
		t.Fatal("nonsolid blocked movement", tr)
	}
	body.Solid = 31
	if _, _, ok = body.SolidBounds(); ok {
		t.Fatal("BSP marker decoded as a box")
	}
}

func TestSnapshotCollisionBodiesExcludeSelfAndBrushModels(t *testing.T) {
	d := NewDecoder()
	d.PlayerNumber = 1
	solid := uint16(2 | (3 << 5) | (7 << 10))
	s := d.Snapshot(Frame{Entities: map[int]Entity{1: {Number: 1, Solid: solid}, 2: {Number: 2, Solid: solid, Origin: Vec3{100, 0, 0}}, 3: {Number: 3, Solid: 31}, 4: {Number: 4, Solid: 0}}})
	if len(s.CollisionBodies) != 1 || s.CollisionBodies[0].ID != 2 || s.CollisionBodies[0].Solid != solid {
		t.Fatal("incorrect collision set", s.CollisionBodies)
	}
}

func TestSnapshotBodyAppearanceAndDeathSolidComeFromPacket(t *testing.T) {
	const path = "models/monsters/parasite/tris.md2"
	d := Decoder{Config: map[int]string{33: path}}
	for _, solid := range []uint16{0, 7266} {
		entity := Entity{Number: 7, Model: 1, Frame: 38, Solid: solid, Skin: 2, Angles: Vec3{0, 90, 0}}
		s := d.Snapshot(Frame{Number: 100, Entities: map[int]Entity{7: entity}})
		if len(s.Defeated) != 1 {
			t.Fatal("missing observed death animation", s.Defeated)
		}
		body := s.Defeated[0]
		if body.Solid != solid || body.ModelPath != path || body.Frame != 38 || body.Skin != 2 || body.Angles != entity.Angles {
			t.Fatal("lost packet appearance/solid", body)
		}
		if solid != 0 && (len(s.CollisionBodies) != 1 || s.CollisionBodies[0].ModelPath != path || s.CollisionBodies[0].Frame != 38 || s.CollisionBodies[0].Angles != entity.Angles) {
			t.Fatal("lost collision-body appearance", s.CollisionBodies)
		}
	}
}
