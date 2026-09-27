package quake

import "testing"

func TestParasiteDeathAndRevivalClassification(t *testing.T) {
	d := Decoder{Config: map[int]string{33: "models/monsters/parasite/tris.md2"}}
	for _, frame := range []int{31, 32, 33, 38, 39, 31} {
		for _, solid := range []uint16{0, 8290} {
			s := d.Snapshot(Frame{Number: 1, Entities: map[int]Entity{360: {Number: 360, Model: 1, Frame: frame, Solid: solid}}})
			dead := frame >= 32 && frame <= 38
			if (len(s.Defeated) == 1) != dead || (len(s.Enemies) == 0) != dead || (len(s.Obstacles) == 1) != (solid != 0) {
				t.Fatalf("frame=%d solid=%d snapshot=%+v", frame, solid, s)
			}
		}
	}
	// The same live report also contained Infantry185: that is attack102,
	// not a corpse. Do not suppress living targets to fix Parasite38.
	d.Config[33] = "models/monsters/infantry/tris.md2"
	if s := d.Snapshot(Frame{Number: 2, Entities: map[int]Entity{343: {Number: 343, Model: 1, Frame: 185, Solid: 8290}}}); len(s.Enemies) != 1 {
		t.Fatal("live infantry attack incorrectly classified as death")
	}
}

func TestEntitySolidDeltaAndDeathObstacle(t *testing.T) {
	e, err := parseEntity(&reader{data: []byte{0x20, 0x08}}, 7, 0x8000000, Entity{Number: 7})
	if err != nil || e.Solid != 2080 {
		t.Fatalf("solid decode: %+v %v", e, err)
	}
	kept, err := parseEntity(&reader{}, 7, 0, e)
	if err != nil || kept.Solid != e.Solid {
		t.Fatal("delta lost solid")
	}
	cleared, err := parseEntity(&reader{data: []byte{0, 0}}, 7, 0x8000000, e)
	if err != nil || cleared.Solid != 0 {
		t.Fatal("solid removal lost")
	}
	if _, err := parseEntity(&reader{data: []byte{1}}, 7, 0x8000000, e); err == nil {
		t.Fatal("truncated solid accepted")
	}
	d := Decoder{Config: map[int]string{33: "models/monsters/soldier/tris.md2"}}
	e.Model = 1
	e.Frame = 280
	f := Frame{Number: 1, Entities: map[int]Entity{7: e}}
	s := d.Snapshot(f)
	if len(s.Enemies) != 0 || len(s.Obstacles) != 1 {
		t.Fatalf("dying monster: %+v", s)
	}
	e.Solid = 0
	f.Entities[7] = e
	s = d.Snapshot(f)
	if len(s.Enemies) != 0 || len(s.Obstacles) != 0 {
		t.Fatal("non-solid corpse remains an obstacle")
	}
}

func TestTankDeathObservationDoesNotRemainCombatTarget(t *testing.T) {
	for _, frame := range []int{221, 222, 253, 254} {
		d := Decoder{Config: map[int]string{33: "models/monsters/tank/tris.md2"}}
		s := d.Snapshot(Frame{Number: 1, Entities: map[int]Entity{7: {Number: 7, Model: 1, Frame: frame, Solid: 2080}}})
		dead := frame >= 222 && frame <= 253
		if (len(s.Defeated) == 1) != dead || (len(s.Enemies) == 0) != dead || len(s.Obstacles) != 1 {
			t.Fatalf("frame%d: %+v", frame, s)
		}
		if dead && s.Defeated[0].ID != 7 {
			t.Fatal("death lost entity identity")
		}
	}
}
