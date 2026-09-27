package quake

import "testing"

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
