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
