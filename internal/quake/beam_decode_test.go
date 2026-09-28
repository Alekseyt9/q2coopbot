package quake

import (
	"encoding/binary"
	"testing"
)

func TestBeamRenderFXAndEndpointDecode(t *testing.T) {
	data := []byte{128}
	for _, v := range []int16{-266 * 8, -400 * 8, -280 * 8, -800 * 8, -400 * 8, -280 * 8} {
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], uint16(v))
		data = append(data, b[:]...)
	}
	flags := uint32(0x1000 | 0x1000000 | 1 | 2 | 0x200)
	entity, err := parseEntity(&reader{data: data}, 174, flags, Entity{Model: 1, Frame: 4})
	if err != nil {
		t.Fatal(err)
	}
	if entity.RenderFX != 128 || entity.Origin != (Vec3{-266, -400, -280}) || entity.OldOrigin != (Vec3{-800, -400, -280}) {
		t.Fatalf("beam fields lost: %+v", entity)
	}
	d := NewDecoder()
	s := d.Snapshot(Frame{Number: 10, Entities: map[int]Entity{174: entity}})
	if len(s.Beams) != 1 || s.Beams[0].ID != 174 || s.Beams[0].End != entity.OldOrigin {
		t.Fatalf("beam omitted from snapshot: %+v", s.Beams)
	}
}

func TestLaserSparksTempEntityDoesNotBreakPacket(t *testing.T) {
	d := NewDecoder()
	packet := append([]byte{3, 15}, make([]byte, 9)...)
	packet = append(packet, 11, 'o', 'k', 0)
	if _, err := d.Parse(packet); err != nil {
		t.Fatal(err)
	}
	if len(d.Commands) != 1 || d.Commands[0] != "ok" {
		t.Fatalf("following command lost: %v", d.Commands)
	}
}
