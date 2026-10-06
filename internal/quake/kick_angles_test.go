package quake

import (
	"encoding/binary"
	"testing"
)

func TestKickAnglesSignedQuarterDegreesAndDeltaInheritance(t *testing.T) {
	d := NewDecoder()
	payload := binary.LittleEndian.AppendUint16(nil, 512)
	payload = append(payload, 202, 6, 255) // -13.5, 1.5, -0.25 degrees
	payload = binary.LittleEndian.AppendUint32(payload, 0)
	f, err := d.playerstate(&reader{data: payload}, Frame{})
	if err != nil || !f.KickAnglesKnown || f.KickAngles != (Vec3{-13.5, 1.5, -.25}) {
		t.Fatalf("kick decode: %+v %v", f, err)
	}
	delta := make([]byte, 6)
	next, err := d.playerstate(&reader{data: delta}, f)
	if err != nil || next.KickAngles != f.KickAngles || !next.KickAnglesKnown {
		t.Fatalf("delta lost kick: %+v %v", next, err)
	}
	s := d.Snapshot(next)
	if s.KickAngles != f.KickAngles || !s.KickAnglesKnown {
		t.Fatal("snapshot lost observed kick")
	}
}
