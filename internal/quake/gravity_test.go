package quake

import (
	"encoding/binary"
	"testing"
)

func gravityPacket(number, delta int32, gravity *int16) []byte {
	b := []byte{20}
	b = binary.LittleEndian.AppendUint32(b, uint32(number))
	b = binary.LittleEndian.AppendUint32(b, uint32(delta))
	b = append(b, 0, 0, 17)
	flags := uint16(16)
	if gravity != nil {
		flags |= 32
	}
	b = binary.LittleEndian.AppendUint16(b, flags)
	b = append(b, 4)
	if gravity != nil {
		b = binary.LittleEndian.AppendUint16(b, uint16(*gravity))
	}
	b = binary.LittleEndian.AppendUint32(b, 2)
	b = binary.LittleEndian.AppendUint16(b, 83)
	return append(b, 18, 0, 0)
}

func TestGravityDeltaAndReset(t *testing.T) {
	d := NewDecoder()
	g := int16(800)
	for _, tc := range []struct {
		number, delta int32
		gravity       *int16
		want          int16
	}{{1, -1, &g, 800}, {2, 1, nil, 800}, {3, -1, nil, 0}, {4, 1, nil, 800}} {
		frames, err := d.Parse(gravityPacket(tc.number, tc.delta, tc.gravity))
		if err != nil || len(frames) != 1 {
			t.Fatalf("decode: %v", err)
		}
		s := d.Snapshot(frames[0])
		if s.Gravity != tc.want || s.Health != 83 || !s.OnGround {
			t.Fatalf("misaligned state: %+v", s)
		}
	}
	packet := gravityPacket(1, -1, &g)
	bad := NewDecoder()
	if _, err := bad.Parse(packet[:16]); err == nil || len(bad.Frames) != 0 {
		t.Fatal("truncated gravity published")
	}
}
