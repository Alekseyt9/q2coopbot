package quake

import (
	"encoding/binary"
	"testing"
)

func pmStatePacket(number, delta int32, state *[3]byte) []byte {
	b := []byte{20}
	b = binary.LittleEndian.AppendUint32(b, uint32(number))
	b = binary.LittleEndian.AppendUint32(b, uint32(delta))
	b = append(b, 0, 0, 17)
	flags := uint16(0)
	if state != nil {
		flags = 1 | 8 | 16
	}
	b = binary.LittleEndian.AppendUint16(b, flags)
	if state != nil {
		b = append(b, state[:]...)
	}
	b = binary.LittleEndian.AppendUint32(b, 2)
	b = binary.LittleEndian.AppendUint16(b, 83)
	return append(b, 18, 0, 0)
}

func TestPMStateDeltaAndIndependentReset(t *testing.T) {
	d := NewDecoder()
	for _, tc := range []struct {
		number, delta int32
		wire          *[3]byte
		want          [3]byte
	}{
		{1, -1, &[3]byte{2, 7, 20}, [3]byte{2, 7, 20}},
		{2, 1, nil, [3]byte{2, 7, 20}},
		{3, 2, &[3]byte{0, 0, 4}, [3]byte{0, 0, 4}},
		{4, 1, nil, [3]byte{2, 7, 20}},
		{5, -1, nil, [3]byte{}},
	} {
		frames, err := d.Parse(pmStatePacket(tc.number, tc.delta, tc.wire))
		if err != nil || len(frames) != 1 {
			t.Fatal(err)
		}
		s := d.Snapshot(frames[0])
		if !s.PMStateKnown || [3]byte{s.PMType, s.PMTime, s.PMFlags} != tc.want || s.Health != 83 {
			t.Fatalf("bad PM state %+v", s)
		}
	}
}

func TestPMStateTruncationDoesNotPublish(t *testing.T) {
	packet := pmStatePacket(1, -1, &[3]byte{0, 7, 20})
	for n := 14; n < 17; n++ {
		d := NewDecoder()
		if _, err := d.Parse(packet[:n]); err == nil || len(d.Frames) != 0 {
			t.Fatal("truncated state published", n)
		}
	}
}
