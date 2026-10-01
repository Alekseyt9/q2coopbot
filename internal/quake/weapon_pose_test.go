package quake

import (
	"encoding/binary"
	"testing"
)

func weaponPosePacket(number, delta int32, pose bool) []byte {
	b := []byte{20}
	b = binary.LittleEndian.AppendUint32(b, uint32(number))
	b = binary.LittleEndian.AppendUint32(b, uint32(delta))
	b = append(b, 0, 0, 17)
	flags := uint16(0)
	if pose {
		flags = 128 | 256 | 512 | 4096 | 8192
	}
	b = binary.LittleEndian.AppendUint16(b, flags)
	if pose {
		b = append(b, 1, 2, 3)
		for _, v := range []int16{120, -16384, 30} {
			b = binary.LittleEndian.AppendUint16(b, uint16(v))
		}
		b = append(b, 4, 5, 6, 9, 11, 7, 8, 9, 10, 11, 12)
	}
	b = binary.LittleEndian.AppendUint32(b, 2)
	b = binary.LittleEndian.AppendUint16(b, 87)
	return append(b, 18, 0, 0)
}

func TestWeaponPoseFullDeltaAndReset(t *testing.T) {
	d := NewDecoder()
	for _, tc := range []struct {
		n, base  int32
		pose     bool
		gunFrame int
	}{{1, -1, true, 11}, {2, 1, false, 11}, {3, -1, false, 0}, {4, 1, false, 11}} {
		frames, err := d.Parse(weaponPosePacket(tc.n, tc.base, tc.pose))
		if err != nil || len(frames) != 1 {
			t.Fatalf("decode: %v", err)
		}
		s := d.Snapshot(frames[0])
		if s.GunFrame != tc.gunFrame || s.Health != 87 {
			t.Fatalf("misaligned pose: %+v", s)
		}
		if tc.gunFrame == 11 && (frames[0].Gun != 9 || s.ViewAngles != ([3]int16{120, -16384, 30})) {
			t.Fatal("pose not inherited")
		}
	}
	packet := weaponPosePacket(1, -1, true)
	for n := 14; n < len(packet); n++ {
		bad := NewDecoder()
		if _, err := bad.Parse(packet[:n]); err == nil || len(bad.Frames) != 0 {
			t.Fatalf("accepted truncated pose at%d", n)
		}
	}
}
