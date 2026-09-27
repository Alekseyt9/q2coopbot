package quake

import (
	"encoding/binary"
	"testing"
)

func velocityFramePacket(number, delta int32, velocity *[3]int16) []byte {
	b := []byte{20}
	b = binary.LittleEndian.AppendUint32(b, uint32(number))
	b = binary.LittleEndian.AppendUint32(b, uint32(delta))
	b = append(b, 0, 0, 17)
	flags := uint16(8 | 16)
	if velocity != nil {
		flags |= 4
	}
	b = binary.LittleEndian.AppendUint16(b, flags)
	if velocity != nil {
		for _, v := range velocity {
			b = binary.LittleEndian.AppendUint16(b, uint16(v))
		}
	}
	b = append(b, 7, 4) // PM_TIME followed by PM_FLAGS: verify stream alignment.
	b = binary.LittleEndian.AppendUint32(b, 2)
	b = binary.LittleEndian.AppendUint16(b, 83)
	return append(b, 18, 0, 0)
}

func TestVelocityFullDeltaAndReset(t *testing.T) {
	d := NewDecoder()
	parse := func(number, delta int32, wire *[3]int16, want Vec3) {
		t.Helper()
		frames, err := d.Parse(velocityFramePacket(number, delta, wire))
		if err != nil || len(frames) != 1 {
			t.Fatalf("decode: %v %+v", err, frames)
		}
		f := frames[0]
		s := d.Snapshot(f)
		if f.Velocity != want || s.SelfVelocity != want || !s.OnGround || s.Health != 83 {
			t.Fatalf("bad playerstate: %+v", s)
		}
	}
	parse(1, -1, &[3]int16{2400, -641, 320}, Vec3{300, -80.125, 40})
	parse(2, 1, nil, Vec3{300, -80.125, 40})
	parse(3, 2, &[3]int16{-32768, 32767, 0}, Vec3{-4096, 4095.875, 0})
	parse(4, 3, &[3]int16{}, Vec3{})
	parse(5, 1, nil, Vec3{300, -80.125, 40}) // requested baseline, not latest frame
	parse(6, -1, nil, Vec3{})                // independent full frame must not retain old velocity
}

func TestVelocityTruncationDoesNotPublishFrame(t *testing.T) {
	packet := velocityFramePacket(1, -1, &[3]int16{2400, -640, 0})
	for n := 14; n < 20; n++ {
		d := NewDecoder()
		if _, err := d.Parse(packet[:n]); err == nil {
			t.Fatalf("accepted truncated velocity at%d", n)
		}
		if len(d.Frames) != 0 {
			t.Fatal("published incomplete playerstate")
		}
	}
	d := NewDecoder()
	if _, err := d.Parse(velocityFramePacket(2, 1, nil)); err == nil {
		t.Fatal("accepted missing velocity baseline")
	}
}
