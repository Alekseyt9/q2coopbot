package main

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"testing"
)

func TestMovePacketMatchesProtocol34Reference(t *testing.T) {
	got := movePacket(UserCmd{Yaw: 8192, Forward: 400, Buttons: 1, Msec: 50}, UserCmd{}, 7)
	want, _ := hex.DecodeString("02a4ffffffff0000000000004a00209001013200")
	if !bytes.Equal(got, want) {
		t.Fatalf("move packet = %x, want %x", got, want)
	}
}
func TestDecodeFullFrameAndTeammate(t *testing.T) {
	d := NewDecoder()
	buf := []byte{12}
	buf = binary.LittleEndian.AppendUint32(buf, 34)
	buf = binary.LittleEndian.AppendUint32(buf, 7)
	buf = append(buf, 0)
	buf = append(buf, []byte("baseq2\x00")...)
	buf = binary.LittleEndian.AppendUint16(buf, 0)
	buf = append(buf, []byte("Outer Base\x00")...)
	config := func(index uint16, value string) {
		buf = append(buf, 13)
		buf = binary.LittleEndian.AppendUint16(buf, index)
		buf = append(buf, []byte(value)...)
		buf = append(buf, 0)
	}
	config(33, "maps/base1.bsp")
	config(30, "4")
	config(34, "models/monsters/soldier/tris.md2")
	if _, e := d.Parse(buf); e != nil {
		t.Fatal(e)
	}
	buf = []byte{20}
	buf = binary.LittleEndian.AppendUint32(buf, 10)
	buf = binary.LittleEndian.AppendUint32(buf, ^uint32(0))
	buf = append(buf, 0, 0, 17)
	buf = binary.LittleEndian.AppendUint16(buf, 0x1042)
	for _, v := range []int16{800, -160, 192, 0, 8192, 0} {
		buf = binary.LittleEndian.AppendUint16(buf, uint16(v))
	}
	buf = append(buf, 4)
	buf = binary.LittleEndian.AppendUint32(buf, 2)
	buf = binary.LittleEndian.AppendUint16(buf, 83)
	buf = append(buf, 18)
	entity := func(id, model byte, x, y, z int16) {
		buf = append(buf, 0x83, 0x0a, id, model)
		for _, v := range []int16{x, y, z} {
			buf = binary.LittleEndian.AppendUint16(buf, uint16(v))
		}
	}
	entity(2, 255, 256, -128, 192)
	entity(5, 2, 720, 0, 192)
	buf = append(buf, 0, 0)
	frames, e := d.Parse(buf)
	if e != nil {
		t.Fatal(e)
	}
	if len(frames) != 1 {
		t.Fatalf("frames=%d", len(frames))
	}
	s := d.Snapshot(frames[0])
	if s.Map != "base1" || s.Self != (Vec3{100, -20, 24}) || s.Teammate == nil || *s.Teammate != (Vec3{32, -16, 24}) || s.Health != 83 || len(s.Enemies) != 1 {
		t.Fatalf("decoded snapshot: %+v", s)
	}
}
func TestRouteUsesReachabilities(t *testing.T) {
	n := &Navigator{Areas: []Area{{}, {Min: Vec3{-10, -10, -10}, Max: Vec3{10, 10, 10}}, {Min: Vec3{90, -10, -10}, Max: Vec3{110, 10, 10}}}, Edges: [][]Edge{{}, {{To: 2, Start: Vec3{8, 0, 0}, End: Vec3{92, 0, 0}, Kind: 2, Cost: 10}}, nil}}
	route, ok := n.Route(Vec3{0, 0, 0}, Vec3{100, 0, 0})
	if !ok || len(route) != 2 || route[1].Position != (Vec3{92, 0, 0}) {
		t.Fatalf("route=%+v ok=%t", route, ok)
	}
	if _, ok = n.Route(Vec3{100, 0, 0}, Vec3{0, 0, 0}); ok {
		t.Fatal("invented route through missing reverse edge")
	}
}
func TestBSPBrushBlocksLineOfFire(t *testing.T) {
	box := &CollisionMap{
		planes: []bspPlane{{Vec3{1, 0, 0}, 10}, {Vec3{-1, 0, 0}, 0}, {Vec3{0, 1, 0}, 10}, {Vec3{0, -1, 0}, 0}, {Vec3{0, 0, 1}, 10}, {Vec3{0, 0, -1}, 0}},
		sides:  []uint16{0, 1, 2, 3, 4, 5}, brushes: []bspBrush{{0, 6, 1}}, worldBrushes: []int{0},
	}
	if box.ClearShot(Vec3{-5, 5, 5}, Vec3{15, 5, 5}) {
		t.Fatal("solid brush did not block shot")
	}
	if !box.ClearShot(Vec3{-5, 20, 5}, Vec3{15, 20, 5}) {
		t.Fatal("shot outside brush was blocked")
	}
}
