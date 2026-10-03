package quake

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestOutlineFloorAndInvalidEdge(t *testing.T) {
	data := make([]byte, 160)
	copy(data, "IBSP")
	binary.LittleEndian.PutUint32(data[4:], 38)
	add := func(i int, b []byte) {
		binary.LittleEndian.PutUint32(data[8+i*8:], uint32(len(data)))
		binary.LittleEndian.PutUint32(data[12+i*8:], uint32(len(b)))
		data = append(data, b...)
	}
	plane := make([]byte, 20)
	binary.LittleEndian.PutUint32(plane[8:], math.Float32bits(1))
	add(1, plane)
	vertices := make([]byte, 36)
	binary.LittleEndian.PutUint32(vertices[12:], math.Float32bits(64))
	binary.LittleEndian.PutUint32(vertices[28:], math.Float32bits(64))
	add(2, vertices)
	face := make([]byte, 20)
	binary.LittleEndian.PutUint16(face[8:], 3)
	add(6, face)
	edges := make([]byte, 12)
	for i := 0; i < 3; i++ {
		binary.LittleEndian.PutUint16(edges[i*4:], uint16(i))
		binary.LittleEndian.PutUint16(edges[i*4+2:], uint16((i+1)%3))
	}
	add(11, edges)
	surf := make([]byte, 12)
	binary.LittleEndian.PutUint32(surf[4:], 1)
	binary.LittleEndian.PutUint32(surf[8:], 2)
	add(12, surf)
	outline, e := parseOutline(data, "test", "fixture")
	if e != nil || len(outline.Floors) != 1 || outline.Floors[0][1][0] != 64 {
		t.Fatalf("outline=%+v error=%v", outline, e)
	}
	off := int(binary.LittleEndian.Uint32(data[8+12*8:]))
	binary.LittleEndian.PutUint32(data[off:], 999)
	if _, e = parseOutline(data, "test", "fixture"); e == nil {
		t.Fatal("invalid edge accepted")
	}
	if _, e = parseOutline([]byte("IBSP"), "test", "fixture"); e == nil {
		t.Fatal("short header accepted")
	}
}
