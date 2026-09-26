package quake

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func areaTreeBytes() ([]byte, []byte) {
	planes, nodes := make([]byte, 20), make([]byte, 24)
	binary.LittleEndian.PutUint32(planes, math.Float32bits(1))
	binary.LittleEndian.PutUint32(planes[4:], math.Float32bits(1))
	binary.LittleEndian.PutUint32(nodes[16:], ^uint32(0)) // front: area 1
	binary.LittleEndian.PutUint32(nodes[20:], ^uint32(1)) // back: area 2
	return planes, nodes
}

func TestAreaTreeDisambiguatesOverlappingBounds(t *testing.T) {
	planes, nodes := areaTreeBytes()
	tree, err := parseAASAreaTree(planes, nodes, 3)
	if err != nil {
		t.Fatal(err)
	}
	n := &Navigator{areaTree: tree, Areas: []Area{{},
		{Min: Vec3{-20, -20, -20}, Max: Vec3{20, 20, 20}, Center: Vec3{10, 10, 0}},
		{Min: Vec3{-20, -20, -20}, Max: Vec3{20, 20, 20}, Center: Vec3{1, 1, 0}}},
		Edges: [][]Edge{nil, nil, {{To: 1, Kind: 2, Cost: 1, Start: Vec3{-1, -1, 0}, End: Vec3{1, 1, 0}}}}}
	if got := n.AreaFor(Vec3{1, 1, 0}); got != 1 {
		t.Fatalf("chose overlapping bounding box: %d", got)
	}
	if got := n.AreaFor(Vec3{}); got != 2 {
		t.Fatalf("plane boundary must select back child: %d", got)
	}
	if route, ok := n.Route(Vec3{-1, -1, 0}, Vec3{1, 1, 0}); !ok || len(route) != 2 {
		t.Fatalf("wrong route through convex areas: %v %v", route, ok)
	}
	tree.nodes[1].children[1] = 0
	if got := n.AreaFor(Vec3{-1, -1, 0}); got != -1 {
		t.Fatalf("solid point used nearby bounding box: %d", got)
	}
}

func TestAreaTreeRejectsMalformedData(t *testing.T) {
	for _, change := range []func([]byte, []byte){
		func(p, n []byte) { binary.LittleEndian.PutUint32(n[12:], 1) },          // plane out of range
		func(p, n []byte) { binary.LittleEndian.PutUint32(n[16:], 2) },          // node out of range
		func(p, n []byte) { binary.LittleEndian.PutUint32(n[16:], ^uint32(2)) }, // area out of range
		func(p, n []byte) { binary.LittleEndian.PutUint32(n[16:], 1) },          // root cycle
		func(p, n []byte) { binary.LittleEndian.PutUint32(p, math.Float32bits(float32(math.NaN()))) },
	} {
		p, n := areaTreeBytes()
		change(p, n)
		if _, err := parseAASAreaTree(p, n, 3); err == nil {
			t.Fatal("accepted malformed tree")
		}
	}
	p, _ := areaTreeBytes()
	if _, err := parseAASAreaTree(p, nil, 3); err == nil {
		t.Fatal("accepted planes without nodes")
	}
}

func TestLoadAreaTreeVersions(t *testing.T) {
	for _, version := range []uint32{2, 3, 4, 5} {
		header := 120
		if version >= 4 {
			header = 124
		}
		planes, nodes := areaTreeBytes()
		data := make([]byte, header)
		copy(data, "EAAS")
		binary.LittleEndian.PutUint32(data[4:], version)
		for _, l := range []struct {
			index int
			data  []byte
		}{{2, planes}, {7, make([]byte, 3*48)}, {8, make([]byte, 3*28)}, {9, nil}, {10, nodes}} {
			at := header - 112 + l.index*8
			binary.LittleEndian.PutUint32(data[at:], uint32(len(data)))
			binary.LittleEndian.PutUint32(data[at+4:], uint32(len(l.data)))
			data = append(data, l.data...)
		}
		if version == 5 {
			for i := 8; i < header; i++ {
				data[i] ^= byte((i - 8) * 119)
			}
		}
		path := filepath.Join(t.TempDir(), "tree.aas")
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
		n, err := LoadAAS(path)
		if err != nil {
			t.Fatalf("version %d: %v", version, err)
		}
		if n.AreaFor(Vec3{1, 1, 0}) != 1 || n.AreaFor(Vec3{-1, -1, 0}) != 2 {
			t.Fatalf("version %d lost tree", version)
		}
	}
}
