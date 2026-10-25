package quake

import (
	"encoding/binary"
	"testing"
)

func TestStaticAreaReadsLeafAreaSeparatelyFromCluster(t *testing.T) {
	vis := make([]byte, 22)
	binary.LittleEndian.PutUint32(vis, 2)
	binary.LittleEndian.PutUint32(vis[8:], 20)
	binary.LittleEndian.PutUint32(vis[16:], 21)
	vis[20] = 3
	vis[21] = 3
	nodes, leaves := make([]byte, 28), make([]byte, 56)
	binary.LittleEndian.PutUint32(nodes[4:], ^uint32(0))
	binary.LittleEndian.PutUint32(nodes[8:], ^uint32(1))
	binary.LittleEndian.PutUint16(leaves[32:], 1)
	binary.LittleEndian.PutUint16(leaves[6:], 7)
	binary.LittleEndian.PutUint16(leaves[34:], 9)
	m := MapInfo{visibility: parseBSPVisibility(vis, nodes, leaves, []bspPlane{{normal: Vec3{1, 0, 0}}})}
	for _, tc := range []struct {
		p    Vec3
		area int
	}{{Vec3{10, 0, 0}, 7}, {Vec3{-10, 0, 0}, 9}} {
		area, ok := m.StaticArea(tc.p)
		if !ok || area != tc.area {
			t.Fatal("area confused with cluster", area, ok)
		}
	}
	if _, ok := (&MapInfo{}).StaticArea(Vec3{}); ok {
		t.Fatal("missing area treated as known")
	}
}
