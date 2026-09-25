package quake

import (
	"encoding/binary"
	"testing"
)

func TestPHSPossibleSourcesInvertsSourceToListenerMask(t *testing.T) {
	vis := make([]byte, 22)
	binary.LittleEndian.PutUint32(vis, 2)
	binary.LittleEndian.PutUint32(vis[8:], 20)  // source cluster 0 PHS
	binary.LittleEndian.PutUint32(vis[16:], 21) // source cluster 1 PHS
	vis[20], vis[21] = 0b11, 0b10
	nodes := make([]byte, 28)
	binary.LittleEndian.PutUint32(nodes[4:], ^uint32(0))
	binary.LittleEndian.PutUint32(nodes[8:], ^uint32(1))
	leaves := make([]byte, 56)
	binary.LittleEndian.PutUint16(leaves[28+4:], 1)
	m := &MapInfo{visibility: parseBSPVisibility(vis, nodes, leaves,
		[]bspPlane{{normal: Vec3{1, 0, 0}}})}
	if m.visibility == nil {
		t.Fatal("valid visibility rejected")
	}
	if got, total, ok := m.PHSPossibleSources(Vec3{10, 0, 0}); !ok || got != 1 || total != 2 {
		t.Fatalf("receiver cluster 0: possible=%d total=%d ok=%t", got, total, ok)
	}
	if got, total, ok := m.PHSPossibleSources(Vec3{-10, 0, 0}); !ok || got != 2 || total != 2 {
		t.Fatalf("receiver cluster 1: possible=%d total=%d ok=%t", got, total, ok)
	}
	vis[21] = 0 // truncated zero run must not be treated as a reliable filter
	m.visibility = parseBSPVisibility(vis, nodes, leaves, []bspPlane{{normal: Vec3{1, 0, 0}}})
	if _, _, ok := m.PHSPossibleSources(Vec3{10, 0, 0}); ok {
		t.Fatal("malformed VIS row accepted")
	}
}
