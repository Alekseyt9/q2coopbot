package quake

import "encoding/binary"

type bspVisNode struct {
	plane    int
	children [2]int
}

type bspVisibility struct {
	data       []byte
	phsOffsets []int
	nodes      []bspVisNode
	leaves     []int
	planes     []bspPlane
}

// parseBSPVisibility returns nil for missing or malformed VIS. Collision and
// normal navigation remain available when acoustic diagnostics cannot be used.
func parseBSPVisibility(raw, nodes, leaves []byte, planes []bspPlane) *bspVisibility {
	if len(raw) < 4 || len(nodes) == 0 || len(leaves) == 0 {
		return nil
	}
	count := int(int32(binary.LittleEndian.Uint32(raw)))
	if count <= 0 || count > (len(raw)-4)/8 || count > len(leaves)/28 {
		return nil
	}
	v := &bspVisibility{data: append([]byte(nil), raw...), phsOffsets: make([]int, count),
		nodes: make([]bspVisNode, len(nodes)/28), leaves: make([]int, len(leaves)/28), planes: planes}
	for i := range v.phsOffsets {
		offset := int(int32(binary.LittleEndian.Uint32(raw[4+i*8+4:])))
		if offset < 4+count*8 || offset >= len(raw) {
			return nil
		}
		v.phsOffsets[i] = offset
	}
	for i := range v.nodes {
		at := i * 28
		v.nodes[i] = bspVisNode{plane: int(int32(binary.LittleEndian.Uint32(nodes[at:]))),
			children: [2]int{int(int32(binary.LittleEndian.Uint32(nodes[at+4:]))),
				int(int32(binary.LittleEndian.Uint32(nodes[at+8:])))}}
	}
	for i := range v.leaves {
		v.leaves[i] = int(int16(binary.LittleEndian.Uint16(leaves[i*28+4:])))
	}
	return v
}

func (v *bspVisibility) pointCluster(point Vec3) int {
	if v == nil || len(v.nodes) == 0 {
		return -1
	}
	node := 0
	for steps := 0; steps <= len(v.nodes); steps++ {
		if node < 0 {
			leaf := -1 - node
			if leaf >= 0 && leaf < len(v.leaves) && v.leaves[leaf] >= 0 && v.leaves[leaf] < len(v.phsOffsets) {
				return v.leaves[leaf]
			}
			return -1
		}
		if node >= len(v.nodes) || v.nodes[node].plane < 0 || v.nodes[node].plane >= len(v.planes) {
			return -1
		}
		plane := v.planes[v.nodes[node].plane]
		d := point[0]*plane.normal[0] + point[1]*plane.normal[1] + point[2]*plane.normal[2] - plane.dist
		child := 0
		if d < 0 {
			child = 1
		}
		node = v.nodes[node].children[child]
	}
	return -1
}

func (v *bspVisibility) phsContains(source, listener int) (bool, bool) {
	if source < 0 || source >= len(v.phsOffsets) || listener < 0 || listener >= len(v.phsOffsets) {
		return false, false
	}
	rowBytes := (len(v.phsOffsets) + 7) / 8
	read, at := 0, v.phsOffsets[source]
	matched := false
	for read < rowBytes {
		if at >= len(v.data) {
			return false, false
		}
		b := v.data[at]
		at++
		if b == 0 {
			if at >= len(v.data) || v.data[at] == 0 || read+int(v.data[at]) > rowBytes {
				return false, false
			}
			read += int(v.data[at])
			at++
			continue
		}
		if read == listener/8 && b&(1<<uint(listener%8)) != 0 {
			matched = true
		}
		read++
	}
	return matched, true
}

// PHSPossibleSources counts source clusters that could reach this listener
// through the BSP PHS, ignoring dynamic area portals. This is conditional:
// server sounds can bypass PHS without indicating that in the UDP packet.
func (m *MapInfo) PHSPossibleSources(listener Vec3) (possible, total int, ok bool) {
	if m == nil || m.visibility == nil {
		return 0, 0, false
	}
	v := m.visibility
	first := v.pointCluster(listener)
	elevated := listener
	elevated[2] += 32 // server can also check this point when the client is in water
	second := v.pointCluster(elevated)
	if first < 0 && second < 0 {
		return 0, len(v.phsOffsets), false
	}
	for source := range v.phsOffsets {
		one, valid := v.phsContains(source, first)
		if first < 0 {
			valid = true
		}
		if !valid {
			return 0, len(v.phsOffsets), false
		}
		two := false
		if second >= 0 && second != first {
			two, valid = v.phsContains(source, second)
			if !valid {
				return 0, len(v.phsOffsets), false
			}
		}
		if one || two {
			possible++
		}
	}
	return possible, len(v.phsOffsets), true
}
