package quake

import (
	"encoding/binary"
	"fmt"
	"math"
)

type aasAreaPlane struct {
	normal   Vec3
	distance float64
}
type aasAreaNode struct {
	plane    int
	children [2]int
}
type aasAreaTree struct {
	planes []aasAreaPlane
	nodes  []aasAreaNode
}

// Area bounds can overlap even when convex areas do not. AAS nodes locate the
// actual volume: node 1 is the root, negative children are area IDs, 0 is solid.
func parseAASAreaTree(planes, nodes []byte, areas int) (*aasAreaTree, error) {
	if len(planes) == 0 && len(nodes) == 0 {
		return nil, nil
	}
	if len(planes) == 0 || len(planes)%20 != 0 || len(nodes) < 24 || len(nodes)%12 != 0 {
		return nil, fmt.Errorf("invalid AAS area tree dimensions")
	}
	t := &aasAreaTree{planes: make([]aasAreaPlane, len(planes)/20), nodes: make([]aasAreaNode, len(nodes)/12)}
	for i := range t.planes {
		p := aasAreaPlane{normal: vec(planes, i*20), distance: float64(math.Float32frombits(binary.LittleEndian.Uint32(planes[i*20+12:])))}
		for _, v := range []float64{p.normal[0], p.normal[1], p.normal[2], p.distance} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, fmt.Errorf("non-finite AAS plane %d", i)
			}
		}
		t.planes[i] = p
	}
	for i := 1; i < len(t.nodes); i++ {
		at := i * 12
		n := aasAreaNode{plane: int(int32(binary.LittleEndian.Uint32(nodes[at:]))), children: [2]int{int(int32(binary.LittleEndian.Uint32(nodes[at+4:]))), int(int32(binary.LittleEndian.Uint32(nodes[at+8:])))}}
		if n.plane < 0 || n.plane >= len(t.planes) {
			return nil, fmt.Errorf("invalid AAS node plane at %d", i)
		}
		for _, child := range n.children {
			if child >= len(t.nodes) || child <= -areas {
				return nil, fmt.Errorf("invalid AAS node child at %d", i)
			}
		}
		t.nodes[i] = n
	}
	// Check cycles without recursion, including unused subtrees.
	colors := make([]byte, len(t.nodes))
	type visit struct {
		node int
		exit bool
	}
	for root := 1; root < len(t.nodes); root++ {
		if colors[root] != 0 {
			continue
		}
		stack := []visit{{node: root}}
		for len(stack) > 0 {
			v := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if v.exit {
				colors[v.node] = 2
				continue
			}
			if colors[v.node] == 2 {
				continue
			}
			if colors[v.node] == 1 {
				return nil, fmt.Errorf("cycle in AAS area tree at %d", v.node)
			}
			colors[v.node] = 1
			stack = append(stack, visit{node: v.node, exit: true})
			for _, child := range t.nodes[v.node].children {
				if child > 0 {
					stack = append(stack, visit{node: child})
				}
			}
		}
	}
	return t, nil
}

func (t *aasAreaTree) areaFor(p Vec3) int {
	for _, v := range p {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return -1
		}
	}
	for node := 1; ; {
		n := t.nodes[node]
		plane := t.planes[n.plane]
		side := 0
		if p[0]*plane.normal[0]+p[1]*plane.normal[1]+p[2]*plane.normal[2]-plane.distance <= 0 {
			side = 1
		}
		node = n.children[side]
		if node == 0 {
			return -1
		}
		if node < 0 {
			return -node
		}
	}
}
