package quake

import (
	"encoding/binary"
	"fmt"
	"math"
)

type lightFace struct {
	vec        [2][4]float64
	mins, size [2]int
	styles     [4]byte
	offset     int
	skip       bool
}
type lightNode struct {
	plane, first, count int
	children            [2]int
}
type bspLight struct {
	planes []bspPlane
	nodes  []lightNode
	faces  []lightFace
	data   []byte
	root   int
}

// StaticLightLevel follows Quake II's downward lightmap sample, at modulate=1.
// It includes server lightstyles, but not transient renderer dynamic lights.
// The bool distinguishes unavailable/malformed data from a genuinely dark sample.
func (m *MapInfo) StaticLightLevel(eye Vec3, config map[int]string, frame int) (byte, bool) {
	if m == nil || m.lighting == nil {
		return 0, false
	}
	l := m.lighting
	if len(l.data) == 0 {
		return 150, true
	} // BSP without lighting is fullbright.
	for _, v := range eye {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, false
		}
	}
	end := eye
	end[2] -= 2048
	budget := len(l.nodes)*2 + 1
	var sample func(int, Vec3, Vec3) (Vec3, bool)
	sample = func(index int, from, to Vec3) (Vec3, bool) {
		if index < 0 || budget <= 0 {
			return Vec3{}, false
		}
		budget--
		n := l.nodes[index]
		p := l.planes[n.plane]
		a, b := -p.dist, -p.dist
		for i := 0; i < 3; i++ {
			a += from[i] * p.normal[i]
			b += to[i] * p.normal[i]
		}
		side := 0
		if a < 0 {
			side = 1
		}
		if (a < 0) == (b < 0) {
			return sample(n.children[side], from, to)
		}
		fraction := a / (a - b)
		mid := from
		for i := 0; i < 3; i++ {
			mid[i] += (to[i] - from[i]) * fraction
		}
		if color, hit := sample(n.children[side], from, mid); hit {
			return color, true
		}
		for i := n.first; i < n.first+n.count; i++ {
			f := l.faces[i]
			if f.skip {
				continue
			}
			coords := [2]int{}
			inside := true
			for axis := 0; axis < 2; axis++ {
				v := f.vec[axis][3]
				for j := 0; j < 3; j++ {
					v += mid[j] * f.vec[axis][j]
				}
				coords[axis] = int(v) - f.mins[axis]
				if coords[axis] < 0 || coords[axis] > (f.size[axis]-1)*16 {
					inside = false
				}
			}
			if !inside {
				continue
			}
			if f.offset < 0 {
				return Vec3{}, true
			}
			color := Vec3{}
			stride := 3 * f.size[0] * f.size[1]
			at := f.offset + 3*((coords[1]>>4)*f.size[0]+(coords[0]>>4))
			for slot, style := range f.styles {
				if style == 255 {
					break
				}
				scale := lightStyle(config[800+int(style)], frame)
				for j := 0; j < 3; j++ {
					color[j] += float64(l.data[at+slot*stride+j]) * scale / 255
				}
			}
			return color, true
		}
		return sample(n.children[1-side], mid, to)
	}
	color, _ := sample(l.root, eye, end)
	if budget <= 0 {
		return 0, false
	}
	level := 150 * max(color[0], color[1], color[2])
	return byte(min(255, max(0, level))), true
}

func lightStyle(pattern string, frame int) float64 {
	if len(pattern) == 0 {
		return 1
	}
	if frame < 0 {
		frame = 0
	}
	v := pattern[frame%len(pattern)]
	if v < 'a' || v > 'z' {
		return 1
	}
	return float64(v-'a') / 12
}

func parseBSPLight(lump func(int, int) ([]byte, error), planes []bspPlane, nodes, models []byte) (*bspLight, error) {
	get := func(i, row int) []byte { b, _ := lump(i, row); return b }
	for _, p := range [][2]int{{2, 12}, {5, 76}, {6, 20}, {7, 0}, {11, 4}, {12, 4}} {
		if _, e := lump(p[0], p[1]); e != nil {
			return nil, e
		}
	}
	vertices, tex, faces, data, edges, surf := get(2, 12), get(5, 76), get(6, 20), get(7, 0), get(11, 4), get(12, 4)
	u16 := func(b []byte, at int) int { return int(binary.LittleEndian.Uint16(b[at:])) }
	i32 := func(b []byte, at int) int { return int(int32(binary.LittleEndian.Uint32(b[at:]))) }
	f32 := func(b []byte, at int) float64 {
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(b[at:])))
	}
	bad := func() (*bspLight, error) { return nil, fmt.Errorf("invalid BSP lighting references or samples") }
	l := &bspLight{planes: planes, nodes: make([]lightNode, len(nodes)/28), faces: make([]lightFace, len(faces)/20), data: data, root: i32(models, 36)}
	for _, p := range planes {
		for _, v := range []float64{p.normal[0], p.normal[1], p.normal[2], p.dist} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return bad()
			}
		}
	}
	if l.root < 0 || l.root >= len(l.nodes) {
		return bad()
	}
	for i := range l.nodes {
		at := i * 28
		n := lightNode{plane: i32(nodes, at), first: u16(nodes, at+24), count: u16(nodes, at+26), children: [2]int{i32(nodes, at+4), i32(nodes, at+8)}}
		if n.plane < 0 || n.plane >= len(planes) || n.first+n.count > len(l.faces) {
			return bad()
		}
		for _, c := range n.children {
			if c >= len(l.nodes) {
				return bad()
			}
		}
		l.nodes[i] = n
	}
	for i := range l.faces {
		at := i * 20
		t := u16(faces, at+10)
		first, count := i32(faces, at+4), u16(faces, at+8)
		if t >= len(tex)/76 || first < 0 || count < 3 || first > len(surf)/4 || count > len(surf)/4-first {
			return bad()
		}
		f := lightFace{offset: i32(faces, at+16), skip: i32(tex, t*76+32)&12 != 0}
		copy(f.styles[:], faces[at+12:at+16])
		for a := 0; a < 2; a++ {
			for j := 0; j < 4; j++ {
				v := f32(tex, t*76+(a*4+j)*4)
				if math.IsNaN(v) || math.IsInf(v, 0) {
					return bad()
				}
				f.vec[a][j] = v
			}
		}
		lo, hi := [2]float64{math.Inf(1), math.Inf(1)}, [2]float64{math.Inf(-1), math.Inf(-1)}
		for j := 0; j < count; j++ {
			e := i32(surf, (first+j)*4)
			side := 0
			if e < 0 {
				e = -e
				side = 1
			}
			if e >= len(edges)/4 {
				return bad()
			}
			v := u16(edges, e*4+side*2)
			if v >= len(vertices)/12 {
				return bad()
			}
			for a := 0; a < 2; a++ {
				x := f.vec[a][3]
				for k := 0; k < 3; k++ {
					x += f32(vertices, v*12+k*4) * f.vec[a][k]
				}
				if math.IsNaN(x) || math.IsInf(x, 0) || math.Abs(x) > 1e8 {
					return bad()
				}
				lo[a] = min(lo[a], x)
				hi[a] = max(hi[a], x)
			}
		}
		for a := 0; a < 2; a++ {
			f.mins[a] = int(math.Floor(lo[a]/16)) * 16
			f.size[a] = int(math.Ceil(hi[a]/16)) - f.mins[a]/16 + 1
			if f.size[a] <= 0 || f.size[a] > 1<<20 {
				return bad()
			}
		}
		if f.offset < -1 {
			return bad()
		}
		if !f.skip && f.offset >= 0 && len(data) > 0 {
			slots := 0
			for _, s := range f.styles {
				if s == 255 {
					break
				}
				slots++
			}
			size := int64(f.size[0]) * int64(f.size[1]) * 3 * int64(slots)
			if f.offset > len(data) || size > int64(len(data)-f.offset) {
				return bad()
			}
		}
		l.faces[i] = f
	}
	// Reject cycles before any runtime recursion.
	state := make([]byte, len(l.nodes))
	var visit func(int, int) bool
	visit = func(n, depth int) bool {
		if n < 0 {
			return true
		}
		if depth > 1024 || state[n] == 1 {
			return false
		}
		if state[n] == 2 {
			return true
		}
		state[n] = 1
		for _, c := range l.nodes[n].children {
			if !visit(c, depth+1) {
				return false
			}
		}
		state[n] = 2
		return true
	}
	if !visit(l.root, 0) {
		return bad()
	}
	return l, nil
}
