package quake

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
)

// MapOutline contains upward-facing BSP polygons in original world coordinates.
// Overlapping floors retain their Z coordinates for height filtering in the viewer.
type MapOutline struct {
	Name   string   `json:"name"`
	Source string   `json:"source"`
	Floors [][]Vec3 `json:"floors"`
}

func LoadMapOutline(root, name string) (MapOutline, error) {
	if !regexp.MustCompile(`^[a-zA-Z0-9_-]+$`).MatchString(name) {
		return MapOutline{}, fmt.Errorf("invalid map name")
	}
	data, source, err := readMapAsset(root, name)
	if err != nil {
		source = filepath.Join(root, name+".bsp")
		data, err = os.ReadFile(source)
	}
	if err != nil {
		return MapOutline{}, err
	}
	return parseOutline(data, name, source)
}

func parseOutline(data []byte, name, source string) (MapOutline, error) {
	out := MapOutline{Name: name, Source: source, Floors: make([][]Vec3, 0)}
	if len(data) < 160 || string(data[:4]) != "IBSP" || binary.LittleEndian.Uint32(data[4:]) != 38 {
		return out, fmt.Errorf("invalid Quake II BSP")
	}
	lump := func(i, row int) ([]byte, error) {
		p := 8 + i*8
		off := uint64(binary.LittleEndian.Uint32(data[p:]))
		n := uint64(binary.LittleEndian.Uint32(data[p+4:]))
		if off+n > uint64(len(data)) || n%uint64(row) != 0 {
			return nil, fmt.Errorf("invalid BSP lump %d", i)
		}
		return data[off : off+n], nil
	}
	planes, e := lump(1, 20)
	if e != nil {
		return out, e
	}
	vertices, e := lump(2, 12)
	if e != nil {
		return out, e
	}
	faces, e := lump(6, 20)
	if e != nil {
		return out, e
	}
	edges, e := lump(11, 4)
	if e != nil {
		return out, e
	}
	surf, e := lump(12, 4)
	if e != nil {
		return out, e
	}
	f32 := func(b []byte) float64 { return float64(math.Float32frombits(binary.LittleEndian.Uint32(b))) }
	for p := 0; p < len(faces); p += 20 {
		face := faces[p:]
		plane := int(binary.LittleEndian.Uint16(face)) * 20
		if plane+20 > len(planes) {
			return out, fmt.Errorf("invalid face plane")
		}
		nz := f32(planes[plane+8:])
		if binary.LittleEndian.Uint16(face[2:]) != 0 {
			nz = -nz
		}
		if nz < 0.5 {
			continue
		}
		first := int(int32(binary.LittleEndian.Uint32(face[4:])))
		count := int(binary.LittleEndian.Uint16(face[8:]))
		if first < 0 || first+count > len(surf)/4 {
			return out, fmt.Errorf("invalid face edges")
		}
		poly := make([]Vec3, 0, count)
		for i := first; i < first+count; i++ {
			edge := int(int32(binary.LittleEndian.Uint32(surf[i*4:])))
			side := 0
			if edge < 0 {
				edge = -edge
				side = 2
			}
			if edge >= len(edges)/4 {
				return out, fmt.Errorf("invalid edge")
			}
			v := int(binary.LittleEndian.Uint16(edges[edge*4+side:])) * 12
			if v+12 > len(vertices) {
				return out, fmt.Errorf("invalid vertex")
			}
			point := Vec3{f32(vertices[v:]), f32(vertices[v+4:]), f32(vertices[v+8:])}
			for _, x := range point {
				if math.IsNaN(x) || math.IsInf(x, 0) {
					return out, fmt.Errorf("nonfinite vertex")
				}
			}
			poly = append(poly, point)
		}
		if len(poly) >= 3 {
			out.Floors = append(out.Floors, poly)
		}
	}
	return out, nil
}
