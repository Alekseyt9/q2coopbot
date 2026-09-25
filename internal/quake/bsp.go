package quake

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

type MapEntity struct {
	Class      string `json:"class"`
	Origin     Vec3   `json:"origin"`
	Target     string `json:"target,omitempty"`
	TargetName string `json:"target_name,omitempty"`
	Map        string `json:"map,omitempty"`
}
type MapInfo struct {
	Name      string      `json:"name"`
	BSPSource string      `json:"bsp_source"`
	Planes    int         `json:"planes"`
	Nodes     int         `json:"nodes"`
	Leaves    int         `json:"leaves"`
	Brushes   int         `json:"brushes"`
	Entities  []MapEntity `json:"entities"`
	Models    []BSPModel  `json:"-"`
	collision *CollisionMap
}

type BSPModel struct{ Min, Max, Origin Vec3 }

func (m *MapInfo) Model(index int) (BSPModel, bool) {
	if m == nil || index <= 0 || index >= len(m.Models) {
		return BSPModel{}, false
	}
	return m.Models[index], true
}

// ClearShot reports whether the loaded BSP has a clear static line of fire.
func (m *MapInfo) ClearShot(from, to Vec3) bool {
	if m == nil || m.collision == nil {
		return false
	}
	return m.collision.ClearShot(from, to)
}

func (m *MapInfo) HasCollision() bool { return m != nil && m.collision != nil }

type bspPlane struct {
	normal Vec3
	dist   float64
}
type bspBrush struct{ first, count, contents int }
type CollisionMap struct {
	planes       []bspPlane
	sides        []uint16
	brushes      []bspBrush
	worldBrushes []int
}

func (m *CollisionMap) ClearShot(from, to Vec3) bool {
	if m == nil {
		return false
	}
	for _, index := range m.worldBrushes {
		brush := m.brushes[index]
		if brush.contents&3 == 0 {
			continue
		}
		enter, leave := 0.0, 1.0
		outside := false
		for side := brush.first; side < brush.first+brush.count; side++ {
			plane := m.planes[m.sides[side]]
			d1 := -plane.dist + 1
			d2 := d1
			for axis := 0; axis < 3; axis++ {
				d1 += from[axis] * plane.normal[axis]
				d2 += to[axis] * plane.normal[axis]
			}
			if d1 > 0 && d2 > 0 {
				outside = true
				break
			}
			if d1 <= 0 && d2 <= 0 {
				continue
			}
			fraction := d1 / (d1 - d2)
			if d1 > d2 {
				enter = math.Max(enter, fraction)
			} else {
				leave = math.Min(leave, fraction)
			}
		}
		if !outside && enter < leave && enter < 1 && leave > 0 {
			return false
		}
	}
	return true
}

func readMapAsset(root, name string) ([]byte, string, error) {
	asset := "maps/" + name + ".bsp"
	path := filepath.Join(root, "maps", name+".bsp")
	if data, e := os.ReadFile(path); e == nil {
		return data, path, nil
	}
	for _, pak := range []string{"pak2.pak", "pak1.pak", "pak0.pak"} {
		path = filepath.Join(root, pak)
		f, e := os.Open(path)
		if e != nil {
			continue
		}
		data, found, e := assetFromPak(f, asset)
		f.Close()
		if e != nil {
			return nil, "", e
		}
		if found {
			return data, path + ":" + asset, nil
		}
	}
	return nil, "", os.ErrNotExist
}
func assetFromPak(f *os.File, name string) ([]byte, bool, error) {
	header := make([]byte, 12)
	if _, e := io.ReadFull(f, header); e != nil {
		return nil, false, e
	}
	if string(header[:4]) != "PACK" {
		return nil, false, errors.New("invalid PAK header")
	}
	info, e := f.Stat()
	if e != nil {
		return nil, false, e
	}
	offset := int64(binary.LittleEndian.Uint32(header[4:]))
	size := int64(binary.LittleEndian.Uint32(header[8:]))
	if offset < 12 || size < 0 || size%64 != 0 || offset+size > info.Size() {
		return nil, false, errors.New("invalid PAK directory")
	}
	dir := make([]byte, size)
	if _, e = f.ReadAt(dir, offset); e != nil {
		return nil, false, e
	}
	for p := 0; p < len(dir); p += 64 {
		entry := strings.TrimRight(string(dir[p:p+56]), "\x00")
		if !strings.EqualFold(entry, name) {
			continue
		}
		start := int64(binary.LittleEndian.Uint32(dir[p+56:]))
		length := int64(binary.LittleEndian.Uint32(dir[p+60:]))
		if start < 0 || length < 0 || start+length > info.Size() {
			return nil, false, errors.New("invalid PAK asset span")
		}
		data := make([]byte, length)
		_, e = f.ReadAt(data, start)
		return data, true, e
	}
	return nil, false, nil
}

var entityToken = regexp.MustCompile(`"([^"\\]*(?:\\.[^"\\]*)*)"|[{}]`)

func parseMapEntities(text string) []MapEntity {
	tokens := entityToken.FindAllStringSubmatch(text, -1)
	var out []MapEntity
	var props map[string]string
	var key string
	for _, t := range tokens {
		switch t[0] {
		case "{":
			props = map[string]string{}
			key = ""
		case "}":
			if props != nil {
				class := props["classname"]
				if class != "" {
					e := MapEntity{Class: class, Target: props["target"], TargetName: props["targetname"], Map: props["map"]}
					values := strings.Fields(props["origin"])
					if len(values) == 3 {
						for i, v := range values {
							e.Origin[i], _ = strconv.ParseFloat(v, 64)
						}
					}
					out = append(out, e)
				}
			}
			props = nil
			key = ""
		default:
			if props == nil {
				continue
			}
			if key == "" {
				key = t[1]
			} else {
				props[key] = t[1]
				key = ""
			}
		}
	}
	return out
}
func LoadMap(root, name string) (MapInfo, error) {
	data, source, e := readMapAsset(root, name)
	if e != nil {
		return MapInfo{}, e
	}
	if len(data) < 160 || string(data[:4]) != "IBSP" || binary.LittleEndian.Uint32(data[4:]) != 38 {
		return MapInfo{}, fmt.Errorf("invalid Quake II BSP for %s", name)
	}
	lump := func(i, row int) ([]byte, error) {
		p := 8 + i*8
		off := int(int32(binary.LittleEndian.Uint32(data[p:])))
		n := int(int32(binary.LittleEndian.Uint32(data[p+4:])))
		if off < 0 || n < 0 || off > len(data) || n > len(data)-off || row > 0 && n%row != 0 {
			return nil, fmt.Errorf("invalid BSP lump %d", i)
		}
		return data[off : off+n], nil
	}
	entities, e := lump(0, 0)
	if e != nil {
		return MapInfo{}, e
	}
	planes, e := lump(1, 20)
	if e != nil {
		return MapInfo{}, e
	}
	nodes, e := lump(4, 28)
	if e != nil {
		return MapInfo{}, e
	}
	leaves, e := lump(8, 28)
	if e != nil {
		return MapInfo{}, e
	}
	brushes, e := lump(14, 12)
	if e != nil {
		return MapInfo{}, e
	}
	sides, e := lump(15, 4)
	if e != nil {
		return MapInfo{}, e
	}
	leafBrushes, e := lump(10, 2)
	if e != nil {
		return MapInfo{}, e
	}
	models, e := lump(13, 48)
	if e != nil {
		return MapInfo{}, e
	}
	if len(models) < 48 {
		return MapInfo{}, errors.New("BSP has no world model")
	}
	modelBounds := make([]BSPModel, len(models)/48)
	for i := range modelBounds {
		for axis := 0; axis < 3; axis++ {
			at := i*48 + axis*4
			modelBounds[i].Min[axis] = float64(math.Float32frombits(binary.LittleEndian.Uint32(models[at:])))
			modelBounds[i].Max[axis] = float64(math.Float32frombits(binary.LittleEndian.Uint32(models[at+12:])))
			modelBounds[i].Origin[axis] = float64(math.Float32frombits(binary.LittleEndian.Uint32(models[at+24:])))
		}
	}
	c := &CollisionMap{planes: make([]bspPlane, len(planes)/20), sides: make([]uint16, len(sides)/4), brushes: make([]bspBrush, len(brushes)/12)}
	for i := range c.planes {
		p := i * 20
		for j := 0; j < 3; j++ {
			c.planes[i].normal[j] = float64(math.Float32frombits(binary.LittleEndian.Uint32(planes[p+j*4:])))
		}
		c.planes[i].dist = float64(math.Float32frombits(binary.LittleEndian.Uint32(planes[p+12:])))
	}
	for i := range c.sides {
		v := binary.LittleEndian.Uint16(sides[i*4:])
		if int(v) >= len(c.planes) {
			return MapInfo{}, fmt.Errorf("BSP side %d has invalid plane", i)
		}
		c.sides[i] = v
	}
	for i := range c.brushes {
		p := i * 12
		first := int(int32(binary.LittleEndian.Uint32(brushes[p:])))
		count := int(int32(binary.LittleEndian.Uint32(brushes[p+4:])))
		contents := int(int32(binary.LittleEndian.Uint32(brushes[p+8:])))
		if first < 0 || count < 0 || first > len(c.sides) || count > len(c.sides)-first {
			return MapInfo{}, fmt.Errorf("BSP brush %d has invalid sides", i)
		}
		c.brushes[i] = bspBrush{first, count, contents}
	}
	seenNode := map[int]bool{}
	seenBrush := map[int]bool{}
	var walk func(int) error
	walk = func(index int) error {
		if index < 0 {
			leaf := -1 - index
			if leaf >= len(leaves)/28 {
				return errors.New("BSP leaf index out of range")
			}
			p := leaf * 28
			first := int(binary.LittleEndian.Uint16(leaves[p+24:]))
			count := int(binary.LittleEndian.Uint16(leaves[p+26:]))
			if first+count > len(leafBrushes)/2 {
				return errors.New("BSP leaf brush span out of range")
			}
			for i := first; i < first+count; i++ {
				brush := int(binary.LittleEndian.Uint16(leafBrushes[i*2:]))
				if brush >= len(c.brushes) {
					return errors.New("BSP leaf brush index out of range")
				}
				if !seenBrush[brush] {
					c.worldBrushes = append(c.worldBrushes, brush)
					seenBrush[brush] = true
				}
			}
			return nil
		}
		if index >= len(nodes)/28 {
			return errors.New("BSP node index out of range")
		}
		if seenNode[index] {
			return nil
		}
		seenNode[index] = true
		p := index * 28
		for _, at := range []int{4, 8} {
			if e := walk(int(int32(binary.LittleEndian.Uint32(nodes[p+at:])))); e != nil {
				return e
			}
		}
		return nil
	}
	if e = walk(int(int32(binary.LittleEndian.Uint32(models[36:])))); e != nil {
		return MapInfo{}, e
	}
	return MapInfo{Name: name, BSPSource: source, Planes: len(planes) / 20, Nodes: len(nodes) / 28, Leaves: len(leaves) / 28, Brushes: len(brushes) / 12, Entities: parseMapEntities(string(entities)), Models: modelBounds, collision: c}, nil
}
