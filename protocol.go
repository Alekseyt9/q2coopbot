package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

type reader struct {
	data []byte
	pos  int
}

func (r *reader) take(n int) ([]byte, error) {
	if n < 0 || r.pos+n > len(r.data) {
		return nil, errors.New("truncated server packet")
	}
	v := r.data[r.pos : r.pos+n]
	r.pos += n
	return v, nil
}
func (r *reader) byte() (byte, error) {
	v, e := r.take(1)
	if e != nil {
		return 0, e
	}
	return v[0], nil
}
func (r *reader) short() (int16, error) {
	v, e := r.take(2)
	if e != nil {
		return 0, e
	}
	return int16(binary.LittleEndian.Uint16(v)), nil
}
func (r *reader) ushort() (uint16, error) {
	v, e := r.take(2)
	if e != nil {
		return 0, e
	}
	return binary.LittleEndian.Uint16(v), nil
}
func (r *reader) long() (int32, error) {
	v, e := r.take(4)
	if e != nil {
		return 0, e
	}
	return int32(binary.LittleEndian.Uint32(v)), nil
}
func (r *reader) str() (string, error) {
	start := r.pos
	for r.pos < len(r.data) && r.data[r.pos] != 0 {
		r.pos++
	}
	if r.pos >= len(r.data) {
		return "", errors.New("unterminated server string")
	}
	v := string(r.data[start:r.pos])
	r.pos++
	return v, nil
}
func (r *reader) skip(n int) error { _, e := r.take(n); return e }

type Entity struct {
	Number, Model, Frame int
	Origin               Vec3
}
type Frame struct {
	Number      int
	Suppressed  byte
	Origin      Vec3
	Stats       [32]int16
	Gun         int
	DeltaAngles [3]int16
	Entities    map[int]Entity
}
type Decoder struct {
	Config         map[int]string
	Baselines      map[int]Entity
	Frames         map[int]Frame
	Commands       []string
	Map            string
	PlayerNumber   int
	Teammate       *Vec3
	Errors         int
	LastError      string
	ServerdataSeen bool
	Spawncount     int
}

func NewDecoder() *Decoder {
	return &Decoder{Config: map[int]string{}, Baselines: map[int]Entity{}, Frames: map[int]Frame{}}
}
func bits(r *reader) (uint32, int, error) {
	b, e := r.byte()
	if e != nil {
		return 0, 0, e
	}
	v := uint32(b)
	if v&0x80 != 0 {
		x, e := r.byte()
		if e != nil {
			return 0, 0, e
		}
		v |= uint32(x) << 8
	}
	if v&0x8000 != 0 {
		x, e := r.byte()
		if e != nil {
			return 0, 0, e
		}
		v |= uint32(x) << 16
	}
	if v&0x800000 != 0 {
		x, e := r.byte()
		if e != nil {
			return 0, 0, e
		}
		v |= uint32(x) << 24
	}
	if v&0x100 != 0 {
		x, e := r.ushort()
		return v, int(x), e
	}
	x, e := r.byte()
	return v, int(x), e
}
func parseEntity(r *reader, number int, b uint32, old Entity) (Entity, error) {
	out := old
	out.Number = number
	for _, bit := range []uint32{0x800, 0x100000, 0x200000, 0x400000} {
		if b&bit != 0 {
			x, e := r.byte()
			if e != nil {
				return out, e
			}
			if bit == 0x800 {
				out.Model = int(x)
			}
		}
	}
	if b&0x10 != 0 {
		x, e := r.byte()
		if e != nil {
			return out, e
		}
		out.Frame = int(x)
	}
	if b&0x20000 != 0 {
		x, e := r.ushort()
		if e != nil {
			return out, e
		}
		out.Frame = int(x)
	}
	if b&0x10000 != 0 && b&0x2000000 != 0 {
		if e := r.skip(4); e != nil {
			return out, e
		}
	} else if b&0x10000 != 0 {
		if e := r.skip(1); e != nil {
			return out, e
		}
	} else if b&0x2000000 != 0 {
		if e := r.skip(2); e != nil {
			return out, e
		}
	}
	for _, pair := range [][2]uint32{{0x4000, 0x80000}, {0x1000, 0x40000}} {
		n := 0
		if b&pair[0] != 0 {
			n = 1
		}
		if b&pair[1] != 0 {
			n = 2
		}
		if b&pair[0] != 0 && b&pair[1] != 0 {
			n = 4
		}
		if e := r.skip(n); e != nil {
			return out, e
		}
	}
	for i, bit := range []uint32{1, 2, 0x200} {
		if b&bit != 0 {
			x, e := r.short()
			if e != nil {
				return out, e
			}
			out.Origin[i] = float64(x) / 8
		}
	}
	for _, bit := range []uint32{0x400, 4, 8, 0x4000000, 0x20} {
		if b&bit != 0 {
			if e := r.skip(1); e != nil {
				return out, e
			}
		}
	}
	if b&0x1000000 != 0 {
		if e := r.skip(6); e != nil {
			return out, e
		}
	}
	if b&0x8000000 != 0 {
		if e := r.skip(2); e != nil {
			return out, e
		}
	}
	return out, nil
}
func (d *Decoder) playerstate(r *reader, old Frame) (Frame, error) {
	f := old
	flags, e := r.ushort()
	if e != nil {
		return f, e
	}
	skip := func(n int) error { return r.skip(n) }
	if flags&1 != 0 {
		if e = skip(1); e != nil {
			return f, e
		}
	}
	if flags&2 != 0 {
		for i := 0; i < 3; i++ {
			var v int16
			v, e = r.short()
			if e != nil {
				return f, e
			}
			f.Origin[i] = float64(v) / 8
		}
	}
	for _, p := range [][2]int{{4, 6}, {8, 1}, {16, 1}, {32, 2}} {
		if flags&uint16(p[0]) != 0 {
			if e = skip(p[1]); e != nil {
				return f, e
			}
		}
	}
	if flags&64 != 0 {
		for i := 0; i < 3; i++ {
			f.DeltaAngles[i], e = r.short()
			if e != nil {
				return f, e
			}
		}
	}
	for _, p := range [][2]int{{128, 3}, {256, 6}, {512, 3}} {
		if flags&uint16(p[0]) != 0 {
			if e = skip(p[1]); e != nil {
				return f, e
			}
		}
	}
	if flags&4096 != 0 {
		var v byte
		v, e = r.byte()
		if e != nil {
			return f, e
		}
		f.Gun = int(v)
	}
	for _, p := range [][2]int{{8192, 7}, {1024, 4}, {2048, 1}, {16384, 1}} {
		if flags&uint16(p[0]) != 0 {
			if e = skip(p[1]); e != nil {
				return f, e
			}
		}
	}
	mask, e := r.long()
	if e != nil {
		return f, e
	}
	for i := 0; i < 32; i++ {
		if uint32(mask)&(uint32(1)<<i) != 0 {
			f.Stats[i], e = r.short()
			if e != nil {
				return f, e
			}
		}
	}
	return f, nil
}
func (d *Decoder) frame(r *reader) (Frame, error) {
	number, e := r.long()
	if e != nil {
		return Frame{}, e
	}
	delta, e := r.long()
	if e != nil {
		return Frame{}, e
	}
	old := Frame{Entities: map[int]Entity{}}
	if delta > 0 {
		v, ok := d.Frames[int(delta)]
		if !ok {
			return Frame{}, errors.New("missing delta frame")
		}
		old = v
	}
	f := old
	f.Number = int(number)
	f.Entities = make(map[int]Entity, len(old.Entities))
	for k, v := range old.Entities {
		f.Entities[k] = v
	}
	f.Suppressed, e = r.byte()
	if e != nil {
		return f, e
	}
	areaBytes, e := r.byte()
	if e != nil {
		return f, e
	}
	if e = r.skip(int(areaBytes)); e != nil {
		return f, e
	}
	op, e := r.byte()
	if e != nil {
		return f, e
	}
	if op != 17 {
		return f, errors.New("frame missing playerinfo")
	}
	f, e = d.playerstate(r, f)
	if e != nil {
		return f, e
	}
	op, e = r.byte()
	if e != nil {
		return f, e
	}
	if op != 18 {
		return f, errors.New("frame missing packetentities")
	}
	for {
		b, id, e := bits(r)
		if e != nil {
			return f, e
		}
		if id == 0 {
			break
		}
		if b&0x40 != 0 {
			delete(f.Entities, id)
			continue
		}
		oldEntity, ok := old.Entities[id]
		if !ok {
			oldEntity = d.Baselines[id]
		}
		entity, e := parseEntity(r, id, b, oldEntity)
		if e != nil {
			return f, e
		}
		f.Entities[id] = entity
	}
	d.Frames[f.Number] = f
	for key := range d.Frames {
		if f.Number-key > 16 {
			delete(d.Frames, key)
		}
	}
	return f, nil
}
func (d *Decoder) Parse(data []byte) ([]Frame, error) {
	r := reader{data: data}
	d.Commands = nil
	d.ServerdataSeen = false
	var frames []Frame
	for r.pos < len(data) {
		op, e := r.byte()
		if e != nil {
			return frames, e
		}
		switch op {
		case 6:
		case 12:
			d.ServerdataSeen = true
			d.Config = map[int]string{}
			d.Baselines = map[int]Entity{}
			d.Frames = map[int]Frame{}
			d.Map = ""
			d.Teammate = nil
			_, e = r.long()
			if e != nil {
				return frames, e
			}
			var spawncount int32
			spawncount, e = r.long()
			if e != nil {
				return frames, e
			}
			d.Spawncount = int(spawncount)
			e = r.skip(1)
			if e != nil {
				return frames, e
			}
			_, e = r.str()
			if e != nil {
				return frames, e
			}
			n, e := r.short()
			if e != nil {
				return frames, e
			}
			d.PlayerNumber = int(n) + 1
			_, e = r.str()
			if e != nil {
				return frames, e
			}
		case 13:
			idx, e := r.ushort()
			if e != nil {
				return frames, e
			}
			value, e := r.str()
			if e != nil {
				return frames, e
			}
			d.Config[int(idx)] = value
			if idx == 33 && strings.HasPrefix(value, "maps/") && strings.HasSuffix(value, ".bsp") {
				d.Map = strings.TrimSuffix(strings.TrimPrefix(value, "maps/"), ".bsp")
			}
		case 14:
			b, id, e := bits(&r)
			if e != nil {
				return frames, e
			}
			v, e := parseEntity(&r, id, b, Entity{})
			if e != nil {
				return frames, e
			}
			d.Baselines[id] = v
		case 20:
			f, e := d.frame(&r)
			if e != nil {
				return frames, e
			}
			frames = append(frames, f)
		case 10:
			if e = r.skip(1); e != nil {
				return frames, e
			}
			_, e = r.str()
			if e != nil {
				return frames, e
			}
		case 11:
			v, e := r.str()
			if e != nil {
				return frames, e
			}
			d.Commands = append(d.Commands, strings.TrimSpace(v))
		case 8:
			d.Commands = append(d.Commands, "reconnect")
		case 15, 4:
			_, e = r.str()
			if e != nil {
				return frames, e
			}
		case 1, 2:
			e = r.skip(3)
			if e != nil {
				return frames, e
			}
		case 3:
			effect, e := r.byte()
			if e != nil {
				return frames, e
			}
			if effect == 0 || effect == 1 || effect == 2 || effect == 4 || effect == 9 || effect == 12 || effect == 13 || effect == 14 {
				e = r.skip(7)
				if e != nil {
					return frames, e
				}
			} else if effect == 5 || effect == 6 || effect == 7 || effect == 8 || effect == 17 || effect == 18 {
				// Explosion temporary entities carry one packed position.
				if e = r.skip(6); e != nil {
					return frames, e
				}
			} else if effect == 10 {
				if e = r.skip(9); e != nil {
					return frames, e
				}
			} else {
				return frames, fmt.Errorf("unsupported temp entity %d", effect)
			}
		case 9:
			flags, e := r.byte()
			if e != nil {
				return frames, e
			}
			if e = r.skip(1); e != nil {
				return frames, e
			}
			n := 0
			for _, bit := range []byte{1, 2, 16} {
				if flags&bit != 0 {
					n++
				}
			}
			if flags&8 != 0 {
				n += 2
			}
			if flags&4 != 0 {
				n += 6
			}
			if e = r.skip(n); e != nil {
				return frames, e
			}
		case 5:
			e = r.skip(512)
			if e != nil {
				return frames, e
			}
		case 16:
			size, e := r.short()
			if e != nil {
				return frames, e
			}
			if e = r.skip(1); e != nil {
				return frames, e
			}
			if size > 0 {
				if e = r.skip(int(size)); e != nil {
					return frames, e
				}
			}
		default:
			return frames, fmt.Errorf("unsupported opcode %d at %d", op, r.pos-1)
		}
	}
	return frames, nil
}

type Object struct {
	ID        int    `json:"id"`
	Class     string `json:"class"`
	Origin    Vec3   `json:"origin"`
	Frame     int    `json:"frame,omitempty"`
	ClearShot *bool  `json:"clear_shot,omitempty"`
}
type Snapshot struct {
	Map         string   `json:"map"`
	Frame       int      `json:"frame"`
	Self        Vec3     `json:"self"`
	Teammate    *Vec3    `json:"teammate,omitempty"`
	Health      int16    `json:"health"`
	Armor       int16    `json:"armor"`
	Ammo        int16    `json:"ammo"`
	Weapon      string   `json:"weapon"`
	DeltaAngles [3]int16 `json:"delta_angles"`
	Enemies     []Object `json:"enemies"`
	Pickups     []Object `json:"pickups"`
}

func (d *Decoder) Snapshot(f Frame) Snapshot {
	s := Snapshot{Map: d.Map, Frame: f.Number, Self: f.Origin, Health: f.Stats[1], Armor: f.Stats[5], Ammo: f.Stats[3], DeltaAngles: f.DeltaAngles}
	maxclients, _ := strconv.Atoi(d.Config[30])
	if maxclients <= 0 {
		maxclients = 4
	}
	gun := strings.ToLower(d.Config[32+f.Gun])
	if strings.Contains(gun, "blast") {
		s.Weapon = "Blaster"
	} else {
		s.Weapon = gun
	}
	for _, entity := range f.Entities {
		if entity.Number > 0 && entity.Number <= maxclients && entity.Number != d.PlayerNumber && entity.Model == 255 {
			p := entity.Origin
			if s.Teammate == nil || entity.Number < maxclients {
				s.Teammate = &p
			}
		}
	}
	if s.Teammate != nil {
		d.Teammate = s.Teammate
	} else {
		s.Teammate = d.Teammate
	}
	for _, entity := range f.Entities {
		path := strings.ToLower(d.Config[32+entity.Model])
		if strings.Contains(path, "/monsters/") && distance(entity.Origin, f.Origin) < 1024 {
			kind := strings.SplitN(strings.SplitN(path, "/monsters/", 2)[1], "/", 2)[0]
			if kind == "soldier" && entity.Frame >= 272 && entity.Frame <= 474 || kind == "infantry" && entity.Frame >= 125 && entity.Frame <= 178 {
				continue
			}
			s.Enemies = append(s.Enemies, Object{ID: entity.Number, Class: "monster_" + kind, Origin: entity.Origin, Frame: entity.Frame})
		} else if strings.Contains(path, "/items/") && distance(entity.Origin, f.Origin) < 384 {
			kind := strings.SplitN(strings.SplitN(path, "/items/", 2)[1], "/", 2)[0]
			if strings.Contains(kind, "heal") {
				kind = "health"
			}
			s.Pickups = append(s.Pickups, Object{ID: entity.Number, Class: "item_" + kind, Origin: entity.Origin, Frame: entity.Frame})
		}
	}
	return s
}
func yawTo(from, to Vec3, delta int16) int16 {
	radians := math.Atan2(to[1]-from[1], to[0]-from[0])
	return int16(int(math.Round(radians*65536/(2*math.Pi))) - int(delta))
}
func pitchTo(from, to Vec3, delta int16) int16 {
	radians := math.Atan2(to[2]-from[2], horizontal(from, to))
	return int16(-int(math.Round(radians*65536/(2*math.Pi))) - int(delta))
}
