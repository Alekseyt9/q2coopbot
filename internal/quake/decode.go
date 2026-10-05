package quake

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
	Skin                 uint32
	RenderFX             int
	Solid                uint16
	Origin               Vec3
	Angles               Vec3
	OldOrigin            Vec3
}
type Frame struct {
	Number          int
	DeltaFrame      int
	Suppressed      byte
	RemovedEntities []int
	Origin          Vec3
	Velocity        Vec3
	PMFlags         byte
	Gravity         int16
	Stats           [32]int16
	Gun             int
	GunFrame        int
	ViewAngles      [3]int16
	DeltaAngles     [3]int16
	Entities        map[int]Entity
}
type SoundEvent struct {
	Index       byte    `json:"index"`
	Name        string  `json:"name,omitempty"`
	Entity      int     `json:"entity,omitempty"`
	Channel     int     `json:"channel,omitempty"`
	Attenuation float64 `json:"attenuation"`
	Position    *Vec3   `json:"position,omitempty"`
}
type ExplosionEvent struct {
	Kind     byte `json:"kind"`
	Position Vec3 `json:"position"`
}

const playerSkinsConfigBase = 32 + 5*256 // CS_PLAYERSKINS in protocol 34.
type Decoder struct {
	Inventory          [256]int16
	InventoryKnown     bool
	InventoryFrame     int
	latestFrame        int
	Config             map[int]string
	Baselines          map[int]Entity
	Frames             map[int]Frame
	Commands           []string
	Sounds             []SoundEvent
	Explosions         []ExplosionEvent
	Map                string
	PlayerNumber       int
	lastTeammate       *Vec3
	lastTeammateAt     int
	lastTeammateEntity int
	lastMap            string
	Errors             int
	LastError          string
	ServerdataSeen     bool
	Spawncount         int
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
		v, e := r.long()
		if e != nil {
			return out, e
		}
		out.Skin = uint32(v)
	} else if b&0x10000 != 0 {
		v, e := r.byte()
		if e != nil {
			return out, e
		}
		out.Skin = uint32(v)
	} else if b&0x2000000 != 0 {
		v, e := r.ushort()
		if e != nil {
			return out, e
		}
		out.Skin = uint32(v)
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
		if pair[0] == 0x1000 && n > 0 {
			switch n {
			case 1:
				v, e := r.byte()
				if e != nil {
					return out, e
				}
				out.RenderFX = int(v)
			case 2:
				v, e := r.ushort()
				if e != nil {
					return out, e
				}
				out.RenderFX = int(v)
			case 4:
				v, e := r.long()
				if e != nil {
					return out, e
				}
				out.RenderFX = int(uint32(v))
			}
		} else if e := r.skip(n); e != nil {
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
	for axis, bit := range []uint32{0x400, 4, 8} {
		if b&bit != 0 {
			v, err := r.byte()
			if err != nil {
				return out, err
			}
			out.Angles[axis] = float64(v) * 360 / 256
		}
	}
	for _, bit := range []uint32{0x4000000, 0x20} {
		if b&bit != 0 {
			if e := r.skip(1); e != nil {
				return out, e
			}
		}
	}
	if b&0x1000000 != 0 {
		for i := range out.OldOrigin {
			v, e := r.short()
			if e != nil {
				return out, e
			}
			out.OldOrigin[i] = float64(v) / 8
		}
	}
	if b&0x8000000 != 0 {
		var e error
		out.Solid, e = r.ushort()
		if e != nil {
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
	if flags&4 != 0 {
		for i := range f.Velocity {
			v, err := r.short()
			if err != nil {
				return f, err
			}
			f.Velocity[i] = float64(v) / 8
		}
	}
	if flags&8 != 0 {
		if e = skip(1); e != nil {
			return f, e
		}
	}
	if flags&16 != 0 {
		f.PMFlags, e = r.byte()
		if e != nil {
			return f, e
		}
	}
	if flags&32 != 0 {
		f.Gravity, e = r.short()
		if e != nil {
			return f, e
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
			if p[0] == 256 {
				for axis := range f.ViewAngles {
					f.ViewAngles[axis], e = r.short()
					if e != nil {
						return f, e
					}
				}
				continue
			}
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
	if flags&8192 != 0 {
		var frame byte
		frame, e = r.byte()
		if e != nil {
			return f, e
		}
		f.GunFrame = int(frame)
		if e = skip(6); e != nil {
			return f, e
		}
	}
	for _, p := range [][2]int{{1024, 4}, {2048, 1}, {16384, 1}} {
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
	f.DeltaFrame = int(delta)
	f.RemovedEntities = nil
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
			f.RemovedEntities = append(f.RemovedEntities, id)
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
	d.Sounds = nil
	d.Explosions = nil
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
			d.Inventory, d.InventoryKnown, d.InventoryFrame, d.latestFrame = [256]int16{}, false, 0, 0
			d.ServerdataSeen = true
			d.Sounds = nil
			d.Explosions = nil
			d.Config = map[int]string{}
			d.Baselines = map[int]Entity{}
			d.Frames = map[int]Frame{}
			d.Map = ""
			d.lastTeammate = nil
			d.lastTeammateAt = 0
			d.lastTeammateEntity = 0
			d.lastMap = ""
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
			if d.lastTeammateEntity > 0 && int(idx) == playerSkinsConfigBase+d.lastTeammateEntity-1 {
				d.lastTeammate = nil
				d.lastTeammateAt = 0
				d.lastTeammateEntity = 0
			}
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
			d.latestFrame = f.Number
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
			// svc_reconnect drops the netchannel (e.g. server restart). It is
			// distinct from stufftext reconnect used by ordinary map changes.
			d.Commands = append(d.Commands, "server_reconnect")
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
			} else if effect == 3 {
				// TE_RAILTRAIL: two packed positions.
				if e = r.skip(12); e != nil {
					return frames, e
				}
			} else if effect == 15 {
				// TE_LASER_SPARKS: count, position, direction, color.
				if e = r.skip(9); e != nil {
					return frames, e
				}
			} else if effect == 5 || effect == 6 || effect == 7 || effect == 8 || effect == 17 || effect == 18 {
				// Explosion temporary entities carry one packed position.
				var position Vec3
				for i := range position {
					coord, err := r.short()
					if err != nil {
						return frames, err
					}
					position[i] = float64(coord) / 8
				}
				d.Explosions = append(d.Explosions, ExplosionEvent{effect, position})
			} else if effect == 16 || effect == 19 {
				// Parasite/medic beam: entity short and two packed positions.
				if e = r.skip(14); e != nil {
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
			index, e := r.byte()
			if e != nil {
				return frames, e
			}
			if flags&^byte(31) != 0 {
				return frames, fmt.Errorf("unsupported sound flags %d", flags)
			}
			sound := SoundEvent{Index: index, Name: d.Config[288+int(index)], Attenuation: 1}
			if flags&1 != 0 {
				if e = r.skip(1); e != nil {
					return frames, e
				}
			}
			if flags&2 != 0 {
				attenuation, err := r.byte()
				if err != nil {
					return frames, err
				}
				sound.Attenuation = float64(attenuation) / 64
			}
			if flags&16 != 0 {
				if e = r.skip(1); e != nil {
					return frames, e
				}
			}
			if flags&8 != 0 {
				channel, err := r.ushort()
				if err != nil {
					return frames, err
				}
				sound.Entity, sound.Channel = int(channel>>3), int(channel&7)
			}
			if flags&4 != 0 {
				var position Vec3
				for i := range position {
					coord, err := r.short()
					if err != nil {
						return frames, err
					}
					position[i] = float64(coord) / 8
				}
				sound.Position = &position
			}
			d.Sounds = append(d.Sounds, sound)
		case 5:
			data, err := r.take(512)
			e = err
			if e != nil {
				return frames, e
			}
			for i := range d.Inventory {
				d.Inventory[i] = int16(binary.LittleEndian.Uint16(data[i*2:]))
			}
			d.InventoryKnown, d.InventoryFrame = true, d.latestFrame
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
	ModelPath    string `json:"model_path,omitempty"`
	Angles       Vec3   `json:"angles,omitempty"`
	Skin         uint32 `json:"skin,omitempty"`
	ID           int    `json:"id"`
	Class        string `json:"class"`
	Origin       Vec3   `json:"origin"`
	Frame        int    `json:"frame,omitempty"`
	Solid        uint16 `json:"solid,omitempty"`
	ClearShot    *bool  `json:"clear_shot,omitempty"`
	HealthAmount int    `json:"health_amount,omitempty"`
}
type Mover struct {
	ID     int  `json:"id"`
	Model  int  `json:"model"`
	Origin Vec3 `json:"origin"`
	Angles Vec3 `json:"angles"`
}
type BeamObservation struct {
	ID     int  `json:"id"`
	Origin Vec3 `json:"origin"`
	End    Vec3 `json:"end"`
	Frame  int  `json:"frame"`
}
type Snapshot struct {
	Ducked             bool              `json:"ducked"`
	Inventory          []InventoryItem   `json:"inventory,omitempty"`
	InventoryKnown     bool              `json:"inventory_known"`
	InventoryAgeFrames int               `json:"inventory_age_frames"`
	InventoryOpen      bool              `json:"inventory_open"`
	Map                string            `json:"map"`
	Frame              int               `json:"frame"`
	DeltaFrame         int               `json:"delta_frame,omitempty"`
	Self               Vec3              `json:"self"`
	SelfVelocity       Vec3              `json:"self_velocity"`
	Gravity            int16             `json:"gravity"`
	OnGround           bool              `json:"on_ground"`
	Teammate           *Vec3             `json:"teammate,omitempty"`
	TeammateEntity     int               `json:"teammate_entity,omitempty"`
	LastTeammate       *Vec3             `json:"last_teammate,omitempty"`
	LastTeammateEntity int               `json:"last_teammate_entity,omitempty"`
	TeammateAgeFrames  *int              `json:"teammate_age_frames,omitempty"`
	Health             int16             `json:"health"`
	Armor              int16             `json:"armor"`
	Ammo               int16             `json:"ammo"`
	Weapon             string            `json:"weapon"`
	GunFrame           int               `json:"gun_frame"`
	ViewAngles         [3]int16          `json:"view_angles"`
	DeltaAngles        [3]int16          `json:"delta_angles"`
	Enemies            []Object          `json:"enemies"`
	Projectiles        []Object          `json:"projectiles,omitempty"`
	Barrels            []Object          `json:"barrels,omitempty"`
	Obstacles          []Object          `json:"obstacles,omitempty"`
	Defeated           []Object          `json:"defeated,omitempty"`
	Pickups            []Object          `json:"pickups"`
	Movers             []Mover           `json:"movers,omitempty"`
	Beams              []BeamObservation `json:"beams,omitempty"`
	RemovedEntities    []int             `json:"removed_entities,omitempty"`
	Suppressed         byte              `json:"suppressed,omitempty"`
	Sounds             []SoundEvent      `json:"sounds,omitempty"`
	Explosions         []ExplosionEvent  `json:"explosions,omitempty"`
}

// ResetTeammateHistory discards observations made during harness placement.
// Current visible entities will be observed normally on the next snapshot.
func (d *Decoder) ResetTeammateHistory() {
	d.lastTeammate = nil
	d.lastTeammateAt = 0
	d.lastTeammateEntity = 0
}

func (d *Decoder) Snapshot(f Frame) Snapshot {
	s := Snapshot{Map: d.Map, Frame: f.Number, DeltaFrame: f.DeltaFrame, Self: f.Origin, SelfVelocity: f.Velocity, Gravity: f.Gravity, Ducked: f.PMFlags&1 != 0, OnGround: f.PMFlags&4 != 0, Health: f.Stats[1], Armor: f.Stats[5], Ammo: f.Stats[3], DeltaAngles: f.DeltaAngles, RemovedEntities: append([]int(nil), f.RemovedEntities...), Suppressed: f.Suppressed}
	s.GunFrame, s.ViewAngles = f.GunFrame, f.ViewAngles
	s.InventoryKnown, s.InventoryOpen = d.InventoryKnown, f.Stats[13]&2 != 0
	if d.InventoryKnown {
		s.InventoryAgeFrames = max(0, f.Number-d.InventoryFrame)
		for id, count := range d.Inventory {
			if count != 0 {
				s.Inventory = append(s.Inventory, InventoryItem{ID: id, Name: d.Config[32+4*256+id], Count: int(count)})
			}
		}
	}
	if d.lastMap != d.Map || f.Number < d.lastTeammateAt {
		d.lastTeammate = nil
		d.lastTeammateAt = 0
		d.lastTeammateEntity = 0
		d.lastMap = d.Map
	}
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
		if entity.RenderFX&128 != 0 && entity.Model != 0 {
			s.Beams = append(s.Beams, BeamObservation{ID: entity.Number, Origin: entity.Origin, End: entity.OldOrigin, Frame: entity.Frame})
		}
		if entity.Number > 0 && entity.Number <= maxclients && entity.Number != d.PlayerNumber && entity.Model == 255 {
			p := entity.Origin
			if s.TeammateEntity == 0 || entity.Number < s.TeammateEntity {
				s.Teammate = &p
				s.TeammateEntity = entity.Number
			}
		}
	}
	if s.Teammate != nil {
		p := *s.Teammate
		d.lastTeammate = &p
		d.lastTeammateAt = f.Number
		d.lastTeammateEntity = s.TeammateEntity
	}
	if d.lastTeammate != nil {
		p := *d.lastTeammate
		age := f.Number - d.lastTeammateAt
		s.LastTeammate = &p
		s.LastTeammateEntity = d.lastTeammateEntity
		s.TeammateAgeFrames = &age
	}
	for _, entity := range f.Entities {
		path := strings.ToLower(d.Config[32+entity.Model])
		if path == "models/objects/barrels/tris.md2" && entity.Solid != 0 {
			s.Barrels = append(s.Barrels, Object{ID: entity.Number, Class: "misc_explobox", Origin: entity.Origin, Solid: entity.Solid})
		}
		if path == "models/objects/grenade2/tris.md2" {
			s.Projectiles = append(s.Projectiles, Object{ID: entity.Number, Class: "hand_grenade", Origin: entity.Origin})
		}
		if path == "models/objects/grenade/tris.md2" {
			s.Projectiles = append(s.Projectiles, Object{ID: entity.Number, Class: "grenade", Origin: entity.Origin})
		}
		if path == "models/objects/rocket/tris.md2" {
			s.Projectiles = append(s.Projectiles, Object{ID: entity.Number, Class: "rocket", Origin: entity.Origin})
		}
		if path == "models/objects/laser/tris.md2" {
			s.Projectiles = append(s.Projectiles, Object{ID: entity.Number, Class: "blaster_bolt", Origin: entity.Origin})
		}
		if strings.HasPrefix(path, "*") {
			if model, err := strconv.Atoi(strings.TrimPrefix(path, "*")); err == nil {
				s.Movers = append(s.Movers, Mover{ID: entity.Number, Model: model, Origin: entity.Origin, Angles: entity.Angles})
			}
		}
		if strings.Contains(path, "/monsters/") && Distance(entity.Origin, f.Origin) < 1024 {
			kind := strings.SplitN(strings.SplitN(path, "/monsters/", 2)[1], "/", 2)[0]
			// Death animations stop being combat targets before their server
			// bounding box necessarily stops colliding with players.
			if entity.Solid != 0 {
				s.Obstacles = append(s.Obstacles, Object{ID: entity.Number, Class: "monster_" + kind, Origin: entity.Origin, Frame: entity.Frame})
			}
			if monsterDeathAnimation(kind, entity.Frame) {
				s.Defeated = append(s.Defeated, Object{ID: entity.Number, Class: "monster_" + kind, Origin: entity.Origin, Frame: entity.Frame})
				continue
			}
			s.Enemies = append(s.Enemies, Object{ModelPath: path, Angles: entity.Angles, Skin: entity.Skin, ID: entity.Number, Class: "monster_" + kind, Origin: entity.Origin, Frame: entity.Frame, Solid: entity.Solid})
		} else if kind := pickupModels[path]; kind != "" && Distance(entity.Origin, f.Origin) < 384 {
			s.Pickups = append(s.Pickups, Object{ID: entity.Number, Class: kind, Origin: entity.Origin, Frame: entity.Frame})
		} else if strings.Contains(path, "/items/") && Distance(entity.Origin, f.Origin) < 384 {
			kind := strings.SplitN(strings.SplitN(path, "/items/", 2)[1], "/", 2)[0]
			if strings.Contains(kind, "heal") || healthModelAmount(path) > 0 {
				kind = "health"
			}
			s.Pickups = append(s.Pickups, Object{ID: entity.Number, Class: "item_" + kind, Origin: entity.Origin, Frame: entity.Frame, HealthAmount: healthModelAmount(path)})
		}
	}
	return s
}
func YawTo(from, to Vec3, delta int16) int16 {
	radians := math.Atan2(to[1]-from[1], to[0]-from[0])
	return int16(int(math.Round(radians*65536/(2*math.Pi))) - int(delta))
}
func PitchTo(from, to Vec3, delta int16) int16 {
	radians := math.Atan2(to[2]-from[2], Horizontal(from, to))
	return int16(-int(math.Round(radians*65536/(2*math.Pi))) - int(delta))
}
