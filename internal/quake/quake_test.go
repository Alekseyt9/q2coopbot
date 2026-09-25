package quake

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"strings"
	"testing"
)

func TestConnectRequestUsesQuakeUserinfoSeparators(t *testing.T) {
	request := ConnectRequest(1234, 5678, "GoCoopMate")
	if !strings.Contains(request, `"\name\GoCoopMate\skin\male/grunt`) ||
		strings.Contains(request, `\\name`) {
		t.Fatalf("invalid Quake userinfo in connect request: %q", request)
	}
}

func TestMovePacketMatchesProtocol34Reference(t *testing.T) {
	got := MovePacket(UserCmd{Yaw: 8192, Forward: 400, Buttons: 1, Msec: 50}, UserCmd{}, 7)
	want, _ := hex.DecodeString("02a4ffffffff0000000000004a00209001013200")
	if !bytes.Equal(got, want) {
		t.Fatalf("move packet = %x, want %x", got, want)
	}
}
func TestDecodeFullFrameAndTeammate(t *testing.T) {
	d := NewDecoder()
	buf := []byte{12}
	buf = binary.LittleEndian.AppendUint32(buf, 34)
	buf = binary.LittleEndian.AppendUint32(buf, 7)
	buf = append(buf, 0)
	buf = append(buf, []byte("baseq2\x00")...)
	buf = binary.LittleEndian.AppendUint16(buf, 0)
	buf = append(buf, []byte("Outer Base\x00")...)
	config := func(index uint16, value string) {
		buf = append(buf, 13)
		buf = binary.LittleEndian.AppendUint16(buf, index)
		buf = append(buf, []byte(value)...)
		buf = append(buf, 0)
	}
	config(33, "maps/base1.bsp")
	config(30, "4")
	config(34, "models/monsters/soldier/tris.md2")
	if _, e := d.Parse(buf); e != nil {
		t.Fatal(e)
	}
	if !d.ServerdataSeen || d.Spawncount != 7 {
		t.Fatalf("serverdata not captured: seen=%t spawn=%d", d.ServerdataSeen, d.Spawncount)
	}
	buf = []byte{20}
	buf = binary.LittleEndian.AppendUint32(buf, 10)
	buf = binary.LittleEndian.AppendUint32(buf, ^uint32(0))
	buf = append(buf, 0, 0, 17)
	buf = binary.LittleEndian.AppendUint16(buf, 0x1042)
	for _, v := range []int16{800, -160, 192, 0, 8192, 0} {
		buf = binary.LittleEndian.AppendUint16(buf, uint16(v))
	}
	buf = append(buf, 4)
	buf = binary.LittleEndian.AppendUint32(buf, 2)
	buf = binary.LittleEndian.AppendUint16(buf, 83)
	buf = append(buf, 18)
	entity := func(id, model byte, x, y, z int16) {
		buf = append(buf, 0x83, 0x0a, id, model)
		for _, v := range []int16{x, y, z} {
			buf = binary.LittleEndian.AppendUint16(buf, uint16(v))
		}
	}
	entity(2, 255, 256, -128, 192)
	entity(5, 2, 720, 0, 192)
	buf = append(buf, 0, 0)
	frames, e := d.Parse(buf)
	if e != nil {
		t.Fatal(e)
	}
	if len(frames) != 1 {
		t.Fatalf("frames=%d", len(frames))
	}
	s := d.Snapshot(frames[0])
	if s.Map != "base1" || s.Self != (Vec3{100, -20, 24}) || s.Teammate == nil || *s.Teammate != (Vec3{32, -16, 24}) || s.Health != 83 || len(s.Enemies) != 1 {
		t.Fatalf("decoded snapshot: %+v", s)
	}
}
func TestSnapshotExposesGroundAndMover(t *testing.T) {
	d := NewDecoder()
	d.Config[30] = "4"
	d.Config[39] = "*50"
	s := d.Snapshot(Frame{Number: 3, PMFlags: 4, Entities: map[int]Entity{9: {Number: 9, Model: 7, Origin: Vec3{0, 0, -190}}}})
	if !s.OnGround || len(s.Movers) != 1 || s.Movers[0].Model != 50 || s.Movers[0].Origin[2] != -190 {
		t.Fatalf("mover snapshot: %+v", s)
	}
}

func TestSnapshotSeparatesVisibleAndLastSeenTeammate(t *testing.T) {
	d := NewDecoder()
	d.Map, d.PlayerNumber = "base1", 1
	d.Config[30] = "4"
	position := Vec3{32, -16, 24}
	visible := d.Snapshot(Frame{Number: 10, Entities: map[int]Entity{2: {Number: 2, Model: 255, Origin: position}}})
	if visible.Teammate == nil || visible.LastTeammate == nil || visible.TeammateAgeFrames == nil || *visible.TeammateAgeFrames != 0 {
		t.Fatalf("visible teammate memory=%+v", visible)
	}
	hidden := d.Snapshot(Frame{Number: 13, Entities: map[int]Entity{}})
	if hidden.Teammate != nil || hidden.LastTeammate == nil || *hidden.LastTeammate != position || hidden.TeammateAgeFrames == nil || *hidden.TeammateAgeFrames != 3 {
		t.Fatalf("PVS loss was treated as current visibility: %+v", hidden)
	}
	d.Map = "base2"
	changed := d.Snapshot(Frame{Number: 1, Entities: map[int]Entity{}})
	if changed.Teammate != nil || changed.LastTeammate != nil || changed.TeammateAgeFrames != nil {
		t.Fatalf("old-map teammate survived transition: %+v", changed)
	}
}

func TestSoundPacketDistinguishesEntityFromExplicitPosition(t *testing.T) {
	d := NewDecoder()
	d.Config[288+7] = "player/step.wav"
	entitySound := []byte{9, 8, 7}
	entitySound = binary.LittleEndian.AppendUint16(entitySound, 2<<3|3)
	if _, err := d.Parse(entitySound); err != nil {
		t.Fatal(err)
	}
	if len(d.Sounds) != 1 || d.Sounds[0].Entity != 2 || d.Sounds[0].Channel != 3 ||
		d.Sounds[0].Position != nil || d.Sounds[0].Name != "player/step.wav" {
		t.Fatalf("entity-relative sound invented a position: %+v", d.Sounds)
	}
	positioned := []byte{9, 1 | 2 | 4 | 8 | 16, 7, 255, 64, 5}
	positioned = binary.LittleEndian.AppendUint16(positioned, 2<<3|3)
	for _, coord := range []int16{800, -160, 192} {
		positioned = binary.LittleEndian.AppendUint16(positioned, uint16(coord))
	}
	if _, err := d.Parse(positioned); err != nil {
		t.Fatal(err)
	}
	if len(d.Sounds) != 1 || d.Sounds[0].Position == nil || *d.Sounds[0].Position != (Vec3{100, -20, 24}) {
		t.Fatalf("explicit sound position was lost: %+v", d.Sounds)
	}
	if _, err := d.Parse([]byte{9, 4, 7, 0}); err == nil {
		t.Fatal("truncated sound position was accepted")
	}
}
func TestDecodePlayerGroundFlag(t *testing.T) {
	d := NewDecoder()
	packet := binary.LittleEndian.AppendUint16(nil, 16)  // PS_M_FLAGS
	packet = append(packet, 4)                           // PMF_ON_GROUND
	packet = binary.LittleEndian.AppendUint32(packet, 0) // no changed stats
	f, err := d.playerstate(&reader{data: packet}, Frame{})
	if err != nil || f.PMFlags != 4 {
		t.Fatalf("playerstate flags=%d err=%v", f.PMFlags, err)
	}
}
func TestMapReconnectOpcode(t *testing.T) {
	d := NewDecoder()
	_, e := d.Parse([]byte{8})
	if e != nil {
		t.Fatal(e)
	}
	if len(d.Commands) != 1 || d.Commands[0] != "reconnect" {
		t.Fatalf("commands=%v", d.Commands)
	}
}
func TestDecodeExplosionTemporaryEntity(t *testing.T) {
	d := NewDecoder()
	for _, effect := range []byte{5, 6, 7, 8, 17, 18} {
		packet := append([]byte{3, effect}, make([]byte, 6)...)
		if _, err := d.Parse(packet); err != nil {
			t.Fatalf("temporary entity %d: %v", effect, err)
		}
	}
}
func TestRouteUsesReachabilities(t *testing.T) {
	n := &Navigator{Areas: []Area{{}, {Min: Vec3{-10, -10, -10}, Max: Vec3{10, 10, 10}}, {Min: Vec3{90, -10, -10}, Max: Vec3{110, 10, 10}}}, Edges: [][]Edge{{}, {{To: 2, Start: Vec3{8, 0, 0}, End: Vec3{92, 0, 0}, Kind: 2, Cost: 10}}, nil}}
	route, ok := n.Route(Vec3{0, 0, 0}, Vec3{100, 0, 0})
	if !ok || len(route) != 2 || route[1].Position != (Vec3{92, 0, 0}) {
		t.Fatalf("route=%+v ok=%t", route, ok)
	}
	if _, ok = n.Route(Vec3{100, 0, 0}, Vec3{0, 0, 0}); ok {
		t.Fatal("invented route through missing reverse edge")
	}
}
func TestRoutePreservesElevatorReach(t *testing.T) {
	n := &Navigator{Areas: []Area{{},
		{Min: Vec3{-100, -10, -60}, Max: Vec3{-1, 10, -20}, Center: Vec3{-50, 0, -40}},
		{Min: Vec3{0, -10, 80}, Max: Vec3{100, 10, 120}, Center: Vec3{50, 0, 100}},
	}, Edges: [][]Edge{{}, {{To: 2, Start: Vec3{-1, 0, -40}, End: Vec3{1, 0, 100}, Kind: 11, Model: 50, Rise: 190, Cost: 145}}, nil}}
	route, ok := n.Route(Vec3{-50, 0, -40}, Vec3{50, 0, 100})
	if !ok || len(route) != 2 || route[0].ElevatorPhase != "board" || route[1].ElevatorPhase != "exit" || route[0].Model != 50 || route[0].Rise != 190 || route[0].ToArea != 2 {
		t.Fatalf("elevator route=%+v ok=%t", route, ok)
	}
}
func TestBSPBrushBlocksLineOfFire(t *testing.T) {
	box := &CollisionMap{
		planes: []bspPlane{{Vec3{1, 0, 0}, 10}, {Vec3{-1, 0, 0}, 0}, {Vec3{0, 1, 0}, 10}, {Vec3{0, -1, 0}, 0}, {Vec3{0, 0, 1}, 10}, {Vec3{0, 0, -1}, 0}},
		sides:  []uint16{0, 1, 2, 3, 4, 5}, brushes: []bspBrush{{0, 6, 1}}, worldBrushes: []int{0},
	}
	if box.ClearShot(Vec3{-5, 5, 5}, Vec3{15, 5, 5}) {
		t.Fatal("solid brush did not block shot")
	}
	if !box.ClearShot(Vec3{-5, 20, 5}, Vec3{15, 20, 5}) {
		t.Fatal("shot outside brush was blocked")
	}
}
