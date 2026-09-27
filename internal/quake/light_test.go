package quake

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

func lightFixture() (map[int][]byte, []byte, []byte) {
	lumps := map[int][]byte{2: make([]byte, 36), 5: make([]byte, 76), 6: make([]byte, 20), 7: make([]byte, 12), 11: make([]byte, 12), 12: make([]byte, 12)}
	u32 := func(b []byte, at int, v uint32) { binary.LittleEndian.PutUint32(b[at:], v) }
	u16 := func(b []byte, at int, v uint16) { binary.LittleEndian.PutUint16(b[at:], v) }
	u32(lumps[2], 12, math.Float32bits(16))
	u32(lumps[2], 24, math.Float32bits(16))
	u32(lumps[2], 28, math.Float32bits(16))
	u32(lumps[5], 0, math.Float32bits(1))
	u32(lumps[5], 20, math.Float32bits(1))
	u16(lumps[6], 8, 3)
	copy(lumps[6][12:], []byte{0, 255, 255, 255})
	for i := 0; i < 3; i++ {
		u16(lumps[11], i*4, uint16(i))
		u16(lumps[11], i*4+2, uint16((i+1)%3))
		u32(lumps[12], i*4, uint32(i))
	}
	for i := range lumps[7] {
		lumps[7][i] = 170
	}
	nodes := make([]byte, 28)
	u32(nodes, 4, 0xffffffff)
	u32(nodes, 8, 0xffffffff)
	u16(nodes, 26, 1)
	return lumps, nodes, make([]byte, 48)
}

func TestStaticLightSamplingAndStyles(t *testing.T) {
	lumps, nodes, models := lightFixture()
	l, err := parseBSPLight(func(i, row int) ([]byte, error) { return lumps[i], nil }, []bspPlane{{normal: Vec3{0, 0, 1}}}, nodes, models)
	if err != nil {
		t.Fatal(err)
	}
	m := &MapInfo{lighting: l}
	for _, tc := range []struct {
		pattern string
		frame   int
		want    byte
	}{{"", 0, 100}, {"a", 0, 0}, {"m", 0, 100}, {"am", 0, 0}, {"am", 1, 100}, {"z", 0, 208}} {
		v, ok := m.StaticLightLevel(Vec3{8, 8, 24}, map[int]string{800: tc.pattern}, tc.frame)
		if !ok || v != tc.want {
			t.Fatalf("%+v: %d %v", tc, v, ok)
		}
	}
	if v, ok := m.StaticLightLevel(Vec3{40, 40, 24}, nil, 0); !ok || v != 0 {
		t.Fatal("no surface must be dark")
	}
	if _, ok := (*MapInfo)(nil).StaticLightLevel(Vec3{}, nil, 0); ok {
		t.Fatal("missing map claimed valid light")
	}
	l.data = nil
	if v, ok := m.StaticLightLevel(Vec3{}, nil, 0); !ok || v != 150 {
		t.Fatal("unlit BSP must use renderer fullbright fallback")
	}
}

func TestRejectMalformedLightData(t *testing.T) {
	for _, tc := range []string{"offset", "samples", "edge", "tex", "cycle", "plane"} {
		t.Run(tc, func(t *testing.T) {
			lumps, nodes, models := lightFixture()
			switch tc {
			case "offset":
				binary.LittleEndian.PutUint32(lumps[6][16:], 1000)
			case "samples":
				lumps[7] = lumps[7][:3]
			case "edge":
				binary.LittleEndian.PutUint32(lumps[12], 1000)
			case "tex":
				binary.LittleEndian.PutUint16(lumps[6][10:], 1000)
			case "cycle":
				binary.LittleEndian.PutUint32(nodes[4:], 0)
			case "plane":
				binary.LittleEndian.PutUint32(nodes, 1000)
			}
			if _, err := parseBSPLight(func(i, row int) ([]byte, error) { return lumps[i], nil }, []bspPlane{{normal: Vec3{0, 0, 1}}}, nodes, models); err == nil {
				t.Fatal("malformed lighting accepted")
			}
		})
	}
}

func TestBase1StaticLighting(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("base1 BSP required")
	}
	m, err := LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	if m.LightingError != "" {
		t.Fatal(m.LightingError)
	}
	for _, p := range []Vec3{{608, 192, -9.875}, {1184, 192, -9.875}, {32, -224, 46}} {
		v, ok := m.StaticLightLevel(p, nil, 0)
		if !ok || v <= 5 {
			t.Fatalf("lit fixture %v got %d %v", p, v, ok)
		}
		t.Logf("%v light=%d", p, v)
	}
}

func TestLightStylesFollowProtocolUpdatesAndReset(t *testing.T) {
	lumps, nodes, models := lightFixture()
	l, err := parseBSPLight(func(i, row int) ([]byte, error) { return lumps[i], nil }, []bspPlane{{normal: Vec3{0, 0, 1}}}, nodes, models)
	if err != nil {
		t.Fatal(err)
	}
	m, d := &MapInfo{lighting: l}, NewDecoder()
	set := func(pattern string) {
		packet := binary.LittleEndian.AppendUint16([]byte{13}, 800)
		packet = append(packet, []byte(pattern)...)
		packet = append(packet, 0)
		if _, err := d.Parse(packet); err != nil {
			t.Fatal(err)
		}
	}
	check := func(frame int, want byte) {
		t.Helper()
		got, ok := m.StaticLightLevel(Vec3{8, 8, 24}, d.Config, frame)
		if !ok || got != want {
			t.Fatalf("frame=%d got=%d want=%d", frame, got, want)
		}
	}
	set("m")
	check(10, 100)
	set("a")
	check(10, 0) // Update at the same frame must not use a cached value.
	set("am")
	check(10, 0)
	check(11, 100)
	packet := binary.LittleEndian.AppendUint32([]byte{12}, 34)
	packet = binary.LittleEndian.AppendUint32(packet, 2)
	packet = append(packet, 0)
	packet = append(packet, []byte("baseq2\x00")...)
	packet = binary.LittleEndian.AppendUint16(packet, 0)
	packet = append(packet, []byte("next map\x00")...)
	if _, err := d.Parse(packet); err != nil {
		t.Fatal(err)
	}
	if _, exists := d.Config[800]; exists {
		t.Fatal("previous map lightstyle survived serverdata")
	}
	set("a")
	check(1, 0)
}
