package quake

import "testing"

func TestEntitySkinWidthsAndDeltaRetention(t *testing.T) {
	for _, item := range []struct {
		flags uint32
		data  []byte
		want  uint32
	}{
		{0x10000, []byte{7}, 7}, {0x2000000, []byte{0x34, 0x12}, 0x1234},
		{0x10000 | 0x2000000, []byte{0x78, 0x56, 0x34, 0x12}, 0x12345678},
		{0, nil, 99},
	} {
		r := &reader{data: append(append([]byte{}, item.data...), 42)}
		e, err := parseEntity(r, 1, item.flags, Entity{Skin: 99})
		if err != nil || e.Skin != item.want {
			t.Fatalf("skin %x: %+v %v", item.flags, e, err)
		}
		if sentinel, err := r.byte(); err != nil || sentinel != 42 {
			t.Fatal("skin byte alignment")
		}
		if len(item.data) > 0 {
			if _, err := parseEntity(&reader{data: item.data[:len(item.data)-1]}, 1, item.flags, Entity{}); err == nil {
				t.Fatal("truncated skin accepted")
			}
		}
	}
}
