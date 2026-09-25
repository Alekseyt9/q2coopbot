package quake

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadAASVersions(t *testing.T) {
	for _, version := range []uint32{3, 5} {
		t.Run(string(rune('0'+version)), func(t *testing.T) {
			headerSize := 120
			if version == 5 {
				headerSize = 124
			}
			data := make([]byte, headerSize+2*48+2*28+44)
			copy(data, "EAAS")
			binary.LittleEndian.PutUint32(data[4:], version)
			dir := headerSize - 14*8
			for _, lump := range []struct{ index, offset, length int }{
				{7, headerSize, 2 * 48},
				{8, headerSize + 2*48, 2 * 28},
				{9, headerSize + 2*48 + 2*28, 44},
			} {
				binary.LittleEndian.PutUint32(data[dir+lump.index*8:], uint32(lump.offset))
				binary.LittleEndian.PutUint32(data[dir+lump.index*8+4:], uint32(lump.length))
			}
			binary.LittleEndian.PutUint32(data[headerSize+2*48+28+20:], 1) // area 1 has one reach
			binary.LittleEndian.PutUint32(data[len(data)-44:], 1)          // destination area
			kind := uint32(2)
			if version == 5 {
				kind = 11
				binary.LittleEndian.PutUint32(data[len(data)-40:], 50)  // model *50
				binary.LittleEndian.PutUint32(data[len(data)-36:], 190) // rise
			}
			binary.LittleEndian.PutUint32(data[len(data)-8:], kind)
			if version == 5 {
				for i := 8; i < headerSize; i++ {
					data[i] ^= byte((i - 8) * 119)
				}
			}
			path := filepath.Join(t.TempDir(), "test.aas")
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			nav, err := LoadAAS(path)
			if err != nil {
				t.Fatal(err)
			}
			if len(nav.Areas) != 2 || len(nav.Edges[1]) != 1 || nav.Edges[1][0].Kind != int(kind) {
				t.Fatalf("unexpected AAS graph: areas=%d edges=%v", len(nav.Areas), nav.Edges[1])
			}
			if version == 5 && (nav.Edges[1][0].Model != 50 || nav.Edges[1][0].Rise != 190) {
				t.Fatalf("elevator metadata: %+v", nav.Edges[1][0])
			}
		})
	}
}
