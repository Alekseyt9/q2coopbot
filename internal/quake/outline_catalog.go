package quake

import (
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// OutlineMapNames reads loose filenames and small PAK directories, never full
// PAK payloads. Unavailable or incomplete assets are retried on the next scan.
func OutlineMapNames(root string) []string {
	names := make(map[string]bool)
	for _, dir := range []string{root, filepath.Join(root, "maps")} {
		entries, _ := os.ReadDir(dir)
		for _, entry := range entries {
			if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".bsp") {
				names[strings.TrimSuffix(entry.Name(), ".bsp")] = true
			}
		}
	}
	for _, pak := range []string{"pak2.pak", "pak1.pak", "pak0.pak"} {
		func() {
			f, e := os.Open(filepath.Join(root, pak))
			if e != nil {
				return
			}
			defer f.Close()
			header := make([]byte, 12)
			if _, e = io.ReadFull(f, header); e != nil || string(header[:4]) != "PACK" {
				return
			}
			offset := int64(binary.LittleEndian.Uint32(header[4:]))
			size := int64(binary.LittleEndian.Uint32(header[8:]))
			info, e := f.Stat()
			if e != nil || offset < 12 || size%64 != 0 || size > 16<<20 || offset+size > info.Size() {
				return
			}
			dir := make([]byte, size)
			if _, e = f.ReadAt(dir, offset); e != nil {
				return
			}
			for p := 0; p < len(dir); p += 64 {
				name := strings.TrimRight(string(dir[p:p+56]), "\x00")
				if strings.HasPrefix(name, "maps/") && strings.HasSuffix(name, ".bsp") {
					names[strings.TrimSuffix(strings.TrimPrefix(name, "maps/"), ".bsp")] = true
				}
			}
		}()
	}
	out := make([]string, 0, len(names))
	for name := range names {
		out = append(out, name)
	}
	return out
}
