package main

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"sort"
	"strings"
)

type demoEntity struct {
	ID     int        `json:"id"`
	Model  string     `json:"model"`
	Origin quake.Vec3 `json:"origin"`
	Angles quake.Vec3 `json:"angles"`
	Frame  int        `json:"frame"`
	Solid  uint16     `json:"solid"`
}
type demoFrame struct {
	Map      string       `json:"map"`
	Frame    int          `json:"frame"`
	Self     quake.Vec3   `json:"self"`
	Eye      quake.Vec3   `json:"eye"`
	Angles   quake.Vec3   `json:"angles"`
	FOV      float64      `json:"fov"`
	Health   int16        `json:"health"`
	Armor    int16        `json:"armor"`
	Ammo     int16        `json:"ammo"`
	Weapon   string       `json:"weapon"`
	GunFrame int          `json:"gun_frame"`
	Entities []demoEntity `json:"entities"`
}
type cachedDemo struct {
	size         int64
	modified     int64
	body, zipped []byte
	etag         string
	maps         []string
	frames       int
	complete     bool
}

func readDemo(path string) ([]demoFrame, bool, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, false, err
	}
	defer file.Close()
	decoder := quake.NewDecoder()
	frames := make([]demoFrame, 0)
	var total int64
	for {
		var header [4]byte
		_, err := io.ReadFull(file, header[:])
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return frames, false, nil
		}
		if err != nil {
			return nil, false, err
		}
		n := int(int32(binary.LittleEndian.Uint32(header[:])))
		if n == -1 {
			return frames, true, nil
		}
		if n < 0 || n > 65535 {
			return nil, false, fmt.Errorf("invalid DM2 message length %d", n)
		}
		total += int64(n) + 4
		if total > 256<<20 || len(frames) > 100000 {
			return nil, false, fmt.Errorf("demo exceeds playback limit")
		}
		packet := make([]byte, n)
		if _, err := io.ReadFull(file, packet); err == io.EOF || err == io.ErrUnexpectedEOF {
			return frames, false, nil
		} else if err != nil {
			return nil, false, err
		}
		decoded, err := decoder.Parse(packet)
		if err != nil {
			return nil, false, fmt.Errorf("DM2 packet: %w", err)
		}
		for _, f := range decoded {
			frame := demoFrame{Map: decoder.Map, Frame: f.Number, Self: f.Origin, Angles: quake.Vec3{}, FOV: f.FOV, Health: f.Stats[1], Armor: f.Stats[5], Ammo: f.Stats[3], GunFrame: f.GunFrame, Weapon: decoder.Config[32+f.Gun], Entities: make([]demoEntity, 0, len(f.Entities))}
			for axis := range frame.Eye {
				frame.Eye[axis] = f.Origin[axis] + f.ViewOffset[axis]
				frame.Angles[axis] = float64(f.ViewAngles[axis]) * 360 / 65536
			}
			if frame.FOV == 0 {
				frame.FOV = 90
			}
			ids := make([]int, 0, len(f.Entities))
			for id := range f.Entities {
				ids = append(ids, id)
			}
			sort.Ints(ids)
			for _, id := range ids {
				e := f.Entities[id]
				if id == decoder.PlayerNumber || e.Model == 0 {
					continue
				}
				frame.Entities = append(frame.Entities, demoEntity{id, decoder.Config[32+e.Model], e.Origin, e.Angles, e.Frame, e.Solid})
			}
			frames = append(frames, frame)
		}
	}
}

func zippedJSON(value any) ([]byte, []byte, string, error) {
	body, err := json.Marshal(value)
	if err != nil {
		return nil, nil, "", err
	}
	var compressed bytes.Buffer
	w := gzip.NewWriter(&compressed)
	if _, err = w.Write(body); err != nil {
		return nil, nil, "", err
	}
	if err = w.Close(); err != nil {
		return nil, nil, "", err
	}
	return body, compressed.Bytes(), fmt.Sprintf(`W/"%x"`, sha256.Sum256(body)), nil
}
func serveCached(w http.ResponseWriter, r *http.Request, body, zipped []byte, etag string) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "private, no-cache")
	w.Header().Set("ETag", etag)
	w.Header().Set("Vary", "Accept-Encoding")
	if r.Header.Get("If-None-Match") == etag {
		w.WriteHeader(304)
		return
	}
	if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
		w.Header().Set("Content-Encoding", "gzip")
		body = zipped
	}
	w.Write(body)
}

// Only demos next to an actual catalogued trace are reachable. Resolve symlinks
// too, so a file inside artifacts cannot redirect the API outside that tree.
func (s *server) demoPaths(id string) ([]string, error) {
	runs, err := s.catalog()
	if err != nil {
		return nil, err
	}
	found := false
	for _, r := range runs {
		if r.ID == id {
			found = true
			break
		}
	}
	if !found {
		return nil, os.ErrNotExist
	}
	trace := filepath.Join(s.root, filepath.FromSlash(id))
	stem := strings.TrimSuffix(filepath.Base(trace), filepath.Ext(trace))
	entries, err := os.ReadDir(filepath.Dir(trace))
	if err != nil {
		return nil, err
	}
	root, err := filepath.EvalSymlinks(s.root)
	if err != nil {
		return nil, err
	}
	paths := []string{}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !(name == stem+".dm2" || strings.HasPrefix(name, stem+"-") && strings.HasSuffix(name, ".dm2")) {
			continue
		}
		path, err := filepath.EvalSymlinks(filepath.Join(filepath.Dir(trace), name))
		if err != nil {
			continue
		}
		rel, err := filepath.Rel(root, path)
		if err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			paths = append(paths, path)
		}
	}
	sort.Strings(paths)
	return paths, nil
}
func (s *server) demoData(path string) (*cachedDemo, error) {
	s.demoMu.Lock()
	defer s.demoMu.Unlock()
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if info.Size() > 256<<20 {
		return nil, fmt.Errorf("demo exceeds 256 MiB")
	}
	if s.demoCache == nil {
		s.demoCache = map[string]*cachedDemo{}
	}
	entry := s.demoCache[path]
	if entry != nil && entry.size == info.Size() && entry.modified == info.ModTime().UnixNano() {
		return entry, nil
	}
	frames, complete, err := readDemo(path)
	if err != nil {
		return nil, err
	}
	maps := []string{}
	for _, f := range frames {
		if len(maps) == 0 || maps[len(maps)-1] != f.Map {
			maps = append(maps, f.Map)
		}
	}
	body, zipped, etag, err := zippedJSON(map[string]any{"frames": frames, "complete": complete, "maps": maps})
	if err != nil {
		return nil, err
	}
	entry = &cachedDemo{info.Size(), info.ModTime().UnixNano(), body, zipped, etag, maps, len(frames), complete}
	s.demoCache[path] = entry
	return entry, nil
}
func (s *server) serveDemos(w http.ResponseWriter, r *http.Request) {
	paths, err := s.demoPaths(r.URL.Query().Get("id"))
	if err != nil {
		http.Error(w, "Trace not found", 404)
		return
	}
	items := []map[string]any{}
	for _, path := range paths {
		entry, err := s.demoData(path)
		if err != nil {
			items = append(items, map[string]any{"name": filepath.Base(path), "error": err.Error()})
			continue
		}
		items = append(items, map[string]any{"name": filepath.Base(path), "maps": entry.maps, "frames": entry.frames, "complete": entry.complete})
	}
	respond(w, map[string]any{"items": items})
}
func (s *server) serveDemo(w http.ResponseWriter, r *http.Request) {
	paths, err := s.demoPaths(r.URL.Query().Get("id"))
	if err != nil {
		http.Error(w, "Trace not found", 404)
		return
	}
	for _, path := range paths {
		if filepath.Base(path) != r.URL.Query().Get("name") {
			continue
		}
		entry, err := s.demoData(path)
		if err != nil {
			http.Error(w, err.Error(), 422)
			return
		}
		serveCached(w, r, entry.body, entry.zipped, entry.etag)
		return
	}
	http.Error(w, "Demo not found", 404)
}
