package main

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"regexp"
	"strings"
	"sync"
	"time"
)

var mapNamePattern = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

type cachedMap struct {
	mu                 sync.Mutex
	signature          string
	body, compressed   []byte
	etag               string
	svg, svgCompressed []byte
	svgETag            string
}

// Stat loose BSPs and PAKs without opening their contents. This also detects
// a new overriding BSP or a previously missing file on the very next request.
func (s *server) mapSignature(name string) string {
	var signature strings.Builder
	for _, root := range s.assets {
		for _, rel := range []string{filepath.Join("maps", name+".bsp"), name + ".bsp", "pak2.pak", "pak1.pak", "pak0.pak"} {
			path := filepath.Join(root, rel)
			if info, e := os.Stat(path); e == nil {
				fmt.Fprintf(&signature, "%s:%d:%d;", path, info.Size(), info.ModTime().UnixNano())
			}
		}
	}
	return signature.String()
}

// Each map has its own lock: simultaneous requests share one parse and encode,
// while different maps can load independently. Failed loads are never cached.
func (s *server) mapData(name string, asSVG bool) (body, compressed []byte, etag string, hit bool, err error) {
	if !mapNamePattern.MatchString(name) {
		return nil, nil, "", false, fmt.Errorf("invalid map name")
	}
	s.mapMu.Lock()
	if s.mapCache == nil {
		s.mapCache = make(map[string]*cachedMap)
	}
	entry := s.mapCache[name]
	if entry == nil {
		entry = &cachedMap{}
		s.mapCache[name] = entry
	}
	s.mapMu.Unlock()
	entry.mu.Lock()
	defer entry.mu.Unlock()
	signature := s.mapSignature(name)
	if entry.body != nil && signature == entry.signature {
		if asSVG {
			return entry.svg, entry.svgCompressed, entry.svgETag, true, nil
		}
		return entry.body, entry.compressed, entry.etag, true, nil
	}
	for _, root := range s.assets {
		outline, e := quake.LoadMapOutline(root, name)
		if e != nil {
			continue
		}
		body, e = json.Marshal(outline)
		if e != nil {
			return nil, nil, "", false, e
		}
		var zipped bytes.Buffer
		writer := gzip.NewWriter(&zipped)
		if _, e = writer.Write(body); e != nil {
			return nil, nil, "", false, e
		}
		if e = writer.Close(); e != nil {
			return nil, nil, "", false, e
		}
		entry.signature = signature
		entry.body = body
		entry.compressed = zipped.Bytes()
		entry.etag = fmt.Sprintf(`W/"%x"`, sha256.Sum256(body))
		entry.svg = outlineSVG(outline)
		var svgZip bytes.Buffer
		svgWriter := gzip.NewWriter(&svgZip)
		svgWriter.Write(entry.svg)
		svgWriter.Close()
		entry.svgCompressed = svgZip.Bytes()
		entry.svgETag = fmt.Sprintf(`W/"%x"`, sha256.Sum256(entry.svg))
		if asSVG {
			return entry.svg, entry.svgCompressed, entry.svgETag, false, nil
		}
		return entry.body, entry.compressed, entry.etag, false, nil
	}
	return nil, nil, "", false, fmt.Errorf("BSP не найден или повреждён; трейс доступен без схемы")
}

func (s *server) serveMap(w http.ResponseWriter, r *http.Request) {
	asSVG := strings.HasSuffix(r.URL.Path, ".svg")
	body, zipped, etag, hit, e := s.mapData(r.URL.Query().Get("name"), asSVG)
	if e != nil {
		http.Error(w, e.Error(), 404)
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	if asSVG {
		w.Header().Set("Content-Type", "image/svg+xml; charset=utf-8")
	}
	w.Header().Set("Cache-Control", "public, no-cache")
	w.Header().Set("ETag", etag)
	w.Header().Set("Vary", "Accept-Encoding")
	if hit {
		w.Header().Set("X-Map-Cache", "HIT")
	} else {
		w.Header().Set("X-Map-Cache", "MISS")
	}
	for _, tag := range strings.Split(r.Header.Get("If-None-Match"), ",") {
		if strings.TrimSpace(tag) == etag || strings.TrimSpace(tag) == "*" {
			w.WriteHeader(http.StatusNotModified)
			return
		}
	}
	if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
		w.Header().Set("Content-Encoding", "gzip")
		body = zipped
	}
	w.Write(body)
}

// Warm existing maps and discover newly added BSP/PAK contents every 10 seconds.
func (s *server) warmMaps() {
	for {
		names := make(map[string]bool)
		for _, root := range s.assets {
			for _, name := range quake.OutlineMapNames(root) {
				names[name] = true
			}
		}
		for name := range names {
			s.mapData(name, false)
		}
		time.Sleep(10 * time.Second)
	}
}
