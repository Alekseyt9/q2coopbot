package main

import (
	"fmt"
	"net/http"
	"q2coopbot/internal/quake"
)

func (s *server) serveMesh(w http.ResponseWriter, r *http.Request) {
	name := r.URL.Query().Get("name")
	if !mapNamePattern.MatchString(name) {
		http.Error(w, "invalid map name", 400)
		return
	}
	s.mapMu.Lock()
	if s.mapCache == nil {
		s.mapCache = map[string]*cachedMap{}
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
	hit := entry.mesh != nil && entry.meshSignature == signature
	if !hit {
		var geometry quake.MapOutline
		var err error
		found := false
		for _, root := range s.assets {
			geometry, err = quake.LoadMapGeometry(root, name)
			if err == nil {
				found = true
				break
			}
		}
		if !found {
			http.Error(w, fmt.Sprintf("BSP geometry unavailable: %v", err), 404)
			return
		}
		entry.mesh, entry.meshCompressed, entry.meshETag, err = zippedJSON(map[string]any{"name": name, "polygons": geometry.Floors, "models": geometry.FaceModels})
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		entry.meshSignature = signature
	}
	if hit {
		w.Header().Set("X-Map-Cache", "HIT")
	} else {
		w.Header().Set("X-Map-Cache", "MISS")
	}
	serveCached(w, r, entry.mesh, entry.meshCompressed, entry.meshETag)
}
