package main

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"io"
	"net/http/httptest"
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestMapCacheChangesAndNewFiles(t *testing.T) {
	root := t.TempDir()
	s := &server{assets: []string{root}}
	h := s.handler()
	request := func(name, etag string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		r := httptest.NewRequest("GET", "/api/map?name="+name, nil)
		r.Header.Set("If-None-Match", etag)
		h.ServeHTTP(w, r)
		return w
	}
	if request("newmap", "").Code != 404 {
		t.Fatal("missing map accepted")
	}
	fixture := make([]byte, 160)
	copy(fixture, "IBSP")
	binary.LittleEndian.PutUint32(fixture[4:], 38)
	path := filepath.Join(root, "newmap.bsp")
	if e := os.WriteFile(path, fixture, 0600); e != nil {
		t.Fatal(e)
	}
	names := quake.OutlineMapNames(root)
	if len(names) != 1 || names[0] != "newmap" {
		t.Fatalf("new map not discovered: %v", names)
	}
	first := request("newmap", "")
	if first.Code != 200 || first.Header().Get("X-Map-Cache") != "MISS" {
		t.Fatalf("first=%d headers=%v", first.Code, first.Header())
	}
	second := request("newmap", "")
	if second.Header().Get("X-Map-Cache") != "HIT" || !bytes.Equal(first.Body.Bytes(), second.Body.Bytes()) {
		t.Fatal("cache not reused")
	}
	if request("newmap", first.Header().Get("ETag")).Code != 304 {
		t.Fatal("conditional request not cached")
	}
	svgRequest := httptest.NewRequest("GET", "/api/map.svg?name=newmap", nil)
	svgResponse := httptest.NewRecorder()
	h.ServeHTTP(svgResponse, svgRequest)
	if svgResponse.Code != 200 || svgResponse.Header().Get("Content-Type") != "image/svg+xml; charset=utf-8" || svgResponse.Header().Get("X-Map-Cache") != "HIT" {
		t.Fatalf("SVG cache response: %d %v", svgResponse.Code, svgResponse.Header())
	}
	svgRequest.Header.Set("If-None-Match", svgResponse.Header().Get("ETag"))
	svgConditional := httptest.NewRecorder()
	h.ServeHTTP(svgConditional, svgRequest)
	if svgConditional.Code != 304 {
		t.Fatal("SVG conditional cache failed")
	}
	stamp := time.Now().Add(2 * time.Second)
	os.Chtimes(path, stamp, stamp)
	if request("newmap", "").Header().Get("X-Map-Cache") != "MISS" {
		t.Fatal("file modification did not invalidate cache")
	}
	os.WriteFile(path, []byte("broken BSP"), 0600)
	if request("newmap", "").Code != 404 {
		t.Fatal("stale geometry served after corruption")
	}
	os.Remove(path)
	if request("newmap", "").Code != 404 {
		t.Fatal("removed map still served")
	}
	os.WriteFile(path, fixture, 0600)
	if request("newmap", "").Code != 200 {
		t.Fatal("reappearing map not loaded")
	}
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/map?name=newmap", nil)
	r.Header.Set("Accept-Encoding", "gzip")
	h.ServeHTTP(w, r)
	reader, e := gzip.NewReader(w.Body)
	if e != nil {
		t.Fatal(e)
	}
	body, e := io.ReadAll(reader)
	reader.Close()
	if e != nil || !bytes.Equal(body, first.Body.Bytes()) {
		t.Fatal("gzip representation differs")
	}
}
