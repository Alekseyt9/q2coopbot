package main

import (
	"bufio"
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

//go:embed web/*
var web embed.FS

type point struct {
	Map          string          `json:"map"`
	Frame        int             `json:"frame"`
	Connection   int             `json:"connection"`
	Spawncount   int             `json:"spawncount"`
	Self         *quake.Vec3     `json:"self"`
	Teammate     *quake.Vec3     `json:"teammate,omitempty"`
	Health       int             `json:"health"`
	Armor        *int            `json:"armor"`
	Ammo         *int            `json:"ammo"`
	Weapon       string          `json:"weapon"`
	SelfVelocity *quake.Vec3     `json:"self_velocity,omitempty"`
	OnGround     *bool           `json:"on_ground,omitempty"`
	Ducked       *bool           `json:"ducked,omitempty"`
	WeaponReason string          `json:"weapon_reason,omitempty"`
	Goal         string          `json:"goal"`
	Arbitration  json.RawMessage `json:"arbitration,omitempty"`
}
type run struct {
	ID       string    `json:"id"`
	Map      string    `json:"map"`
	Modified time.Time `json:"modified"`
	Bytes    int64     `json:"bytes"`
}
type server struct {
	root     string
	assets   []string
	mu       sync.Mutex
	runs     []run
	scanned  time.Time
	mapMu    sync.Mutex
	mapCache map[string]*cachedMap
}

// Ignore non-telemetry JSONL and tolerate an unfinished final line of a live trace.
func readTrace(path string, first bool) ([]point, int, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, 0, e
	}
	defer f.Close()
	scanner := bufio.NewScanner(io.LimitReader(f, 256<<20))
	scanner.Buffer(make([]byte, 65536), 8<<20)
	rows := make([]point, 0)
	skipped := 0
	lines := 0
	for scanner.Scan() {
		lines++
		if first && lines > 8 {
			break
		}
		var p point
		if json.Unmarshal(scanner.Bytes(), &p) != nil {
			skipped++
			continue
		}
		if p.Self == nil || p.Map == "" {
			continue
		}
		rows = append(rows, p)
		if first {
			break
		}
	}
	return rows, skipped, scanner.Err()
}
func (s *server) catalog() ([]run, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if time.Since(s.scanned) < 10*time.Second {
		return s.runs, nil
	}
	runs := make([]run, 0)
	e := filepath.WalkDir(s.root, func(path string, d fs.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if d.IsDir() || !strings.HasSuffix(d.Name(), ".jsonl") {
			return nil
		}
		rows, _, e := readTrace(path, true)
		if e != nil || len(rows) == 0 {
			return nil
		}
		info, e := d.Info()
		if e != nil {
			return nil
		}
		rel, _ := filepath.Rel(s.root, path)
		runs = append(runs, run{filepath.ToSlash(rel), rows[0].Map, info.ModTime(), info.Size()})
		return nil
	})
	if e != nil {
		return nil, e
	}
	sort.Slice(runs, func(i, j int) bool {
		if runs[i].Modified.Equal(runs[j].Modified) {
			return runs[i].ID < runs[j].ID
		}
		return runs[i].Modified.After(runs[j].Modified)
	})
	s.runs = runs
	s.scanned = time.Now()
	return runs, nil
}
func respond(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(v)
}
func (s *server) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/runs", func(w http.ResponseWriter, r *http.Request) {
		runs, e := s.catalog()
		if e != nil {
			http.Error(w, e.Error(), 500)
			return
		}
		page, _ := strconv.Atoi(r.URL.Query().Get("page"))
		if page < 1 {
			page = 1
		}
		size, _ := strconv.Atoi(r.URL.Query().Get("size"))
		if size < 1 {
			size = 25
		}
		if size > 100 {
			size = 100
		}
		q := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("q")))
		mapFilter := r.URL.Query().Get("map")
		mapSet := make(map[string]bool)
		filtered := make([]run, 0)
		for _, run := range runs {
			mapSet[run.Map] = true
			if (mapFilter == "" || run.Map == mapFilter) && strings.Contains(strings.ToLower(run.ID+" "+run.Map), q) {
				filtered = append(filtered, run)
			}
		}
		total := len(filtered)
		pages := (total + size - 1) / size
		if pages < 1 {
			pages = 1
		}
		if page > pages {
			page = pages
		}
		start := (page - 1) * size
		end := start + size
		if end > total {
			end = total
		}
		maps := make([]string, 0, len(mapSet))
		for name := range mapSet {
			maps = append(maps, name)
		}
		sort.Strings(maps)
		respond(w, map[string]any{"items": filtered[start:end], "page": page, "pages": pages, "size": size, "total": total, "maps": maps})
	})
	mux.HandleFunc("GET /api/trace", func(w http.ResponseWriter, r *http.Request) {
		id := r.URL.Query().Get("id")
		runs, e := s.catalog()
		if e != nil {
			http.Error(w, e.Error(), 500)
			return
		}
		found := false
		for _, run := range runs {
			if run.ID == id {
				found = true
				break
			}
		}
		if !found {
			http.Error(w, "Trace not found", 404)
			return
		}
		path := filepath.Join(s.root, filepath.FromSlash(id))
		info, e := os.Stat(path)
		if e != nil {
			http.Error(w, e.Error(), 404)
			return
		}
		if info.Size() > 256<<20 {
			http.Error(w, "Trace exceeds 256 MiB", 413)
			return
		}
		rows, skipped, e := readTrace(path, false)
		if e != nil {
			http.Error(w, e.Error(), 500)
			return
		}
		respond(w, map[string]any{"points": rows, "skipped": skipped})
	})
	mux.HandleFunc("GET /api/map", func(w http.ResponseWriter, r *http.Request) {
		s.serveMap(w, r)
	})
	mux.HandleFunc("GET /api/map.svg", s.serveMap)
	files, _ := fs.Sub(web, "web")
	mux.Handle("/", http.FileServer(http.FS(files)))
	return mux
}
func main() {
	addr := flag.String("listen", "127.0.0.1:18791", "HTTP address")
	root := flag.String("runs", "workspace/artifacts", "JSONL directory")
	assets := flag.String("assets", "../assets/baseq2;workspace/runtime/live-coop-20260927-214416/baseq2;../reference/aas-work/q2-campaign-20260920/bsp", "semicolon-separated BSP/PAK directories")
	flag.Parse()
	abs, e := filepath.Abs(*root)
	if e != nil {
		log.Fatal(e)
	}
	s := &server{root: abs, assets: strings.Split(*assets, ";")}
	go s.warmMaps()
	fmt.Printf("Quake II Run Explorer: http://%s\nTraces: %s\n", *addr, abs)
	srv := http.Server{Addr: *addr, Handler: s.handler(), ReadHeaderTimeout: 5 * time.Second, IdleTimeout: 60 * time.Second}
	log.Fatal(srv.ListenAndServe())
}
