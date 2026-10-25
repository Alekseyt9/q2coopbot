// Offline geometry diagnostics for observed remembered positions. No NN or actions.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

type request struct {
	Map      string                    `json:"map"`
	Position quake.Vec3                `json:"position"`
	Threats  []policy.RememberedThreat `json:"remembered_threats"`
}

func finite(v quake.Vec3) bool {
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) || math.Abs(x) > 16384 {
			return false
		}
	}
	return true
}

func run(input, output, game string) error {
	f, err := os.Open(input)
	if err != nil {
		return err
	}
	defer f.Close()
	out, err := os.OpenFile(output, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer out.Close()
	enc := json.NewEncoder(out)
	scan := bufio.NewScanner(f)
	scan.Buffer(make([]byte, 65536), 4*1024*1024)
	maps := map[string]quake.MapInfo{}
	navs := map[string]*quake.Navigator{}
	for scan.Scan() {
		var r request
		if err = json.Unmarshal(scan.Bytes(), &r); err != nil {
			return err
		}
		if !regexp.MustCompile(`^[a-z][a-z0-9_]{0,31}$`).MatchString(r.Map) || !finite(r.Position) || len(r.Threats) > 8 {
			return fmt.Errorf("invalid observed request")
		}
		world, ok := maps[r.Map]
		if !ok {
			world, err = quake.LoadMap(game, r.Map)
			if err != nil {
				return err
			}
			maps[r.Map] = world
			navs[r.Map], err = quake.LoadAAS(filepath.Join(game, "maps", r.Map+".aas"))
			if err != nil {
				return err
			}
		}
		n := navs[r.Map]
		hints := []map[string]any{}
		for _, t := range r.Threats {
			if t.ID <= 0 || t.AgeFrames < 0 || t.AgeFrames > policy.ThreatMemoryFrames || !finite(t.Relative) {
				return fmt.Errorf("invalid remembered position")
			}
			goal := r.Position
			for i := range goal {
				goal[i] += t.Relative[i]
			}
			path, found := n.SearchRoute(r.Position, goal)
			h := map[string]any{"observed_id": t.ID, "class": t.Class, "memory_age_frames": t.AgeFrames, "last_observed_position": goal,
				"static_route_found": found, "static_direct_hull_clear": world.PlayerMoveClear(r.Position, goal), "waypoint_count": len(path)}
			if found {
				distance := 0.
				previous := r.Position
				var next *quake.Vec3
				for _, p := range path {
					distance += quake.Horizontal(previous, p.Position)
					previous = p.Position
					if next == nil && quake.Horizontal(r.Position, p.Position) > 16 {
						v := p.Position
						next = &v
					}
				}
				distance += quake.Horizontal(previous, goal)
				h["static_horizontal_path_length"] = distance
				if next == nil {
					v := goal
					next = &v
				}
				relative := *next
				for i := range relative {
					relative[i] -= r.Position[i]
				}
				h["next_static_waypoint_relative"] = relative
			}
			hints = append(hints, h)
		}
		if err = enc.Encode(map[string]any{"version": "combat_memory_static_route_audit_v1", "map": r.Map, "position": r.Position, "hints": hints,
			"scope": "Observed remembered endpoints and static BSP/AAS only. Unknown current target state; no dynamic door/monster clearance, native walkability guarantee, policy inputs, actions or NN computation."}); err != nil {
			return err
		}
	}
	return scan.Err()
}

func main() {
	input := flag.String("input", "", "observed request JSONL")
	out := flag.String("out", "", "fresh output JSONL")
	game := flag.String("game-dir", "", "existing baseq2 with BSP/AAS")
	flag.Parse()
	if *input == "" || *out == "" || *game == "" {
		fmt.Fprintln(os.Stderr, "input/out/game-dir required")
		os.Exit(1)
	}
	if err := run(*input, *out, *game); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
