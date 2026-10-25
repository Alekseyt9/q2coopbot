// Offline observation enrichment and feature encoding. Never evaluates a neural net.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"

	"q2coopbot/internal/bot"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func run(input, output, game string) error {
	in, err := os.Open(input)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(output, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer out.Close()
	scan := bufio.NewScanner(in)
	scan.Buffer(make([]byte, 65536), 8*1024*1024)
	enc := json.NewEncoder(out)
	var observer bot.MemoryRouteObserver
	var geometry quake.MapInfo
	var nav *quake.Navigator
	mapName := ""
	valid := regexp.MustCompile(`^[a-z][a-z0-9_]{0,31}$`)
	for scan.Scan() {
		var o policy.Observation
		if err := json.Unmarshal(scan.Bytes(), &o); err != nil {
			return err
		}
		if !valid.MatchString(o.Identity.Map) {
			return fmt.Errorf("invalid observation map")
		}
		if o.Identity.Map != mapName {
			geometry, err = quake.LoadMap(game, o.Identity.Map)
			if err != nil {
				return err
			}
			nav, err = quake.LoadAAS(filepath.Join(game, "maps", o.Identity.Map+".aas"))
			if err != nil {
				return err
			}
			mapName = o.Identity.Map
		}
		observer.Enrich(&o, nav, &geometry)
		x, err := policy.FeaturesForVersion(o, policy.MemoryRouteFeatureVersion)
		if err != nil {
			return err
		}
		searches, hits := observer.Statistics()
		if err := enc.Encode(map[string]any{"observation": o, "features": x, "searches": searches, "cache_hits": hits}); err != nil {
			return err
		}
	}
	return scan.Err()
}

func main() {
	input := flag.String("input", "", "observation JSONL")
	output := flag.String("output", "", "new output JSONL")
	game := flag.String("game-dir", "", "shared baseq2 assets")
	flag.Parse()
	if *input == "" || *output == "" || *game == "" {
		fmt.Fprintln(os.Stderr, "input, output and game-dir required")
		os.Exit(2)
	}
	if err := run(*input, *output, *game); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
