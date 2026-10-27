// q2rocket-candidates prepares offline observed geometry options; no inference.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"q2coopbot/internal/rocketpractice"
)

func main() {
	request := flag.String("request", "", "Observed geometry request JSON")
	out := flag.String("out", "", "Fresh geometry diagnostic JSON")
	flag.Parse()
	if err := run(*request, *out); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(request, out string) error {
	if request == "" || out == "" {
		return fmt.Errorf("request and out required")
	}
	file, err := os.Open(request)
	if err != nil {
		return err
	}
	defer file.Close()
	var input struct {
		Version     string `json:"version"`
		RuntimeRoot string `json:"runtime_root"`
		Entries     []struct {
			ID          string             `json:"id"`
			Observation policy.Observation `json:"observation"`
		} `json:"entries"`
	}
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	if err = decoder.Decode(&input); err != nil {
		return err
	}
	var trailing any
	if err = decoder.Decode(&trailing); err != io.EOF {
		return fmt.Errorf("trailing request content")
	}
	if input.Version != "combat_observed_rocket_candidate_request_v1" || input.RuntimeRoot == "" || len(input.Entries) == 0 {
		return fmt.Errorf("invalid request")
	}
	type result struct {
		ID         string                     `json:"id"`
		Map        string                     `json:"map"`
		Frame      int                        `json:"frame"`
		BSPSHA256  string                     `json:"bsp_sha256"`
		Candidates []rocketpractice.Candidate `json:"candidates"`
	}
	rows := []result{}
	seen := map[string]bool{}
	worlds := map[string]*quake.MapInfo{}
	for _, e := range input.Entries {
		if e.ID == "" || seen[e.ID] {
			return fmt.Errorf("invalid entry id")
		}
		seen[e.ID] = true
		world := worlds[e.Observation.Identity.Map]
		if world == nil {
			loaded, err := quake.LoadMap(input.RuntimeRoot, e.Observation.Identity.Map)
			if err != nil {
				return err
			}
			world = &loaded
			worlds[world.Name] = world
		}
		c, err := rocketpractice.Generate(e.Observation, world)
		if err != nil {
			return err
		}
		rows = append(rows, result{e.ID, world.Name, e.Observation.Identity.Frame, world.BSPSHA256, c})
	}
	output, err := os.OpenFile(out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return err
	}
	defer output.Close()
	encoder := json.NewEncoder(output)
	encoder.SetIndent("", "  ")
	return encoder.Encode(struct {
		Version         string   `json:"version"`
		StaticOnly      bool     `json:"static_only"`
		PolicySelection bool     `json:"policy_selection"`
		Entries         []result `json:"entries"`
	}{rocketpractice.Version, true, false, rows})
}
