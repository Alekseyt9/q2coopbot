// q2support-audit inspects recorded collision support; it runs no neural policy.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"q2coopbot/internal/quake"
)

func main() {
	request := flag.String("request", "", "Recorded motion request JSON")
	out := flag.String("out", "", "Fresh support diagnostic JSON")
	flag.Parse()
	if err := run(*request, *out); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(request, out string) error {
	if request == "" || out == "" {
		return fmt.Errorf("--request and --out required")
	}
	b, err := os.ReadFile(request)
	if err != nil {
		return err
	}
	var input struct {
		Version string `json:"version"`
		Entries []struct {
			ID       string         `json:"id"`
			Root     string         `json:"runtime_root"`
			Snapshot quake.Snapshot `json:"snapshot"`
		} `json:"entries"`
	}
	if err = json.Unmarshal(b, &input); err != nil {
		return err
	}
	if input.Version != "combat_recorded_motion_request_v1" || len(input.Entries) == 0 {
		return fmt.Errorf("invalid request")
	}
	type result struct {
		ID       string                   `json:"id"`
		OnGround bool                     `json:"snapshot_on_ground"`
		Audit    quake.GroundSupportAudit `json:"audit"`
	}
	results, seen := []result{}, map[string]bool{}
	for _, e := range input.Entries {
		if e.ID == "" || seen[e.ID] || e.Root == "" || e.Snapshot.Map == "" {
			return fmt.Errorf("invalid or duplicate entry")
		}
		seen[e.ID] = true
		g, err := quake.LoadMap(e.Root, e.Snapshot.Map)
		if err != nil {
			return err
		}
		results = append(results, result{e.ID, e.Snapshot.OnGround,
			g.InspectGroundSupport(e.Snapshot.Self, e.Snapshot.Ducked, e.Snapshot.Movers, e.Snapshot.CollisionBodies)})
	}
	file, err := os.OpenFile(out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return err
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(struct {
		Version         string   `json:"version"`
		Results         []result `json:"results"`
		NeuralExecution bool     `json:"neural_execution"`
		GuardChanged    bool     `json:"guard_changed"`
	}{Version: "combat_recorded_support_audit_v1", Results: results})
}
