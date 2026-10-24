// q2motion-audit replays recorded geometric diagnostics, never a neural policy.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"reflect"

	"q2coopbot/internal/bot"
	"q2coopbot/internal/quake"
)

type entry struct {
	ID       string                     `json:"id"`
	Root     string                     `json:"runtime_root"`
	Snapshot quake.Snapshot             `json:"snapshot"`
	Proposed quake.UserCmd              `json:"proposed"`
	Applied  quake.UserCmd              `json:"applied"`
	Logged   *bot.PmoveGroundPrediction `json:"logged_applied_prediction"`
	Surface  string                     `json:"logged_ground_surface"`
}

func main() {
	request := flag.String("request", "", "Frozen observed-state request JSON")
	output := flag.String("out", "", "Fresh diagnostic report JSON")
	flag.Parse()
	if err := run(*request, *output); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(request, output string) error {
	if request == "" || output == "" {
		return fmt.Errorf("--request and --out required")
	}
	b, err := os.ReadFile(request)
	if err != nil {
		return err
	}
	var input struct {
		Version string  `json:"version"`
		Entries []entry `json:"entries"`
	}
	if err = json.Unmarshal(b, &input); err != nil {
		return err
	}
	if input.Version != "combat_recorded_motion_request_v1" || len(input.Entries) == 0 {
		return fmt.Errorf("unsupported or empty request")
	}
	type result struct {
		ID             string                  `json:"id"`
		AppliedExact   bool                    `json:"applied_prediction_exact"`
		SurfaceExact   bool                    `json:"ground_surface_exact"`
		AppliedPresent bool                    `json:"applied_prediction_present"`
		Audit          bot.RecordedMotionAudit `json:"audit"`
	}
	results := make([]result, 0, len(input.Entries))
	seen := map[string]bool{}
	geometry := map[string]*quake.MapInfo{}
	for _, e := range input.Entries {
		if e.ID == "" || seen[e.ID] || e.Root == "" || e.Snapshot.Map == "" || e.Snapshot.Health <= 0 {
			return fmt.Errorf("invalid or duplicated entry %q", e.ID)
		}
		seen[e.ID] = true
		key := e.Root + "\x00" + e.Snapshot.Map
		g := geometry[key]
		if g == nil {
			loaded, loadErr := quake.LoadMap(e.Root, e.Snapshot.Map)
			if loadErr != nil {
				return fmt.Errorf("entry %s: %w", e.ID, loadErr)
			}
			g = &loaded
			geometry[key] = g
		}
		audit := bot.DiagnoseRecordedCombatMotion(e.Snapshot, e.Proposed, e.Applied, g)
		results = append(results, result{ID: e.ID, AppliedExact: reflect.DeepEqual(audit.Applied, e.Logged),
			SurfaceExact: audit.GroundSurface == e.Surface, AppliedPresent: audit.Applied != nil, Audit: audit})
	}
	b, err = json.MarshalIndent(struct {
		Version         string   `json:"version"`
		Results         []result `json:"results"`
		NeuralExecution bool     `json:"neural_execution"`
		GuardChanged    bool     `json:"guard_changed"`
	}{Version: "combat_recorded_motion_geometry_v1", Results: results}, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(b)
	return err
}
