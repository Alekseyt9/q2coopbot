package bot

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"q2coopbot/internal/quake"
)

// Opt-in replay of native sensor/command receipts; no neural inference. Keep
// the game assets and untracked receipts outside the repository's test data.
func TestCombatNativeLedgeReceiptDiagnostic(t *testing.T) {
	root, receipt := os.Getenv("Q2_SEARCH_SCAN_ROOT"), os.Getenv("Q2_LEDGE_RECEIPT")
	if root == "" || receipt == "" {
		t.Skip("requires local BSP/AAS and a native ledge receipt")
	}
	var fixture struct {
		Version string `json:"version"`
		Records []struct {
			Frame           int            `json:"frame"`
			Snapshot        quake.Snapshot `json:"snapshot"`
			Command         quake.UserCmd  `json:"command"`
			ExpectedGuarded *quake.UserCmd `json:"expected_guarded,omitempty"`
			ExpectedSupport *bool          `json:"expected_aas_support_next,omitempty"`
			Next            struct {
				Position quake.Vec3 `json:"position"`
				Velocity quake.Vec3 `json:"velocity"`
				OnGround bool       `json:"on_ground"`
			} `json:"next_observation"`
		} `json:"records"`
	}
	data, err := os.ReadFile(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Version != "combat_native_ledge_physics_receipt_v1" || len(fixture.Records) == 0 {
		t.Fatal("invalid receipt")
	}
	geometry, err := quake.LoadMap(root, fixture.Records[0].Snapshot.Map)
	if err != nil {
		t.Fatal(err)
	}
	nav, err := quake.LoadAAS(filepath.Join(root, "maps", fixture.Records[0].Snapshot.Map+".aas"))
	if err != nil {
		t.Fatal(err)
	}
	p := Planner{World: World{Geometry: &geometry}, Nav: nav}
	records := []map[string]any{}
	for _, row := range fixture.Records {
		s, command := row.Snapshot, row.Command
		guarded, changes := p.guardDirectCombatMotion(s, command, true)
		if row.ExpectedGuarded != nil && guarded != *row.ExpectedGuarded {
			t.Fatalf("frame %d: unexpected guarded command: got %+v want %+v", row.Frame, guarded, *row.ExpectedGuarded)
		}
		if row.ExpectedSupport != nil && nav.GroundedNear(row.Next.Position) != *row.ExpectedSupport {
			t.Fatalf("frame %d: native support classification changed", row.Frame)
		}
		drop, supported := geometry.GroundDrop(row.Next.Position, 24)
		r := map[string]any{"frame": row.Frame, "position": s.Self, "velocity": s.SelfVelocity,
			"command": command, "guarded": guarded, "changes": changes,
			"native_next": row.Next.Position, "native_grounded_next": row.Next.OnGround,
			"bsp_support_at_native_next": supported, "bsp_drop_at_native_next": nil,
			"aas_grounded_near_native_next": nav.GroundedNear(row.Next.Position),
			"aas_exact_area_native_next":    nav.ExactAreaFor(row.Next.Position)}
		if supported {
			r["bsp_drop_at_native_next"] = drop
		}
		if area := nav.ExactAreaFor(row.Next.Position); area > 0 {
			r["aas_exact_area"] = nav.Areas[area]
		}
		near := []int{}
		for id, area := range nav.Areas {
			if id == 0 || area.Flags&1 == 0 || row.Next.Position[2]-area.Min[2] > 24 {
				continue
			}
			outside := 0.0
			for axis, value := range row.Next.Position {
				d := math.Max(area.Min[axis]-value, math.Max(0, value-area.Max[axis]))
				outside += d * d
			}
			if outside <= 4 {
				near = append(near, id)
			}
		}
		r["aas_supporting_aabb_areas"] = near
		if simple := predictCombatGroundStep(s, command); simple != nil {
			end := s.Self
			for axis := range end {
				end[axis] += simple.Displacement[axis]
			}
			_, supported := geometry.GroundDrop(end, 24)
			r["simple_prediction"], r["simple_end"] = simple, end
			r["simple_end_bsp_support"] = supported
			r["simple_end_aas_support"] = nav.GroundedNear(end)
			r["simple_end_hazard"] = geometry.GroundMoveHazardStep(nav, s.Self, simple.Displacement[0], simple.Displacement[1], quake.Horizontal(s.Self, end))
		}
		if detailed := predictPmoveGround(s, command, &geometry); detailed != nil {
			r["detailed_prediction"] = detailed
			r["detailed_paths_safe"] = p.checkedCombatMotionPaths(s, command, detailed)
		}
		records = append(records, r)
	}
	encoded, err := json.MarshalIndent(map[string]any{"version": "combat_native_ledge_receipt_diagnostic_v1", "records": records, "neural_execution": false}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if output := os.Getenv("Q2_LEDGE_REPORT"); output != "" {
		f, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = f.Write(encoded); err != nil {
			f.Close()
			t.Fatal(err)
		}
		if err = f.Close(); err != nil {
			t.Fatal(err)
		}
	} else {
		t.Log(string(encoded))
	}
}
