package bot

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestCampaignUnitMapsConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "unit.json")
	for _, data := range []string{
		`{"run":{"mode":"companion","campaign_unit_maps":["unit_b"]}}`,
		`{"run":{"mode":"campaign","campaign_unit_maps":["unit_b"]}}`,
		`{"run":{"mode":"campaign","campaign_route":["unit_a","unit_c"],"campaign_unit_maps":["../unit_b"]}}`,
		`{"run":{"mode":"campaign","campaign_route":["unit_a","unit_c"],"campaign_unit_maps":["unit_b","unit_b"]}}`,
	} {
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadConfig(path); err == nil {
			t.Fatal("invalid unit map config accepted", data)
		}
	}
	if err := os.WriteFile(path, []byte(`{"run":{"mode":"campaign","campaign_route":["unit_a","unit_c"],"campaign_unit_maps":["unit_b"]}}`), 0600); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(path)
	if err != nil || !slices.Equal(cfg.CampaignUnitMaps, []string{"unit_b"}) {
		t.Fatal(cfg.CampaignUnitMaps, err)
	}
}

func TestCampaignRouteConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "route.json")
	for _, data := range []string{
		`{"run":{"mode":"companion","campaign_route":["base1","base2"]}}`,
		`{"run":{"mode":"campaign","next_map":"base2","campaign_route":["base1","base2"]}}`,
		`{"run":{"mode":"campaign","campaign_route":["base1"]}}`,
	} {
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadConfig(path); err == nil {
			t.Fatal("invalid route config accepted", data)
		}
	}
	if err := os.WriteFile(path, []byte(`{"run":{"mode":"campaign","campaign_route":["base1","base2","base3"]}}`), 0600); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(path)
	if err != nil || !cfg.Campaign || cfg.CampaignNextMap != "" || !slices.Equal(cfg.CampaignRoute, []string{"base1", "base2", "base3"}) {
		t.Fatal(cfg.CampaignRoute, err)
	}
}

func TestLoadConfigResolvesPathsAndDefaults(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bot.json")
	data := `{"client":{"game_dir":"runtime/baseq2"},"run":{"duration":"5s","frame_paced":true,"game_frames":40},"output":{"trace_jsonl":"traces/bot.jsonl"}}`
	if err := os.WriteFile(path, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Host != "127.0.0.1" || cfg.Port != 27910 || cfg.Name != "GoCoopMate" || cfg.Duration != 5*time.Second || cfg.TestChangeAfter != 20 || !cfg.FramePaced || cfg.GameFrames != 40 {
		t.Fatalf("defaults or run settings lost: %+v", cfg)
	}
	if cfg.GameDir != filepath.Join(dir, "runtime", "baseq2") || cfg.TracePath != filepath.Join(dir, "traces", "bot.jsonl") {
		t.Fatalf("relative paths not resolved from config: %+v", cfg)
	}
}

func TestDoorProbeSpeedConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bot.json")
	if err := os.WriteFile(path, []byte(`{"test":{"door_pass_probe":true,"door_pass_speed":160}}`), 0600); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.TestDoorPassProbe || cfg.TestDoorPassSpeed != 160 {
		t.Fatal("door probe speed not forwarded")
	}
}

func TestLoadConfigRejectsTyposAndTrailingData(t *testing.T) {
	for _, tc := range []struct {
		name, data, want string
	}{
		{"unknown", `{"client":{"game_dri":"runtime"}}`, "unknown field"},
		{"duration", `{"run":{"duration":"tomorrow"}}`, "invalid run.duration"},
		{"second_document", `{} {}`, "more than one JSON value"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "bot.json")
			if err := os.WriteFile(path, []byte(tc.data), 0600); err != nil {
				t.Fatal(err)
			}
			_, err := LoadConfig(path)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v, want %q", err, tc.want)
			}
		})
	}
}
