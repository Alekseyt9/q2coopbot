package bot

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

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
