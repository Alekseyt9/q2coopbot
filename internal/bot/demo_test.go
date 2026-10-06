package bot

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDemoConfig(t *testing.T) {
	for _, tc := range []struct{ data, want string }{
		{`{"output":{"trace_jsonl":"bot.jsonl"}}`, "bot.dm2"},
		{`{"output":{"trace_jsonl":"bot.jsonl","record_demo":false}}`, ""},
		{`{"output":{"demo_dm2":"demos/recording.dm2"}}`, "demos/recording.dm2"},
		{`{}`, ""},
	} {
		dir := t.TempDir()
		path := filepath.Join(dir, "bot.json")
		if err := os.WriteFile(path, []byte(tc.data), 0600); err != nil {
			t.Fatal(err)
		}
		cfg, err := LoadConfig(path)
		if err != nil {
			t.Fatal(err)
		}
		want := tc.want
		if want != "" {
			want = filepath.Join(dir, want)
		}
		if cfg.DemoPath != want {
			t.Fatalf("got %q want %q", cfg.DemoPath, want)
		}
	}
}
