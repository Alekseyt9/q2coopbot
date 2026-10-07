package trainingepisodes

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func rolloutFixture(t *testing.T, seed int, model string) string {
	t.Helper()
	dir := t.TempDir()
	source := filepath.Join(dir, "native.log")
	if err := os.WriteFile(source, []byte("native receipt"), 0600); err != nil {
		t.Fatal(err)
	}
	h, _ := Hash(source)
	data := []byte(fmt.Sprintf("{\"seed\":%d,\"index\":0}\n{\"seed\":%d,\"index\":1}\n", seed, seed))
	if err := os.WriteFile(filepath.Join(dir, "rollout.jsonl"), data, 0600); err != nil {
		t.Fatal(err)
	}
	rolloutHash, _ := Hash(filepath.Join(dir, "rollout.jsonl"))
	write(t, filepath.Join(dir, "report.json"), map[string]any{"version": "combat_ppo_rollout_v1", "feature_version": "test", "reward_config_sha256": "reward", "model_sha256": model, "rows": 2, "skipped": 0, "terminals": 1, "rollout_sha256": rolloutHash, "source_sha256": map[string]string{source: h}})
	return dir
}
func TestMergeRejectsMixedPolicyAndDuplicateEpisodes(t *testing.T) {
	a := rolloutFixture(t, 1, "model-a")
	b := rolloutFixture(t, 2, "model-b")
	if err := MergeRollouts([]string{a, b}, filepath.Join(t.TempDir(), "merged")); err == nil {
		t.Fatal("mixed policies accepted")
	}
	b = rolloutFixture(t, 1, "model-a")
	if err := MergeRollouts([]string{a, b}, filepath.Join(t.TempDir(), "merged")); err == nil {
		t.Fatal("reused episode accepted")
	}
	b = rolloutFixture(t, 2, "model-a")
	out := filepath.Join(t.TempDir(), "merged")
	if err := MergeRollouts([]string{a, b}, out); err != nil {
		t.Fatal(err)
	}
	var meta map[string]any
	if err := read(filepath.Join(out, "report.json"), &meta); err != nil {
		t.Fatal(err)
	}
	if meta["rows"] != float64(4) || len(meta["members"].([]any)) != 2 {
		t.Fatal("curriculum contributions lost")
	}
	// A modification to even an original native input invalidates the merge.
	data, err := os.ReadFile(filepath.Join(a, "report.json"))
	if err != nil {
		t.Fatal(err)
	}
	var original map[string]any
	if err := json.Unmarshal(data, &original); err != nil {
		t.Fatal(err)
	}
	for path := range original["source_sha256"].(map[string]any) {
		if err := os.WriteFile(path, []byte("changed"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	if err := MergeRollouts([]string{a, b}, filepath.Join(t.TempDir(), "changed")); err == nil {
		t.Fatal("changed native receipt accepted")
	}
}
