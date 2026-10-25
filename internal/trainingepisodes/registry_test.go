package trainingepisodes

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixture(t *testing.T) (string, Episode) {
	t.Helper()
	r, err := Load(filepath.Join("..", "..", "scripts", "scenarios", "combat-training", "index.json"))
	if err != nil {
		t.Fatal(err)
	}
	d := t.TempDir()
	var ep Episode
	for _, candidate := range r.Episodes {
		if candidate.Status == "runnable" && candidate.Recipe.Runner == "combat-baseline" {
			ep = candidate
			break
		}
	}
	if ep.ID == "" {
		t.Fatal("no runnable combat fixture")
	}
	write(t, filepath.Join(d, "index.json"), Manifest{Version: 1, Files: []string{"scene.json"}})
	write(t, filepath.Join(d, "scene.json"), ep)
	return d, ep
}
func write(t *testing.T, path string, v any) {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
}
func TestRegistryRejectsLeakageAndUnregisteredScenes(t *testing.T) {
	d, ep := fixture(t)
	ep.Splits["test"] = ep.Splits["train"]
	write(t, filepath.Join(d, "scene.json"), ep)
	if _, err := Load(filepath.Join(d, "index.json")); err == nil || !strings.Contains(err.Error(), "leakage") {
		t.Fatalf("leaked seeds accepted: %v", err)
	}
	d, ep = fixture(t)
	write(t, filepath.Join(d, "extra.json"), ep)
	if _, err := Load(filepath.Join(d, "index.json")); err == nil || !strings.Contains(err.Error(), "unregistered") {
		t.Fatalf("unlisted scene accepted: %v", err)
	}
}

func TestRegistryNonterminalHandoffRecipe(t *testing.T) {
	_, ep := fixture(t)
	ep.Recipe.RewardConfig = "scripts/scenarios/combat-reward-nonterminal-handoff-v13.json"
	if err := ep.validate(); err != nil {
		t.Fatal(err)
	}
	ep.Recipe.RewardConfig = "scripts/scenarios/unregistered-reward.json"
	if err := ep.validate(); err == nil {
		t.Fatal("unregistered reward accepted")
	}
}

func TestRegistryStartingShotgunRecipe(t *testing.T) {
	d, ep := fixture(t)
	ep.Recipe.Loadout = "shotgun"
	write(t, filepath.Join(d, "scene.json"), ep)
	if _, err := Load(filepath.Join(d, "index.json")); err != nil {
		t.Fatal(err)
	}
	ep.Recipe.Loadout = "arbitrary-weapon"
	write(t, filepath.Join(d, "scene.json"), ep)
	if _, err := Load(filepath.Join(d, "index.json")); err == nil {
		t.Fatal("unknown loadout admitted")
	}
}

func TestRegistryBlockedGroundObjectiveRecipes(t *testing.T) {
	_, ep := fixture(t)
	for _, path := range []string{"scripts/scenarios/combat-reward-blocked-ground-control-v14.json", "scripts/scenarios/combat-reward-blocked-ground-cost-v14.json"} {
		ep.Recipe.RewardConfig = path
		if err := ep.validate(); err != nil {
			t.Fatal(path, err)
		}
	}
	ep.Recipe.RewardConfig = "workspace/artifacts/unregistered-reward.json"
	if ep.validate() == nil {
		t.Fatal("arbitrary reward path accepted")
	}
}
func TestPlanRejectsPlannedSceneAndInvalidSplitBudget(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	r, err := Load(filepath.Join(root, "scripts", "scenarios", "combat-training", "index.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		id, split     string
		count, offset int
	}{
		{"soldier-infantry-corner", "train", 4, 0},
		{"parasite-blaster-solo", "test", 4, 63},
		{"parasite-blaster-solo", "validation", 36, 0},
		{"base1-natural-campaign", "train", 4, 0},
		{"missing", "train", 4, 0},
	} {
		if _, err := Build(r, root, []string{tc.id}, tc.split, "rules", "", t.TempDir(), tc.count, tc.offset); err == nil {
			t.Fatalf("invalid plan accepted: %+v", tc)
		}
	}
	p, err := Build(r, root, []string{"campaign-base1-native-site-01-blaster", "campaign-base1-native-site-02-blaster"}, "train", "rules", "", t.TempDir(), 4, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Tasks) != 2 || p.Workers != 4 || p.Tasks[0].Seeds[0] != 80000008 || p.Tasks[1].Seeds[0] != 80040008 {
		t.Fatalf("wrong scene/seed schedule: %+v", p)
	}
}
func TestRegistryRejectsTraversalAndUnknownParameters(t *testing.T) {
	d, _ := fixture(t)
	write(t, filepath.Join(d, "index.json"), Manifest{Version: 1, Files: []string{"../scene.json"}})
	if _, err := Load(filepath.Join(d, "index.json")); err == nil {
		t.Fatal("path traversal accepted")
	}
	d, _ = fixture(t)
	b, err := os.ReadFile(filepath.Join(d, "scene.json"))
	if err != nil {
		t.Fatal(err)
	}
	b = append([]byte(`{"shell_command":"bad",`), b[1:]...)
	if err = os.WriteFile(filepath.Join(d, "scene.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(filepath.Join(d, "index.json")); err == nil {
		t.Fatal("unknown executable field accepted")
	}
}

func TestSmallEpisodeCohortsRemainVerifiable(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	r, err := Load(filepath.Join(root, "scripts", "scenarios", "combat-training", "index.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, count := range []int{1, 2, 3, 4} {
		p, err := Build(r, root, []string{"campaign-base1-native-site-01-blaster"}, "validation", "rules", "", t.TempDir(), count, 8)
		if err != nil {
			t.Fatal(err)
		}
		if len(p.Tasks[0].Seeds) != count {
			t.Fatal("budget changed", p.Tasks[0].Seeds)
		}
		path := filepath.Join(t.TempDir(), "plan.json")
		write(t, path, p)
		if err := VerifyPlan(path, root); err != nil {
			t.Fatal(err)
		}
	}
	for _, count := range []int{0, -1, 81} {
		if _, err := Build(r, root, []string{"campaign-base1-native-site-01-blaster"}, "validation", "rules", "", t.TempDir(), count, 8); err == nil {
			t.Fatal("unbounded budget accepted", count)
		}
	}
}
