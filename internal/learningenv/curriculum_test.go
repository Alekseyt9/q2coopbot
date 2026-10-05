package learningenv

import (
	"strings"
	"testing"
)

func TestCurriculumProofIsReleaseBoundAndDefaultAbsent(t *testing.T) {
	release := "sv_test_combat spawncount=42 server_frame=31 g_test_combat_start game_frame=30 ready=1 seed=14500\n"
	init := "g_test_curriculum monster_health map=base1 game_frame=30 entity=67 class=monster_parasite before=175 after=20\n"
	if target, err := VerifyCurriculum(strings.NewReader(release+init), 20, 14500); err != nil || target != 67 {
		t.Fatal(target, err)
	}
	if _, err := VerifyCurriculum(strings.NewReader(release), 0, 14500); err != nil {
		t.Fatal(err)
	}
	for _, s := range []string{release, release + init + init, release + strings.Replace(init, "game_frame=30", "game_frame=31", 1), release + strings.Replace(init, "after=20", "after=10", 1), release + strings.Replace(init, "before=175", "before=40", 1), strings.Replace(release, "seed=14500", "seed=14501", 1) + init} {
		if _, err := VerifyCurriculum(strings.NewReader(s), 20, 14500); err == nil {
			t.Fatal("accepted invalid initialization", s)
		}
	}
	if _, err := VerifyCurriculum(strings.NewReader(release+init), 0, 14500); err == nil {
		t.Fatal("hidden override in ordinary episode")
	}
}
