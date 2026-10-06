package learningenv

import (
	"strings"
	"testing"
)

func TestPostFrameRNGReceipt(t *testing.T) {
	release := "sv_test_combat spawncount=7 server_frame=38 g_test_combat_start game_frame=37 ready=1 seed=14900\n"
	marker := "g_test_rng_start game_frame=37 phase=post_frame seed=14900 cursor_before=257 cursor_after=256\n"
	if err := VerifyPostFrameRNG(strings.NewReader(release+marker), 14900, 0); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{release, release + marker + marker, marker, release + strings.ReplaceAll(marker, "game_frame=37", "game_frame=38"), release + strings.ReplaceAll(marker, "seed=14900", "seed=14901"), release + strings.ReplaceAll(marker, "after=256", "after=257"), release + strings.ReplaceAll(marker, "before=257", "before=9999999999999999999999999")} {
		if err := VerifyPostFrameRNG(strings.NewReader(bad), 14900, 0); err == nil {
			t.Fatal("accepted invalid receipt")
		}
	}
	w := "g_test_weapon_start game_frame=37 actor=1 weapon=Blaster gunframe_before=16 gunframe_after=9\n"
	if err := VerifyPostFrameRNG(strings.NewReader(release+w+marker), 14900, 37); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{"", w + w, strings.ReplaceAll(w, "after=9", "after=10"), strings.ReplaceAll(w, "actor=1", "actor=2"), strings.ReplaceAll(w, "game_frame=37", "game_frame=38")} {
		if err := VerifyPostFrameRNG(strings.NewReader(release+bad+marker), 14900, 37); err == nil {
			t.Fatal("accepted bad fixed weapon receipt")
		}
	}
}

func TestMachinegunReceiptRequiresDeclaredWeaponAndIdlePhase(t *testing.T) {
	release := "sv_test_combat spawncount=7 server_frame=38 g_test_combat_start game_frame=37 ready=1 seed=14900\n"
	marker := "g_test_rng_start game_frame=37 phase=post_frame seed=14900 cursor_before=257 cursor_after=256\n"
	w := "g_test_weapon_start game_frame=37 actor=1 weapon=Machinegun gunframe_before=24 gunframe_after=6\n"
	if err := VerifyPostFrameRNG(strings.NewReader(release+w+marker), 14900, 37, "Machinegun"); err != nil {
		t.Fatal(err)
	}
	if err := VerifyPostFrameRNG(strings.NewReader(release+w+marker), 14900, 37); err == nil {
		t.Fatal("Machinegun accepted as a Blaster receipt")
	}
	for _, bad := range []string{strings.ReplaceAll(w, "after=6", "after=9"), strings.ReplaceAll(w, "before=24", "before=5"), strings.ReplaceAll(w, "before=24", "before=46"), strings.ReplaceAll(w, "weapon=Machinegun", "weapon=Blaster"), w + w} {
		if err := VerifyPostFrameRNG(strings.NewReader(release+bad+marker), 14900, 37, "Machinegun"); err == nil {
			t.Fatal("bad Machinegun receipt accepted")
		}
	}
}
