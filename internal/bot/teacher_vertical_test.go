package bot

import (
	"context"
	"q2coopbot/internal/quake"
	"strings"
	"testing"
)

func TestVerticalTeacherRejectsWrongFixture(t *testing.T) {
	cfg := Config{Host: "127.0.0.1", CombatMode: "rules", TestTeacherVertical: true, TestSynchronous: true, CombatCapture: true, FramePaced: true, TestCombatBarrier: true, TestTeleport: "32,-224,24", TestTeleportMap: "base1", TestWeaponSwitchFixture: "parasite_blaster", TracePath: "unused"}
	if err := Run(context.Background(), cfg); err == nil || !strings.Contains(err.Error(), "vertical teacher requires") {
		t.Fatal(err)
	}
}

func TestVerticalTeacherIsFiniteAndReleasesJump(t *testing.T) {
	original := quake.UserCmd{Forward: 100, Side: 20, Buttons: 1}
	for age := -1; age <= 21; age++ {
		cmd, active := teacherVerticalCommand(original, age)
		if active != (age >= 0 && age < 20) {
			t.Fatal(age, active)
		}
		if !active {
			if cmd != original {
				t.Fatal("outside exercise changed")
			}
			continue
		}
		if cmd.Forward != 0 || cmd.Side != 0 || cmd.Buttons != 0 {
			t.Fatal("horizontal/fire hold absent", cmd)
		}
		want := int16(0)
		if age == 1 {
			want = 400
		}
		if age >= 10 && age < 14 {
			want = -400
		}
		if cmd.Up != want {
			t.Fatal(age, cmd.Up, want)
		}
	}
}
