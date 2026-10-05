package learningenv

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAppendLogPreservesPartialLines(t *testing.T) {
	path := filepath.Join(t.TempDir(), "growing.log")
	var reader AppendLog
	if lines, err := reader.Read(path); err != nil || len(lines) != 0 {
		t.Fatal(lines, err)
	}
	os.WriteFile(path, []byte("first\r\npar"), 0600)
	lines, err := reader.Read(path)
	if err != nil || len(lines) != 1 || lines[0] != "first" {
		t.Fatal(lines, err)
	}
	f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
	f.WriteString("tial\n")
	f.Close()
	lines, err = reader.Read(path)
	if err != nil || len(lines) != 1 || lines[0] != "partial" {
		t.Fatal(lines, err)
	}
	lines, err = reader.Read(path)
	if err != nil || len(lines) != 0 {
		t.Fatal(lines, err)
	}
	os.WriteFile(path, []byte("new\n"), 0600)
	if _, err := reader.Read(path); err == nil {
		t.Fatal("accepted truncated live log")
	}
}

func TestLiveWindowMatchesImmediateDamageAndExactDispatch(t *testing.T) {
	_, step, _ := rewardFixture()
	command := "sv_test_applied_cmd spawncount=42 frame=10 seq=23 kind=new pitch=0 yaw=0 roll=0 forward=0 side=0 up=0 buttons=0 impulse=0 msec=0 light=0"
	damage := "sv_test_damage spawncount=42 server_frame=10 g_test_damage version=1 map=base1 frame=5 attacker=1 target=2 inflictor=1 mod=1 health_before=100 health_after=60 take=40 armor=0 power=0 protection=0 target_class=monster_soldier attacker_class=player"
	lines := []string{"SoloRetreatBot connected", "g_test_damage ready version=1", damage,
		"sv_test_step version=1 phase=begin spawncount=42 frame=9 seq=22 actor=1",
		"sv_test_combat spawncount=42 server_frame=10 g_test_combat_start game_frame=4 ready=1 seed=9",
		"sv_test_step version=1 phase=end spawncount=42 frame=10 seq=22 actor=1",
		"sv_test_step version=1 phase=begin spawncount=42 frame=10 seq=23 actor=1", command, damage,
		"sv_test_step version=1 phase=end spawncount=42 frame=11 seq=23 actor=1"}
	live := LiveTelemetry{ClientName: "SoloRetreatBot"}
	for i, line := range lines {
		if err := live.Push(line); err != nil {
			t.Fatal(i, err)
		}
		if i == len(lines)-2 {
			if _, ready, err := live.Enrich(&step); err != nil || ready {
				t.Fatal("published unfinished pulse", ready, err)
			}
		}
	}
	outcome, ready, err := live.Enrich(&step)
	if err != nil || !ready || !outcome.Available || outcome.MonsterHealthDamage != 40 || !step.Execution.WindowExclusive || step.Native.DamageIndexes[0] != 1 {
		t.Fatal(outcome, ready, err, step.Native)
	}
	step.Command.Forward = 100
	if _, _, err := live.Enrich(&step); err == nil {
		t.Fatal("accepted unmatched command")
	}
	if err := live.Push(lines[len(lines)-1]); err == nil {
		t.Fatal("accepted repeated end")
	}
}

func TestLiveNativeRejectsMalformedOrOverlappingPulse(t *testing.T) {
	for _, bad := range []string{"sv_test_step rejected version=1", "sv_test_step version=1 phase=begin spawncount=42 frame=10 seq=23 actor=1", "sv_test_step version=1 phase=end spawncount=42 frame=12 seq=23 actor=1"} {
		live := LiveTelemetry{ClientName: "SoloRetreatBot"}
		live.Push("g_test_damage ready version=1")
		live.Push("sv_test_step version=1 phase=begin spawncount=42 frame=10 seq=23 actor=1")
		if err := live.Push(bad); err == nil {
			t.Fatal(bad)
		}
	}
	if _, err := ReadNativeSteps(strings.NewReader("sv_test_step version=1 phase=begin spawncount=42 frame=10 seq=23 actor=1\nsv_test_step version=1 phase=end spawncount=42 frame=11 seq=23 actor=1\n"), nil); err == nil {
		t.Fatal("offline parser no longer requires release")
	}
}
