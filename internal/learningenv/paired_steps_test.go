package learningenv

import (
	"strings"
	"testing"
)

func pairedFixture() string {
	return "sv_test_pair_step version=1 phase=queued spawncount=42 frame=98 role=1 seq=20 actor=2\n" +
		"sv_test_pair_step version=1 phase=queued spawncount=42 frame=98 role=0 seq=19 actor=1\n" +
		"sv_test_pair_cmd version=1 spawncount=42 frame=98 role=0 seq=19 actor=1 pitch=0 yaw=0 roll=0 forward=0 side=0 up=0 buttons=0 impulse=0 msec=100 light=30\n" +
		"sv_test_pair_cmd version=1 spawncount=42 frame=98 role=1 seq=20 actor=2 pitch=0 yaw=0 roll=0 forward=10 side=0 up=0 buttons=0 impulse=0 msec=100 light=40\n" +
		"sv_test_combat spawncount=42 server_frame=99 g_test_combat_start game_frame=100 ready=2 seed=123\n" +
		"sv_test_pair_step version=1 phase=end spawncount=42 frame=99 role=0 seq=19 actor=1\n" +
		"sv_test_pair_step version=1 phase=end spawncount=42 frame=99 role=1 seq=20 actor=2\n"
}
func TestPairedNativeReceiptRequiresBothActors(t *testing.T) {
	text := pairedFixture()
	p, err := ReadPairedNativeSteps(strings.NewReader(text), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Pairs) != 1 || p.Release.Seed != 123 || p.Pairs[0].Commands[1].Command.Forward != 10 {
		t.Fatal(p)
	}
	for _, bad := range []string{
		strings.Replace(text, "phase=queued spawncount=42 frame=98 role=1", "phase=queued spawncount=42 frame=97 role=1", 1),
		strings.Replace(text, "phase=end spawncount=42 frame=99 role=1", "phase=end spawncount=42 frame=100 role=1", 1),
		strings.Replace(text, "ready=2", "ready=1", 1),
		text + "sv_test_pair_step rejected: duplicate\n",
		strings.Replace(text, "role=1 seq=20 actor=2", "role=1 seq=20 actor=1", 1),
		strings.Replace(text, "side=0", "side=40000", 1),
		text[:strings.LastIndex(text, "sv_test_pair_step")],
	} {
		if _, err := ReadPairedNativeSteps(strings.NewReader(bad), nil); err == nil {
			t.Fatal("invalid pair accepted", bad)
		}
	}
}

func TestPairedEffectsPreserveSharedTickWindow(t *testing.T) {
	// Immediate ClientThink damage happens between the two applications;
	// world damage happens after both. Both belong to one shared tick window.
	text := pairedFixture()
	at := "sv_test_pair_cmd version=1 spawncount=42 frame=98 role=1"
	text = strings.Replace(text, at, "sv_test_damage immediate\n"+at, 1)
	text = strings.Replace(text, "sv_test_pair_step version=1 phase=end spawncount=42 frame=99 role=0", "sv_test_damage world\nsv_test_pair_step version=1 phase=end spawncount=42 frame=99 role=0", 1)
	events := []DamageEvent{{Spawncount: 42, Frame: 98, Attacker: 1}, {Spawncount: 42, Frame: 99, Attacker: 2}}
	p, err := ReadPairedNativeSteps(strings.NewReader(text), events)
	if err != nil {
		t.Fatal(err)
	}
	for _, step := range p.Pairs[0].Steps {
		if len(step.DamageIndexes) != 2 || step.DamageIndexes[0] != 0 || step.DamageIndexes[1] != 1 {
			t.Fatal(step)
		}
	}
	events[1].Frame = 100
	if _, err = ReadPairedNativeSteps(strings.NewReader(text), events); err == nil {
		t.Fatal("effect outside tick accepted")
	}
	events[1].Frame = 99
	events[1].Spawncount = 43
	if _, err = ReadPairedNativeSteps(strings.NewReader(text), events); err == nil {
		t.Fatal("effect from different world accepted")
	}
	if _, err = ReadPairedNativeSteps(strings.NewReader(text), events[:1]); err == nil {
		t.Fatal("missing effect accepted")
	}
}
