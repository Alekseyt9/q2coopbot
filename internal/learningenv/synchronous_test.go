package learningenv

import (
	"strings"
	"testing"
)

func nativeStepLog() string {
	return "sv_test_step version=1 phase=begin spawncount=2 frame=10 seq=5 actor=1\n" +
		"sv_test_combat spawncount=2 server_frame=11 g_test_combat_start game_frame=13 ready=1 seed=123\n" +
		"sv_test_step version=1 phase=end spawncount=2 frame=11 seq=5 actor=1\n" +
		"sv_test_step version=1 phase=begin spawncount=2 frame=11 seq=6 actor=1\n" +
		nativeDamage + "\n" + strings.Replace(strings.Replace(nativeDamage, "server_frame=11", "server_frame=12", 1), "target=67", "target=68", 1) + "\n" +
		"sv_test_step version=1 phase=end spawncount=2 frame=12 seq=6 actor=1\n"
}

func TestNativeEffectsIncludeImmediateAndWorldTickDamageWithoutShiftingFrames(t *testing.T) {
	text := nativeStepLog()
	events, err := ReadDamageEvents(strings.NewReader("g_test_damage ready version=1\n" + text))
	if err != nil {
		t.Fatal(err)
	}
	n, err := ReadNativeSteps(strings.NewReader(text), events)
	if err != nil || len(n.Steps) != 2 || n.Release.Seed != 123 || len(n.Steps[1].DamageIndexes) != 2 {
		t.Fatal(n, err)
	}
	s := damageStep(11)
	s.ClientSequence = 6
	p, err := n.Match(s)
	if err != nil {
		t.Fatal(err)
	}
	j := DamageJoiner{Events: events}
	o := j.JoinNative(s, p)
	if !o.Available || o.MonsterHealthDamage != 10 || o.MonsterKills != 2 || o.Events[0].Frame != 11 || o.Events[1].Frame != 12 || len(j.Used) != 2 {
		t.Fatal(o)
	}
	j = DamageJoiner{Events: events}
	if old := j.Join(s); old.MonsterHealthDamage != 5 {
		t.Fatal("test did not exercise pre-tick event", old)
	}
	s.Next = nil
	if tail := j.JoinNative(s, p); tail.Available {
		t.Fatal("tail invented next state", tail)
	}
}

func TestNativeStepsRejectOverlapsDiscontinuitiesAndPartialTicks(t *testing.T) {
	text := nativeStepLog()
	events, err := ReadDamageEvents(strings.NewReader("g_test_damage ready version=1\n" + text))
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{
		strings.Replace(text, "phase=end spawncount=2 frame=12 seq=6 actor=1", "phase=end spawncount=2 frame=13 seq=6 actor=1", 1),
		strings.Replace(text, "phase=begin spawncount=2 frame=11 seq=6 actor=1", "phase=begin spawncount=2 frame=12 seq=6 actor=1", 1),
		strings.Replace(text, "phase=end spawncount=2 frame=11 seq=5 actor=1", "phase=begin spawncount=2 frame=11 seq=5 actor=1", 1),
		strings.Replace(text, "phase=end spawncount=2 frame=12 seq=6 actor=1", "phase=end spawncount=2 frame=12 seq=5 actor=1", 1),
		strings.Replace(text, "ready=1 seed=123", "ready=2 seed=123", 1),
		strings.Replace(text, "sv_test_step version=1 phase=end spawncount=2 frame=12 seq=6 actor=1\n", "", 1),
		text + "sv_test_step rejected: recovery\n",
	} {
		if _, err := ReadNativeSteps(strings.NewReader(bad), events); err == nil {
			t.Fatal("invalid stream accepted", bad)
		}
	}
	n, err := ReadNativeSteps(strings.NewReader(text), events)
	if err != nil {
		t.Fatal(err)
	}
	s := damageStep(11)
	s.ClientSequence = 6
	s.Next.Identity.Frame++
	if _, err := n.Match(s); err == nil {
		t.Fatal("unaligned next observation accepted")
	}
}
