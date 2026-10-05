package learningenv

import (
	"q2coopbot/internal/policy"
	"strings"
	"testing"
)

const nativeDamage = "sv_test_damage spawncount=2 server_frame=11 g_test_damage version=1 map=base1 frame=13 attacker=1 target=67 inflictor=90 mod=1 health_before=5 health_after=-5 take=10 armor=0 power=0 protection=0 target_class=monster_parasite attacker_class=player shot=4294967295"

func TestReadDamageRequiresConfirmedConsistentNativeContext(t *testing.T) {
	text := "g_test_damage ready version=1\r\n" + nativeDamage + "\r\n"
	events, err := ReadDamageEvents(strings.NewReader(text))
	if err != nil || len(events) != 1 || events[0].Frame != 11 || events[0].GameFrame != 13 || events[0].Shot != 4294967295 {
		t.Fatal(events, err)
	}
	for _, bad := range []string{
		nativeDamage, strings.Replace(text, "health_after=-5", "health_after=-4", 1),
		strings.Replace(text, "version=1 map=", "version=2 map=", 1),
		strings.Replace(text, "sv_test_damage spawncount=2 server_frame=11 ", "", 1),
		strings.Replace(text, "shot=4294967295", "shot=4294967296", 1),
		text + strings.Replace(nativeDamage, "server_frame=11", "server_frame=10", 1),
	} {
		if _, err := ReadDamageEvents(strings.NewReader(bad)); err == nil {
			t.Fatal("accepted invalid native log", bad)
		}
	}
}

func damageStep(frame int) *Step {
	id := policy.Identity{Life: 1, Map: "base1", Connection: 1, Spawncount: 2, Actor: 1, Frame: frame}
	next := id
	next.Frame++
	return &Step{Index: frame, Worker: "worker-0", Episode: "seed-1", Observation: policy.Observation{Identity: id}, Next: &policy.Observation{Identity: next}}
}

func TestServerWindowsCapLivingDamageExcludeCorpseAndOtherWorld(t *testing.T) {
	e := DamageEvent{Map: "base1", Spawncount: 2, Frame: 11, Attacker: 1, Target: 67, TargetClass: "monster_parasite", HealthBefore: 5, HealthAfter: -5, Take: 10}
	corpse := e
	corpse.HealthBefore = -5
	corpse.HealthAfter = -15
	friendly := e
	friendly.Target = 2
	friendly.TargetClass = "player"
	friendly.HealthBefore = 100
	friendly.HealthAfter = 90
	self := friendly
	self.Target = 1
	incoming := friendly
	incoming.Attacker = 67
	incoming.Target = 1
	foreign := e
	foreign.Spawncount = 3
	previous := e
	previous.Frame = 10
	j := DamageJoiner{Events: []DamageEvent{e, corpse, friendly, self, incoming, foreign, previous}}
	o := j.Join(damageStep(10))
	if !o.Available || o.MonsterHealthDamage != 5 || o.MonsterKills != 1 || o.TeammateHealthDamage != 10 || o.SelfHealthDamage != 10 || o.ReceivedHealthDamage != 20 || o.Score != nil || len(o.Events) != 5 {
		t.Fatal(o)
	}
	if len(j.Used) != 5 {
		t.Fatal(j.Used)
	}
	if next := j.Join(damageStep(11)); len(next.Events) != 0 || !next.Available {
		t.Fatal("double-counted frame", next)
	}
}

func TestServerOutcomeUnavailableAcrossGapOrReset(t *testing.T) {
	for _, change := range []func(*Step){func(s *Step) { s.Next = nil }, func(s *Step) { s.Next.Identity.Frame++ }, func(s *Step) { s.Next.Identity.Life++ }, func(s *Step) { s.Next.Identity.Connection++ }, func(s *Step) { s.Next.Identity.Actor++ }, func(s *Step) { s.Next.Identity.Map = "base2" }} {
		s := damageStep(10)
		change(s)
		j := DamageJoiner{}
		o := j.Join(s)
		if o.Available || o.Reason == "" || len(j.Used) != 0 {
			t.Fatal(o)
		}
	}
}
