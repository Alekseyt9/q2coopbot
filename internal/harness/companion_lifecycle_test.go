package harness

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestCompanionCyclesMatchIdentity(t *testing.T) {
	alive, dead := int16(100), int16(0)
	a := []Trace{{Frame: 40, SelfEntity: 2, Health: &alive}, {Frame: 41, SelfEntity: 2, Health: &dead}, {Frame: 42, SelfEntity: 2, Health: &alive}, {Frame: 43, SelfEntity: 2, Health: &alive}}
	b := []Trace{{Frame: 40}, {Frame: 41, Teammate: &quake.Vec3{}, TeammateEntity: 2}, {Frame: 42, Teammate: &quake.Vec3{}, TeammateEntity: 3, Goal: "follow_teammate", Command: quake.UserCmd{Forward: 100}}, {Frame: 43, Teammate: &quake.Vec3{}, TeammateEntity: 2, Goal: "follow_teammate", Command: quake.UserCmd{Forward: 100}}}
	c := companionLifecycle(a, b)
	if len(c) != 1 || c[0].DeadActorTrackedFrames != 1 || *c[0].RespawnFrame != 42 || *c[0].FirstVisibleFrame != 43 || *c[0].FollowFrame != 43 {
		t.Fatal(c)
	}
	b[3].TeammateEntity = 0
	c = companionLifecycle(a, b)
	if c[0].FirstVisibleFrame != nil || c[0].FollowFrame != nil {
		t.Fatal("unknown identity accepted")
	}
}

func TestEveryRespawnNeedsItsOwnRecovery(t *testing.T) {
	s := fixture()
	s.Expect.Invariants = []string{"respawned_actor_followed"}
	frame := 41
	actor := []Trace{{Frame: 40}, {Frame: 41}, {Frame: 42}}
	bot := []Trace{{Frame: 40}, {Frame: 41}, {Frame: 42}}
	r := Report{Metrics: Metrics{CompanionLifecycle: []CompanionCycle{{DeathFrame: 40, FollowFrame: &frame}, {DeathFrame: 42}}}}
	if checkLifecycle(s, actor, bot, &r) || r.Reason != "respawn_recovery_incomplete" {
		t.Fatal("one recovery covered two cycles", r)
	}
	frame2 := 42
	r = Report{Metrics: Metrics{CompanionLifecycle: []CompanionCycle{{DeathFrame: 40, FollowFrame: &frame}, {DeathFrame: 42, FollowFrame: &frame2}}}}
	if !checkLifecycle(s, actor, bot, &r) {
		t.Fatal(r)
	}
}
