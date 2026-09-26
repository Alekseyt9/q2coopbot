package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestInterruptedSearchWaitsUntilFreshContact(t *testing.T) {
	p, s, _ := searchAttemptFixture()
	s.OnGround = false
	if _, _, ok := p.hiddenTeammateGoal(s); ok {
		t.Fatal("airborne search continued")
	}
	if p.searchAttempt.Outcome != "preconditions_lost" || p.probeTarget != nil {
		t.Fatal("terminal target retained")
	}
	s.OnGround = true
	s.Frame++
	age := 4
	s.TeammateAgeFrames = &age
	if _, _, ok := p.hiddenTeammateGoal(s); ok {
		t.Fatal("completed attempt resumed after landing")
	}
	visible := quake.Vec3{300, 0, 24}
	s.Frame++
	s.Teammate = &visible
	s.TeammateEntity = 2
	age = 0
	p.update(s, "")
	if p.World.Goal != "follow_teammate" || !p.hasGoal || p.probeAttempted {
		t.Fatal("fresh contact failed to restore following")
	}
	s.Frame++
	s.Teammate = nil
	s.LastTeammate = &visible
	age = 1
	if _, kind, ok := p.hiddenTeammateGoal(s); !ok || kind != "search_last_seen" {
		t.Fatal("new loss inherited exhausted budget")
	}
}

func TestNewHiddenObservationCannotReuseOldViewpoint(t *testing.T) {
	for _, changeEntity := range []bool{false, true} {
		p, s, old := searchAttemptFixture()
		s.LastTeammate = &quake.Vec3{300, 0, 24}
		age := 2
		s.TeammateAgeFrames = &age
		if changeEntity {
			s.LastTeammateEntity = 3
			age = 3
		}
		goal, kind, ok := p.hiddenTeammateGoal(s)
		if !ok || kind != "search_last_seen" || goal == old || p.searchAttempt != nil || p.probeTarget != nil {
			t.Fatal("old observation leaked into new search")
		}
	}
}

func TestSearchExpirationAndMapChangeDiscardTarget(t *testing.T) {
	p, s, _ := searchAttemptFixture()
	s.Frame = 49
	age := 41
	s.TeammateAgeFrames = &age
	p.hiddenTeammateGoal(s)
	if p.probeTarget != nil || p.searchAttempt.State != "completed" {
		t.Fatal("expired target retained")
	}
	p.setMap("", "")
	if p.searchAttempt != nil || p.probeAttempted || p.searchApproachStarted || p.probeTarget != nil {
		t.Fatal("map reset retained search state")
	}
}
