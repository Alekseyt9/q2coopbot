package harness

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestLastPlayerRecoveryRejectsDeathTargetAndMissingArrival(t *testing.T) {
	player := quake.Vec3{400, 200, 24}
	f := ObserverRespawn{AfterFrames: 1, TimeoutFrames: 6, RecoveryFrames: 4, PolicyOnly: true, RecoveryExpectation: "last_player_arrival"}
	fixture := func() []Trace {
		r := cycleFixture()
		for i := range r {
			r[i].Self = &quake.Vec3{100, 200, 24}
			if r[i].Frame < 52 {
				r[i].Teammate = &player
			} else {
				r[i].Teammate = nil
			}
			if r[i].ObserverRespawn {
				r[i].ObserverRespawn = false
				r[i].Arbitration.LimitReason = "respawn_request"
			}
			if r[i].Frame >= 55 {
				r[i].Self = &quake.Vec3{800, 200, 24}
				r[i].Goal = "regroup_after_respawn"
				r[i].GoalPoint = &player
				r[i].LastTeammate = &player
			}
		}
		last := r[len(r)-1]
		last.Frame++
		last.Self = &player
		last.Goal = "wait_for_teammate"
		return append(r, last)
	}
	if report := checkObserverCycle(f, 50, fixture()); !report.Passed {
		t.Fatal(report)
	}
	for _, name := range []string{"no-player", "death-target", "wrong-memory", "visible-return", "visible-arrival", "no-arrival", "new-death"} {
		t.Run(name, func(t *testing.T) {
			r := fixture()
			switch name {
			case "no-player":
				for i := range r {
					if r[i].Frame < 52 {
						r[i].Teammate = nil
					}
				}
			case "death-target":
				r[6].GoalPoint = &quake.Vec3{100, 200, 24}
			case "wrong-memory":
				r[6].LastTeammate = &quake.Vec3{}
			case "visible-return":
				r[6].Teammate = &player
			case "no-arrival":
				r[len(r)-1].Self = &quake.Vec3{800, 200, 24}
			case "visible-arrival":
				r[len(r)-1].Teammate = &player
			case "new-death":
				hp := int16(0)
				r[len(r)-1].Health = &hp
			}
			if report := checkObserverCycle(f, 50, r); report.Passed {
				t.Fatal("invalid recovery accepted")
			}
		})
	}
}
