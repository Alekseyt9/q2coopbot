package harness

import "q2coopbot/internal/quake"

// This is a trace contract, independent of the bot's planner implementation.
type SearchAttempt struct {
	Entity        int    `json:"entity"`
	LastSeenFrame int    `json:"last_seen_frame"`
	State         string `json:"state"`
	EndFrame      int    `json:"end_frame,omitempty"`
}
type Check struct {
	Name            string `json:"name"`
	EvaluatedFrames int    `json:"evaluated_frames"`
	State           string `json:"state"`
}
type ContextFrame struct {
	ActorMovementReason string         `json:"actor_movement_reason,omitempty"`
	ActorPosition       *quake.Vec3    `json:"actor_position,omitempty"`
	ActorCommand        quake.UserCmd  `json:"actor_command"`
	BotPosition         *quake.Vec3    `json:"bot_position,omitempty"`
	SearchTarget        *quake.Vec3    `json:"search_target,omitempty"`
	Frame               int            `json:"frame"`
	ActorStep           string         `json:"actor_step,omitempty"`
	Visible             bool           `json:"visible"`
	Goal                string         `json:"goal"`
	Forward             int16          `json:"forward"`
	Side                int16          `json:"side"`
	Up                  int16          `json:"up"`
	SearchAttempt       *SearchAttempt `json:"search_attempt,omitempty"`
}

func knownInvariant(name string) bool {
	switch name {
	case "wait_has_no_movement", "completed_search_stays_finished", "visible_contact_clears_search", "respawned_actor_followed":
		return true
	}
	return false
}
func addContext(r *Report, actor, bot []Trace, index int) {
	for i := max(0, index-2); i <= min(len(bot)-1, index+2); i++ {
		row := bot[i]
		c := ContextFrame{Frame: row.Frame, Visible: row.Teammate != nil, Goal: row.Goal, Forward: row.Command.Forward, Side: row.Command.Side, Up: row.Command.Up, SearchAttempt: row.SearchAttempt, SearchTarget: row.SearchTarget}
		c.ActorPosition = actor[i].Self
		c.ActorCommand = actor[i].Command
		c.BotPosition = row.Self
		if actor[i].Scenario != nil {
			c.ActorMovementReason = actor[i].Scenario.MovementReason
			c.ActorStep = actor[i].Scenario.StepID
		}
		r.Context = append(r.Context, c)
	}
}
func checkLifecycle(s Scenario, actor, bot []Trace, r *Report) bool {
	for _, name := range s.Expect.Invariants {
		r.Checks = append(r.Checks, Check{Name: name, State: "not_exercised"})
	}
	// A completed attempt stays terminal even if the trace temporarily omits it.
	type observation struct{ entity, frame int }
	completed := map[observation]bool{}
	lastKey := observation{}
	for i, row := range bot {
		searching := row.Goal == "search_last_seen" || row.Goal == "probe_last_seen" || row.SearchTarget != nil
		a := row.SearchAttempt
		key := lastKey
		if row.TeammateAgeFrames != nil {
			key.frame = row.Frame - *row.TeammateAgeFrames
		}
		if a != nil {
			key = observation{a.Entity, a.LastSeenFrame}
			if a.State == "completed" {
				completed[key] = true
			}
		}
		lastKey = key
		for j := range r.Checks {
			check := &r.Checks[j]
			applies, violated := false, false
			switch check.Name {
			case "respawned_actor_followed":
				for _, cycle := range r.Metrics.CompanionLifecycle {
					if cycle.FollowFrame != nil && *cycle.FollowFrame == row.Frame {
						applies = true
					}
				}
			case "wait_has_no_movement":
				applies = row.Goal == "wait_for_teammate"
				violated = row.Command.Forward != 0 || row.Command.Side != 0 || row.Command.Up != 0
			case "completed_search_stays_finished":
				applies = completed[key]
				violated = searching || a != nil && a.State == "active"
			case "visible_contact_clears_search":
				applies = row.Teammate != nil
				violated = searching || a != nil && a.State == "active"
			}
			if !applies {
				continue
			}
			check.EvaluatedFrames++
			check.State = "passed"
			if violated {
				check.State = "failed"
				r.State = "behavior_failed"
				r.Reason = check.Name
				r.Frame = row.Frame
				addContext(r, actor, bot, i)
				return false
			}
		}
	}
	for _, check := range r.Checks {
		if check.Name == "respawned_actor_followed" && check.EvaluatedFrames != len(r.Metrics.CompanionLifecycle) {
			r.State = "behavior_failed"
			r.Reason = "respawn_recovery_incomplete"
			r.Frame = bot[len(bot)-1].Frame
			addContext(r, actor, bot, len(bot)-1)
			return false
		}
		if check.EvaluatedFrames == 0 {
			r.State = "behavior_failed"
			r.Reason = "invariant_not_exercised: " + check.Name
			r.Frame = bot[len(bot)-1].Frame
			addContext(r, actor, bot, len(bot)-1)
			return false
		}
	}
	return true
}
