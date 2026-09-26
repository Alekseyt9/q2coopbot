package harness

import "fmt"

type PhaseReport struct {
	Connection    int                  `json:"connection,omitempty"`
	ObserverCycle *ObserverCycleReport `json:"observer_respawn_cycle,omitempty"`
	ID            string               `json:"id"`
	Map           string               `json:"map"`
	Generation    int                  `json:"generation"`
	Report        Report               `json:"report"`
}

type SessionReport struct {
	TransitionChecks []TransitionCheck `json:"transition_checks,omitempty"`
	Commands         *CommandProof     `json:"observer_commands,omitempty"`
	Accepted         bool              `json:"accepted"`
	State            string            `json:"state"`
	Reason           string            `json:"reason,omitempty"`
	Timeline         SessionTimeline   `json:"session_timeline"`
	Phases           []PhaseReport     `json:"phases"`
}

// AnalyzeSession never combines equal frame numbers from different generations.
func AnalyzeSession(s Session, actor, bot []Trace) SessionReport {
	r := SessionReport{State: "trace_invalid", Timeline: SessionTimeline{Actor: traceTimeline(actor), Bot: traceTimeline(bot)}, Phases: []PhaseReport{}}
	if err := s.Validate(); err != nil {
		r.Reason = err.Error()
		return r
	}
	maps := make([]string, len(s.Phases))
	for i, phase := range s.Phases {
		maps[i] = phase.Scenario.Map
	}
	if reason, _ := verifyMapSequence(maps, r.Timeline); reason != "" {
		r.Reason = reason
		return r
	}
	for i, phase := range s.Phases {
		generation := r.Timeline.Actor.Segments[i].First.Generation
		if i > 0 {
			for _, timeline := range []TraceTimeline{r.Timeline.Actor, r.Timeline.Bot} {
				before, after := timeline.Segments[i-1].First, timeline.Segments[i].First
				if phase.Entry == "reconnect" {
					if before.Generation != after.Generation || before.Connection < 1 || after.Connection <= before.Connection {
						r.Reason = "reconnect identity not observed"
						return r
					}
				} else if before.Generation == after.Generation {
					r.Reason = "new map generation not observed"
					return r
				}
			}
		}
		selectRows := func(rows []Trace, connection int) []Trace {
			var selected []Trace
			for _, row := range rows {
				if row.Map == phase.Scenario.Map && row.Generation == generation && row.Connection == connection {
					selected = append(selected, row)
				}
			}
			return selected
		}
		a, b := selectRows(actor, r.Timeline.Actor.Segments[i].First.Connection), selectRows(bot, r.Timeline.Bot.Segments[i].First.Connection)
		definition := phase.Scenario
		if s.ReadinessBarrier {
			start := 0
			for _, row := range a {
				if row.Session != nil && row.Session.StartFrame > 0 {
					if start != 0 && start != row.Session.StartFrame {
						r.Reason = "phase barrier start changed"
						return r
					}
					start = row.Session.StartFrame
				}
			}
			if start == 0 {
				r.Reason = "phase barrier start missing"
				return r
			}
			for _, rows := range [][]Trace{a, b} {
				for _, row := range rows {
					if row.SessionStartFrame != 0 && row.SessionStartFrame != start || row.Frame >= start && row.SessionStartFrame != start {
						r.Reason = "actor/observer barrier start mismatch"
						return r
					}
					if row.Frame < start && (row.Command.Forward != 0 || row.Command.Side != 0 || row.Command.Up != 0 || row.Command.Buttons != 0) {
						r.Reason = "command before session barrier"
						return r
					}
				}
			}
			definition.GameFrames += start - definition.StartFrame
			definition.StartFrame = start
		}
		for _, row := range a {
			if row.Session == nil || row.Session.PhaseIndex != i || row.Session.PhaseID != phase.ID {
				r.Reason = fmt.Sprintf("phase %s: actor session identity missing/mismatched at frame %d", phase.ID, row.Frame)
				return r
			}
		}
		result := Analyze(definition, a, b)
		phaseReport := PhaseReport{ID: phase.ID, Map: phase.Scenario.Map, Generation: generation, Connection: r.Timeline.Actor.Segments[i].First.Connection, Report: result}
		if phase.ObserverRespawn != nil {
			cycle := checkObserverCycle(*phase.ObserverRespawn, definition.StartFrame, b)
			phaseReport.ObserverCycle = &cycle
			if !cycle.Passed {
				result.Accepted = false
				result.State = "behavior_failed"
				result.Reason = cycle.Reason
				phaseReport.Report = result
			}
		}
		r.Phases = append(r.Phases, phaseReport)
		if !result.Accepted {
			r.State = result.State
			r.Reason = "phase " + phase.ID + ": " + result.Reason
			return r
		}
	}
	completed := false
	for _, row := range actor {
		if row.Session != nil && row.Session.State == "failed" {
			r.State = "fixture_failed"
			r.Reason = row.Session.Reason
			return r
		}
		if row.Session != nil && row.Session.State == "completed" && row.Session.PhaseIndex == len(s.Phases)-1 {
			completed = true
		}
	}
	if !completed {
		r.State = "fixture_failed"
		r.Reason = "session_incomplete"
		return r
	}
	if len(s.Invariants) > 0 {
		r.TransitionChecks = checkSearchTransitions(bot)
		if len(r.TransitionChecks) == 0 {
			r.State = "behavior_failed"
			r.Reason = "invariant_not_exercised: search_reset_on_transition"
			return r
		}
		for _, check := range r.TransitionChecks {
			if !check.Passed {
				r.State = "behavior_failed"
				r.Reason = "search_state_after_transition"
				return r
			}
		}
	}
	r.Accepted, r.State = true, "passed"
	return r
}
