package harness

type TransitionCheck struct {
	Name       string        `json:"name"`
	Passed     bool          `json:"passed"`
	Before     FrameLocation `json:"before"`
	After      FrameLocation `json:"after"`
	BeforeGoal string        `json:"before_goal"`
	AfterGoal  string        `json:"after_goal"`
}

func searching(row Trace) bool {
	return row.SearchTarget != nil || row.Goal == "search_last_seen" || row.Goal == "probe_last_seen"
}

// The explicit fixture requires search to be active at the end of a generation
// and absent in the first recorded decision of the next. It does not assert
// anything about unrelated planner state or future searches from new evidence.
func checkSearchTransitions(rows []Trace) []TransitionCheck {
	return checkSearchBoundary(rows, "search_reset_on_transition")
}

func checkSearchBoundary(rows []Trace, name string) []TransitionCheck {
	var checks []TransitionCheck
	for i := 1; i < len(rows); i++ {
		before, after := rows[i-1], rows[i]
		boundary := before.Generation != after.Generation
		if name == "search_reset_on_reconnect" {
			boundary = before.Generation == after.Generation && before.Connection > 0 && after.Connection > before.Connection
		}
		if !boundary || !searching(before) {
			continue
		}
		checks = append(checks, TransitionCheck{Name: name, Passed: !searching(after) && after.SearchAttempt == nil,
			Before: FrameLocation{Row: i, Map: before.Map, Generation: before.Generation, Frame: before.Frame, Connection: before.Connection},
			After:  FrameLocation{Row: i + 1, Map: after.Map, Generation: after.Generation, Frame: after.Frame, Connection: after.Connection}, BeforeGoal: before.Goal, AfterGoal: after.Goal})
	}
	return checks
}
