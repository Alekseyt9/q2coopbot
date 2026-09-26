package harness

func actorLifecycle(rows []Trace) []Event {
	var events []Event
	known, alive := false, false
	for _, r := range rows {
		if r.Health == nil {
			known = false
			continue
		}
		current := *r.Health > 0
		if known && current != alive {
			kind := "actor_died"
			if current {
				kind = "actor_respawned"
			}
			events = append(events, Event{Frame: r.Frame, Kind: kind})
		}
		known = true
		alive = current
	}
	return events
}
