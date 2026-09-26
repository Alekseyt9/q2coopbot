package harness

type ObserverCycleReport struct {
	Passed         bool   `json:"passed"`
	Reason         string `json:"reason,omitempty"`
	Stimulus       string `json:"stimulus"`
	DeathFrame     int    `json:"death_frame,omitempty"`
	RespawnFrame   int    `json:"respawn_frame,omitempty"`
	RecoveryFrames int    `json:"recovery_frames"`
}

func checkObserverCycle(f ObserverRespawn, start int, rows []Trace) ObserverCycleReport {
	r := ObserverCycleReport{Stimulus: "scripted_kill_and_respawn", Reason: "observer_cycle_incomplete"}
	target := start + f.AfterFrames
	kills := 0
	for _, row := range rows {
		if row.ObserverKill {
			kills++
			if row.Frame != target {
				r.Reason = "observer_kill_wrong_frame"
				return r
			}
		}
		if row.Frame < target {
			continue
		}
		if row.Health == nil {
			r.Reason = "observer_health_missing"
			return r
		}
		if row.ObserverRespawn && (*row.Health > 0 || row.Command.Buttons != 1) {
			r.Reason = "invalid_scripted_respawn_command"
			return r
		}
		if *row.Health <= 0 && row.Command.Buttons != 0 && !row.ObserverRespawn {
			r.Reason = "unmarked_respawn_command"
			return r
		}
		if row.Frame == target && (!row.ObserverKill || *row.Health <= 0) {
			r.Reason = "observer_kill_without_living_trigger"
			return r
		}
		if row.Frame > target && *row.Health <= 0 {
			if r.RespawnFrame > 0 {
				r.Reason = "observer_died_again"
				return r
			}
			if row.Command.Forward != 0 || row.Command.Side != 0 || row.Command.Up != 0 {
				r.Reason = "observer_moving_while_dead"
				return r
			}
			if r.DeathFrame == 0 {
				r.DeathFrame = row.Frame
			}
		} else if r.DeathFrame > 0 && r.RespawnFrame == 0 {
			r.RespawnFrame = row.Frame
		}
		if r.RespawnFrame > 0 && row.Frame > r.RespawnFrame && row.Frame <= r.RespawnFrame+f.RecoveryFrames {
			r.RecoveryFrames++
		}
	}
	if kills != 1 {
		r.Reason = "observer_kill_count"
		return r
	}
	if r.DeathFrame == 0 {
		r.Reason = "observer_death_not_observed"
		return r
	}
	if r.RespawnFrame == 0 || r.RespawnFrame > target+f.TimeoutFrames {
		r.Reason = "observer_respawn_timeout"
		return r
	}
	if r.RecoveryFrames < f.RecoveryFrames {
		return r
	}
	r.Passed = true
	r.Reason = ""
	return r
}
