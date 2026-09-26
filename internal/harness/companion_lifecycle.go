package harness

type CompanionCycle struct {
	DeathFrame             int  `json:"death_frame"`
	RespawnFrame           *int `json:"respawn_frame"`
	DeadActorTrackedFrames int  `json:"dead_actor_tracked_frames"`
	IdentitySamples        int  `json:"identity_samples"`
	FirstVisibleFrame      *int `json:"first_visible_after_respawn_frame"`
	FollowFrame            *int `json:"follow_after_respawn_frame"`
}

// Actor health is test-oracle data, never an input to the bot's policy.
// Matching entity IDs prevents a different visible player satisfying recovery.
func companionLifecycle(actor, bot []Trace) []CompanionCycle {
	var out []CompanionCycle
	known, alive := false, false
	for i, a := range actor {
		if a.Health == nil {
			known = false
			continue
		}
		current := *a.Health > 0
		if known && alive && !current {
			out = append(out, CompanionCycle{DeathFrame: a.Frame})
		}
		if len(out) > 0 {
			cycle := &out[len(out)-1]
			if known && !alive && current {
				f := a.Frame
				cycle.RespawnFrame = &f
			}
			b := bot[i]
			identityKnown := a.SelfEntity > 0 && (b.Teammate == nil || b.TeammateEntity > 0)
			if identityKnown {
				cycle.IdentitySamples++
			}
			tracked := identityKnown && b.Teammate != nil && b.TeammateEntity == a.SelfEntity
			if !current && tracked {
				cycle.DeadActorTrackedFrames++
			}
			if current && cycle.RespawnFrame != nil && tracked {
				if cycle.FirstVisibleFrame == nil {
					f := a.Frame
					cycle.FirstVisibleFrame = &f
				}
				if cycle.FollowFrame == nil && b.Goal == "follow_teammate" && (b.Command.Forward != 0 || b.Command.Side != 0) {
					f := a.Frame
					cycle.FollowFrame = &f
				}
			}
		}
		known = true
		alive = current
	}
	return out
}
