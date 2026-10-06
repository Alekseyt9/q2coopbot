package bot

import (
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

// Ownership grace uses native game frames, never wall time or server outcomes.
const combatOcclusionFrames = 30

// Retain only observed identities for ownership bookkeeping. This is not a
// target tracker: it neither stores nor fabricates occluded enemy positions.
type combatEngagement struct {
	identity policy.Identity
	known    map[int]string
	lastSeen int
}

func (e *combatEngagement) observe(o policy.Observation, defeated []quake.Object) (active, continuation bool, fallback string) {
	if e.known == nil || !policy.SameLife(e.identity, o.Identity) || o.Identity.Frame < e.identity.Frame || o.Identity.Frame > e.identity.Frame+1 {
		*e = combatEngagement{known: make(map[int]string)}
	}
	e.identity = o.Identity
	for _, d := range defeated {
		if e.known[d.ID] == d.Class {
			delete(e.known, d.ID)
		}
	}
	for _, enemy := range o.Enemies {
		e.known[enemy.ID] = enemy.Class
	}
	if len(o.Enemies) > 0 {
		e.lastSeen = o.Identity.Frame
		return true, false, ""
	}
	if len(e.known) == 0 {
		return false, false, "system2_noncombat"
	}
	if o.Identity.Frame-e.lastSeen <= combatOcclusionFrames {
		return true, true, ""
	}
	// Explicitly uncertain completion. Do not interpret timeout as a kill.
	clear(e.known)
	return false, false, "combat_visibility_timeout"
}
