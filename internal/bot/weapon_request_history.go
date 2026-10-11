package bot

import "q2coopbot/internal/policy"

// This is local command history, never the server's hidden pending weapon.
// Record only after command() succeeds; that queues reliable data but does not
// prove native acceptance. Cached same-frame observations remain unchanged.
type weaponRequestHistory struct {
	identity policy.Identity
	name     string
}

func (h *weaponRequestHistory) record(id policy.Identity, name string) {
	for _, allowed := range policy.WeaponChoices()[1:] {
		if name == allowed {
			h.identity, h.name = id, name
			return
		}
	}
}

func (h *weaponRequestHistory) enrich(o *policy.Observation) {
	if o.Health <= 0 || h.name != "" && (!policy.SameLife(h.identity, o.Identity) || o.Identity.Frame < h.identity.Frame) {
		*h = weaponRequestHistory{}
	}
	if h.name == "" || o.Identity.Frame <= h.identity.Frame {
		return
	}
	name, age := h.name, o.Identity.Frame-h.identity.Frame
	o.PreviousWeaponRequest, o.PreviousWeaponRequestAgeFrames = &name, &age
}
