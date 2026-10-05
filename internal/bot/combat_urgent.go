package bot

import "q2coopbot/internal/quake"

type urgentRetreat struct {
	target, until int
}

// Latch a short escape after observed fast approach, not every nearby enemy.
// Clear on discontinuity or escape so ordinary spacing keeps its slow control.
func (p *Planner) observeUrgentRetreat(s quake.Snapshot) {
	old := p.World.Snapshot
	if old.Map != s.Map || old.Frame+1 != s.Frame || old.Health <= 0 || s.Health <= 0 || !s.OnGround {
		p.urgentRetreat = urgentRetreat{}
		p.cornerEscape = nil
		p.cornerUrgency = urgentRetreat{}
		return
	}
	active := false
	cornerActive := false
	for _, e := range s.Enemies {
		if e.Class != "monster_parasite" || e.ClearShot == nil || !*e.ClearShot {
			continue
		}
		d := quake.Horizontal(s.Self, e.Origin)
		minimum, _ := combatDistanceBand(s.Weapon, e.Class)
		if e.ID == p.cornerUrgency.target && s.Frame <= p.cornerUrgency.until && d < minimum {
			cornerActive = true
		}
		if e.ID == p.urgentRetreat.target && s.Frame <= p.urgentRetreat.until && d < parasiteFiringDistance {
			active = true
		}
		approach := quake.Horizontal(old.Self, e.Origin) - d
		if d < 128 && approach >= 20 && quake.Distance(old.Self, s.Self) <= 40 {
			p.urgentRetreat = urgentRetreat{target: e.ID, until: s.Frame + 40}
			active = true
		}
	}
	if !active {
		p.urgentRetreat = urgentRetreat{}
	}
	if !cornerActive {
		p.cornerUrgency = urgentRetreat{}
	}
}
