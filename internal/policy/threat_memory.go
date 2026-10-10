package policy

import (
	"q2coopbot/internal/quake"
	"sort"
)

const ThreatMemoryFrames = 200

// RememberedThreat is a past observation, never an occluded current position.
// Relative changes only with own motion; velocity is the last observed vector.
type RememberedThreat struct {
	ID        int         `json:"id"`
	Class     string      `json:"class"`
	ModelPath string      `json:"observed_model"`
	Relative  quake.Vec3  `json:"last_observed_relative"`
	Velocity  *quake.Vec3 `json:"last_observed_velocity"`
	AgeFrames int         `json:"age_frames"`
}

type threatRecord struct {
	enemy    Enemy
	position quake.Vec3
	frame    int
}
type ThreatMemory struct {
	identity Identity
	known    map[int]threatRecord
}

func (m *ThreatMemory) Enrich(o *Observation, defeated []quake.Object) {
	o.RememberedThreats = []RememberedThreat{}
	if o.Health <= 0 || o.AgeMS < 0 || o.AgeMS > 300 {
		*m = ThreatMemory{}
		return
	}
	if m.known == nil || !SameLife(m.identity, o.Identity) || o.Identity.Frame != m.identity.Frame+1 {
		*m = ThreatMemory{known: make(map[int]threatRecord)}
	}
	m.identity = o.Identity
	for _, d := range defeated {
		if r, ok := m.known[d.ID]; ok && r.enemy.Class == d.Class {
			delete(m.known, d.ID)
		}
	}
	for _, e := range o.Enemies {
		if e.ID <= 0 || e.ClearShot == nil || !*e.ClearShot {
			continue
		}
		at := o.Position
		for axis := range at {
			at[axis] += e.Relative[axis]
		}
		copy := cloneEnemies([]Enemy{e})[0]
		m.known[e.ID] = threatRecord{copy, at, o.Identity.Frame}
	}
	for id, r := range m.known {
		age := o.Identity.Frame - r.frame
		if age < 0 || age > ThreatMemoryFrames {
			delete(m.known, id)
			continue
		}
		v := r.enemy.Velocity
		if v != nil {
			copy := *v
			v = &copy
		}
		o.RememberedThreats = append(o.RememberedThreats, RememberedThreat{ID: id, Class: r.enemy.Class, ModelPath: r.enemy.ModelPath, Relative: relative(r.position, o.Position), Velocity: v, AgeFrames: age})
	}
	sort.Slice(o.RememberedThreats, func(i, j int) bool {
		a, b := o.RememberedThreats[i], o.RememberedThreats[j]
		if a.AgeFrames != b.AgeFrames {
			return a.AgeFrames < b.AgeFrames
		}
		return a.ID < b.ID
	})
	for len(o.RememberedThreats) > 8 {
		last := len(o.RememberedThreats) - 1
		delete(m.known, o.RememberedThreats[last].ID)
		o.RememberedThreats = o.RememberedThreats[:last]
	}
}
