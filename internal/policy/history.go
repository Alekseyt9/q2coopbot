package policy

import "q2coopbot/internal/quake"

const HistoryFrames = 4

// History contains only past client observations and the last sent command.
// Velocities use game time (10 Hz), independent of wall-clock timescale.
type HistoryFrame struct {
	Identity        Identity      `json:"identity"`
	Position        quake.Vec3    `json:"position"`
	Velocity        quake.Vec3    `json:"velocity"`
	Health          int16         `json:"health"`
	Armor           int16         `json:"armor"`
	OnGround        bool          `json:"on_ground"`
	Ducked          bool          `json:"ducked"`
	ViewAngles      [3]int16      `json:"view_angles"`
	PreviousCommand quake.UserCmd `json:"previous_applied_command"`
	Enemies         []Enemy       `json:"enemies"`
}

type History struct {
	frames    []HistoryFrame
	nextTrack int
}

// Enrich drops history at gaps, stale frames, death and world/life changes.
// Reappearing IDs receive a new local track; no velocity crosses occlusion.
func (h *History) Enrich(o *Observation) {
	o.History = []HistoryFrame{}
	if o.AgeMS < 0 || o.AgeMS > 300 || o.Health <= 0 {
		h.frames = nil
		return
	}
	if len(h.frames) > 0 {
		last := h.frames[len(h.frames)-1]
		if !SameLife(last.Identity, o.Identity) || last.Identity.Frame+1 != o.Identity.Frame {
			h.frames = nil
		}
	}
	var last *HistoryFrame
	if len(h.frames) > 0 {
		last = &h.frames[len(h.frames)-1]
	}
	for i := range o.Enemies {
		e := &o.Enemies[i]
		e.Track, e.Velocity = nil, nil
		if last != nil {
			for _, prev := range last.Enemies {
				if prev.ID == e.ID && prev.Class == e.Class && prev.Track != nil {
					track := *prev.Track
					e.Track = &track
					v := quake.Vec3{}
					for axis := range v {
						v[axis] = (e.Relative[axis] + o.Position[axis] - prev.Relative[axis] - last.Position[axis]) * 10
					}
					e.Velocity = &v
					break
				}
			}
		}
		if e.Track == nil {
			h.nextTrack++
			track := h.nextTrack
			e.Track = &track
		}
	}
	for _, f := range h.frames {
		f.Enemies = cloneEnemies(f.Enemies)
		o.History = append(o.History, f)
	}
	f := HistoryFrame{Identity: o.Identity, Position: o.Position, Velocity: o.Velocity, Health: o.Health, Armor: o.Armor,
		OnGround: o.OnGround, Ducked: o.Ducked, ViewAngles: o.ViewAngles, PreviousCommand: o.PreviousCommand, Enemies: cloneEnemies(o.Enemies)}
	h.frames = append(h.frames, f)
	if len(h.frames) > HistoryFrames {
		h.frames = h.frames[1:]
	}
}

func cloneEnemies(enemies []Enemy) []Enemy {
	copy := append([]Enemy{}, enemies...)
	for i := range copy {
		if copy[i].Track != nil {
			v := *copy[i].Track
			copy[i].Track = &v
		}
		if copy[i].Velocity != nil {
			v := *copy[i].Velocity
			copy[i].Velocity = &v
		}
		if copy[i].ClearShot != nil {
			v := *copy[i].ClearShot
			copy[i].ClearShot = &v
		}
	}
	return copy
}
