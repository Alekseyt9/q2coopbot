package policy

import (
	"math"
	"q2coopbot/internal/quake"
)

const HistoryFrames = 4

// History contains only past client observations and the last sent command.
// Velocities use game time (10 Hz), independent of wall-clock timescale.
type HistoryFrame struct {
	Projectiles     *[]Enemy      `json:"visible_projectiles"`
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
	var enemies, projectiles []Enemy
	var previousPosition quake.Vec3
	if last != nil {
		enemies = last.Enemies
		previousPosition = last.Position
		if last.Projectiles != nil {
			projectiles = *last.Projectiles
		}
	}
	h.enrichObjects(o.Enemies, enemies, o.Position, previousPosition)
	if o.Projectiles != nil {
		h.enrichObjects(*o.Projectiles, projectiles, o.Position, previousPosition)
	}
	for _, f := range h.frames {
		f.Enemies = cloneEnemies(f.Enemies)
		f.Projectiles = cloneProjectiles(f.Projectiles)
		o.History = append(o.History, f)
	}
	f := HistoryFrame{Identity: o.Identity, Position: o.Position, Velocity: o.Velocity, Health: o.Health, Armor: o.Armor,
		OnGround: o.OnGround, Ducked: o.Ducked, ViewAngles: o.ViewAngles, PreviousCommand: o.PreviousCommand, Enemies: cloneEnemies(o.Enemies), Projectiles: cloneProjectiles(o.Projectiles)}
	h.frames = append(h.frames, f)
	if len(h.frames) > HistoryFrames {
		h.frames = h.frames[1:]
	}
}

func (h *History) enrichObjects(current, previous []Enemy, position, previousPosition quake.Vec3) {
	for i := range current {
		e := &current[i]
		e.Track, e.Velocity = nil, nil
		e.MotionDirection = nil
		for _, prev := range previous {
			if prev.ID == e.ID && prev.Class == e.Class && prev.Track != nil {
				track := *prev.Track
				e.Track = &track
				v := quake.Vec3{}
				for axis := range v {
					v[axis] = (e.Relative[axis] + position[axis] - prev.Relative[axis] - previousPosition[axis]) * 10
				}
				e.Velocity = &v
				speed := math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
				if speed > 1e-6 {
					direction := quake.Vec3{v[0] / speed, v[1] / speed, v[2] / speed}
					e.MotionDirection = &direction
				}
				break
			}
		}
		if e.Track == nil {
			h.nextTrack++
			track := h.nextTrack
			e.Track = &track
		}
	}
}

func cloneProjectiles(p *[]Enemy) *[]Enemy {
	if p == nil {
		return nil
	}
	copy := cloneEnemies(*p)
	return &copy
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
		if copy[i].MotionDirection != nil {
			v := *copy[i].MotionDirection
			copy[i].MotionDirection = &v
		}
		if copy[i].ClearShot != nil {
			v := *copy[i].ClearShot
			copy[i].ClearShot = &v
		}
	}
	return copy
}
