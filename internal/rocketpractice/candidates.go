// Package rocketpractice prepares geometry-only learning diagnostics.
// It does not choose or execute an action and is not called by the live bot.
package rocketpractice

import (
	"fmt"
	"math"
	"sort"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

const Version = "combat_observed_rocket_surface_candidates_v1"
const MaxCandidates = 128

type Exposure struct {
	EntityID       int     `json:"entity_id"`
	OriginDistance float64 `json:"origin_distance"`
	StaticRayClear bool    `json:"static_ray_clear_to_origin"`
}

type Candidate struct {
	Kind           string     `json:"kind"`
	References     []int      `json:"reference_entity_ids"`
	Point          quake.Vec3 `json:"point"`
	Normal         quake.Vec3 `json:"surface_normal"`
	SelfDistance   float64    `json:"self_origin_distance"`
	EyeRayClear    bool       `json:"static_eye_ray_clear"`
	ObservedNearby []Exposure `json:"observed_enemy_geometry"`
}

type anchor struct {
	point quake.Vec3
	ids   []int
}

func finite(v quake.Vec3) bool {
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) || math.Abs(x) > 1e6 {
			return false
		}
	}
	return true
}

func rayClear(world *quake.MapInfo, from, to quake.Vec3) bool {
	t := world.TraceProjectile(from, to)
	return t.Valid && !t.StartSolid && !t.Sky && t.Fraction == 1
}

// Generate uses current visible client positions and static BSP only. Distances
// and rays are not splash damage predictions, dynamic visibility or safety proofs.
func Generate(o policy.Observation, world *quake.MapInfo) ([]Candidate, error) {
	if world == nil || !world.HasCollision() || o.Version != policy.ObservationVersion || o.Identity.Map != world.Name || !finite(o.Position) || o.AgeMS < 0 || o.AgeMS > 300 || len(o.Enemies) > 8 {
		return nil, fmt.Errorf("invalid observed rocket geometry input")
	}
	result := []Candidate{}
	if o.Health <= 0 {
		return result, nil
	}
	enemies := append([]policy.Enemy(nil), o.Enemies...)
	sort.Slice(enemies, func(i, j int) bool { return enemies[i].ID < enemies[j].ID })
	seenIDs := map[int]bool{}
	visible := []policy.Enemy{}
	positions := []quake.Vec3{}
	for _, e := range enemies {
		if e.ID <= 0 || seenIDs[e.ID] || !finite(e.Relative) || math.IsNaN(e.Distance) || math.IsInf(e.Distance, 0) || e.Distance < 0 {
			return nil, fmt.Errorf("invalid observed enemy")
		}
		seenIDs[e.ID] = true
		if e.ClearShot == nil || !*e.ClearShot {
			continue
		}
		p := quake.Vec3{o.Position[0] + e.Relative[0], o.Position[1] + e.Relative[1], o.Position[2] + e.Relative[2]}
		visible = append(visible, e)
		positions = append(positions, p)
	}
	if len(visible) == 0 {
		return result, nil
	}
	eye := o.Position
	eye[2] += 22
	if o.Ducked {
		eye[2] = o.Position[2] - 2
	}
	seen := map[string]bool{}
	add := func(kind string, a anchor, point, normal quake.Vec3) {
		key := fmt.Sprintf("%s:%.0f:%.0f:%.0f", kind, point[0]*4, point[1]*4, point[2]*4)
		if seen[key] || len(result) >= MaxCandidates {
			return
		}
		seen[key] = true
		c := Candidate{Kind: kind, References: append([]int(nil), a.ids...), Point: point, Normal: normal, SelfDistance: quake.Distance(o.Position, point), EyeRayClear: rayClear(world, eye, point), ObservedNearby: []Exposure{}}
		for i, e := range visible {
			c.ObservedNearby = append(c.ObservedNearby, Exposure{e.ID, quake.Distance(point, positions[i]), rayClear(world, point, positions[i])})
		}
		result = append(result, c)
	}
	anchors := []anchor{}
	anchorKeys := map[string]bool{}
	addAnchor := func(a anchor) {
		key := fmt.Sprintf("%.0f:%.0f:%.0f", a.point[0]*4, a.point[1]*4, a.point[2]*4)
		if !anchorKeys[key] && len(anchors) < 16 {
			anchorKeys[key] = true
			anchors = append(anchors, a)
		}
	}
	if len(positions) > 1 {
		center := quake.Vec3{}
		ids := []int{}
		for i, p := range positions {
			for axis := range center {
				center[axis] += p[axis] / float64(len(positions))
			}
			ids = append(ids, visible[i].ID)
		}
		addAnchor(anchor{center, ids})
	}
	for i, e := range visible {
		a := anchor{positions[i], []int{e.ID}}
		addAnchor(a)
		_, known, err := policy.ObservedAimDirection(o, e)
		if err != nil {
			return nil, err
		}
		if known {
			add("body_reference", a, (quake.Object{Origin: positions[i], Solid: *e.Solid}).AimPoint(), quake.Vec3{})
		}
	}
	type pair struct {
		i, j     int
		distance float64
	}
	pairs := []pair{}
	for i := range positions {
		for j := i + 1; j < len(positions); j++ {
			pairs = append(pairs, pair{i, j, quake.Distance(positions[i], positions[j])})
		}
	}
	sort.SliceStable(pairs, func(i, j int) bool { return pairs[i].distance < pairs[j].distance })
	for _, p := range pairs {
		center := quake.Vec3{}
		for axis := range center {
			center[axis] = (positions[p.i][axis] + positions[p.j][axis]) / 2
		}
		addAnchor(anchor{center, []int{visible[p.i].ID, visible[p.j].ID}})
	}
	for _, a := range anchors {
		from, to := a.point, a.point
		from[2] += 8
		to[2] -= 256
		tr := world.TraceProjectile(from, to)
		if tr.Valid && !tr.StartSolid && !tr.Sky && tr.Fraction < 1 && tr.Normal[2] > .5 {
			point := tr.End
			for axis := range point {
				point[axis] += tr.Normal[axis]
			}
			add("floor", a, point, tr.Normal)
		}
	}
	// Interleave directions across anchors so the cap does not favor one enemy.
	for d := 0; d < 8; d++ {
		for _, a := range anchors {
			from, to := a.point, a.point
			from[2] += 8
			to[2] += 8
			to[0] += 128 * math.Cos(float64(d)*math.Pi/4)
			to[1] += 128 * math.Sin(float64(d)*math.Pi/4)
			tr := world.TraceProjectile(from, to)
			if tr.Valid && !tr.StartSolid && !tr.Sky && tr.Fraction < 1 && math.Abs(tr.Normal[2]) < .5 {
				point := tr.End
				for axis := range point {
					point[axis] += tr.Normal[axis]
				}
				add("wall", a, point, tr.Normal)
			}
		}
	}
	return result, nil
}
