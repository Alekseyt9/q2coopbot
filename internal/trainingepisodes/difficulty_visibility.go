package trainingepisodes

import (
	"fmt"
	"math"
	"q2coopbot/internal/quake"
)

const VisibleDifficultyVersion = "combat_visible_resource_difficulty_v2"
const VisibleBeginnerPolicy = "beginner_visible_resources_v2"

func assessVisibleEncounter(v Instance, frames int, world *quake.MapInfo) EncounterDifficulty {
	a := AssessEncounter(v, frames)
	a.Version = VisibleDifficultyVersion
	if !a.EligibleInitialTraining {
		return a
	}
	a.GeometryChecked = true
	eye := v.Player
	eye[2] += 22
	for i, m := range v.Monsters {
		profile, known := monsterProfiles[m.Class]
		visible := known && world != nil && world.HasCollision()
		center, aim := m.Position, m.Position
		center[2] += (profile.Min[2] + profile.Max[2]) / 2
		aim[2] += math.Min(math.Max(22, profile.Min[2]+8), profile.Max[2]-8)
		if visible {
			a1, k1 := world.StaticArea(eye)
			a2, k2 := world.StaticArea(center)
			visible = k1 && k2 && a1 == a2
			if possible, known := world.PointPVS(eye, center); known && !possible {
				visible = false
			}
			visible = visible && world.ClearShot(eye, center) && world.ClearShot(eye, aim) && !world.DoorShotBlocked(nil, eye, center) && !world.DoorShotBlocked(nil, eye, aim)
		}
		if visible {
			a.StartVisibleMonsters++
		} else {
			a.BlockedMonsterIndices = append(a.BlockedMonsterIndices, i)
		}
	}
	if a.StartVisibleMonsters != len(v.Monsters) {
		a.EligibleInitialTraining = false
		a.Tier = "geometry_blocked"
		a.Reasons = append(a.Reasons, "not_all_targets_immediately_engageable")
	}
	return a
}

func assessVisiblePlan(p *Plan, root string) (DifficultyReport, error) {
	r := DifficultyReport{Version: VisibleDifficultyVersion, OutcomeIndependent: true, Counts: map[string]int{}, Cases: []EncounterDifficulty{}}
	worlds := map[string]*quake.MapInfo{}
	for _, t := range p.Tasks {
		for _, v := range t.Instances {
			a := AssessEncounter(v, t.Episode.GameFrames)
			a.Version = VisibleDifficultyVersion
			if a.EligibleInitialTraining {
				world := worlds[t.Episode.Map]
				if world == nil {
					loaded, err := loadGenerationWorld(root, t.Episode.Map)
					if err != nil {
						return r, err
					}
					world = &loaded
					worlds[t.Episode.Map] = world
				}
				a = assessVisibleEncounter(v, t.Episode.GameFrames, world)
			}
			r.Cases = append(r.Cases, a)
			r.Counts[a.Tier]++
		}
	}
	return r, nil
}

func FilterVisibleBeginnerPlan(p *Plan, root string) (DifficultyReport, error) {
	r, err := assessVisiblePlan(p, root)
	if err != nil {
		return r, err
	}
	keepSeed := map[string]map[int]bool{}
	for _, a := range r.Cases {
		if a.EligibleInitialTraining {
			if keepSeed[a.EpisodeID] == nil {
				keepSeed[a.EpisodeID] = map[int]bool{}
			}
			keepSeed[a.EpisodeID][a.Seed] = true
		}
	}
	var tasks []Task
	for _, t := range p.Tasks {
		kept := t
		kept.Seeds = nil
		kept.Instances = nil
		for _, v := range t.Instances {
			if keepSeed[t.Episode.ID][v.EngineSeed] {
				kept.Seeds = append(kept.Seeds, v.EngineSeed)
				kept.Instances = append(kept.Instances, v)
			}
		}
		if len(kept.Seeds) > 0 {
			tasks = append(tasks, kept)
		}
	}
	if len(tasks) == 0 {
		return r, fmt.Errorf("no beginner encounters meet %s", VisibleDifficultyVersion)
	}
	p.Tasks = tasks
	p.DifficultyPolicy = VisibleBeginnerPolicy
	return r, nil
}
