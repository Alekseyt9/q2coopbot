package trainingepisodes

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"q2coopbot/internal/quake"
	"reflect"
)

type PositionRange struct {
	Min quake.Vec3 `json:"min"`
	Max quake.Vec3 `json:"max"`
}
type StartDistribution struct {
	Player  PositionRange  `json:"player"`
	Primary PositionRange  `json:"primary"`
	Flank   *PositionRange `json:"flank,omitempty"`
	Health  []int          `json:"health"`
}
type Generator struct {
	Version       int                          `json:"version"`
	Kind          string                       `json:"kind"`
	Distributions map[string]StartDistribution `json:"distributions"`
}
type GeneratedMonster struct {
	Class    string     `json:"class"`
	Position quake.Vec3 `json:"position"`
}
type Instance struct {
	Version        int                `json:"version"`
	EpisodeID      string             `json:"episode_id"`
	Split          string             `json:"split"`
	EngineSeed     int                `json:"engine_seed"`
	GenerationSeed int                `json:"generation_seed"`
	Attempts       int                `json:"attempts"`
	Rejections     map[string]int     `json:"rejections"`
	Player         quake.Vec3         `json:"player"`
	Health         int                `json:"health"`
	Monsters       []GeneratedMonster `json:"monsters"`
	Loadout        string             `json:"loadout"`
	Skill          int                `json:"skill"`
	GeometrySource string             `json:"geometry_source"`
	GeometrySHA256 string             `json:"geometry_sha256"`
}

func (g *Generator) validate(ep Episode) error {
	if g.Version != 1 || g.Kind != "base1-ground-combat-v1" || ep.Recipe.Runner != "combat-baseline" || len(g.Distributions) != 4 {
		return fmt.Errorf("unsupported generator binding")
	}
	if len(ep.Monsters) == 0 || (!ep.Recipe.Mixed && len(ep.Monsters) != 1) || (ep.Recipe.Mixed && !reflect.DeepEqual(ep.Monsters, []string{"monster_parasite", "monster_gunner"})) {
		return fmt.Errorf("generator composition differs from recipe")
	}
	switch ep.Monsters[0] {
	case "monster_parasite", "monster_soldier", "monster_infantry":
	default:
		return fmt.Errorf("unsupported primary monster")
	}
	for _, split := range splitNames {
		d, ok := g.Distributions[split]
		if !ok || len(d.Health) == 0 || (d.Flank != nil) != ep.Recipe.Mixed {
			return fmt.Errorf("incomplete generator distribution %s", split)
		}
		for _, h := range d.Health {
			if h < 1 || h > 100 {
				return fmt.Errorf("invalid generated player health")
			}
		}
		boxes := []PositionRange{d.Player, d.Primary}
		if d.Flank != nil {
			boxes = append(boxes, *d.Flank)
		}
		for _, b := range boxes {
			for axis := range b.Min {
				if math.IsNaN(b.Min[axis]) || math.IsNaN(b.Max[axis]) || math.IsInf(b.Min[axis], 0) || math.IsInf(b.Max[axis], 0) || b.Min[axis] > b.Max[axis] || math.Abs(b.Min[axis]) > 8192 || math.Abs(b.Max[axis]) > 8192 {
					return fmt.Errorf("invalid position distribution")
				}
			}
		}
	}
	for _, split := range splitNames[1:] {
		if overlaps(g.Distributions["train"].Primary, g.Distributions[split].Primary) {
			return fmt.Errorf("training and %s condition domains overlap", split)
		}
	}
	return nil
}
func overlaps(a, b PositionRange) bool {
	for axis := range a.Min {
		if a.Max[axis] < b.Min[axis] || b.Max[axis] < a.Min[axis] {
			return false
		}
	}
	return true
}
func samplePosition(r *rand.Rand, b PositionRange) quake.Vec3 {
	var p quake.Vec3
	for axis := range p {
		p[axis] = math.Round((b.Min[axis]+r.Float64()*(b.Max[axis]-b.Min[axis]))*8) / 8
	}
	return p
}
func loadGenerationWorld(root string) (quake.MapInfo, error) {
	return quake.LoadMap(filepath.Join(root, "workspace", "runtime", "q2go", "baseq2"), "base1")
}

func generate(ep Episode, split string, seed int, world *quake.MapInfo) (Instance, error) {
	sum := sha256.Sum256([]byte(fmt.Sprintf("combat-generator-v1:%s:%d:%s:%d", ep.ID, ep.Revision, split, seed)))
	gs := int(binary.LittleEndian.Uint32(sum[:4]) & 0x7fffffff)
	r := rand.New(rand.NewSource(int64(gs)))
	d := ep.Generator.Distributions[split]
	v := Instance{Version: 1, EpisodeID: ep.ID, Split: split, EngineSeed: seed, GenerationSeed: gs, Loadout: ep.Recipe.Loadout, Skill: ep.Skill, GeometrySource: world.BSPSource, GeometrySHA256: world.BSPSHA256, Rejections: map[string]int{}}
	for attempt := 1; attempt <= 128; attempt++ {
		v.Attempts = attempt
		v.Player = samplePosition(r, d.Player)
		v.Health = d.Health[r.Intn(len(d.Health))]
		v.Monsters = []GeneratedMonster{{ep.Monsters[0], samplePosition(r, d.Primary)}}
		if d.Flank != nil {
			v.Monsters = append(v.Monsters, GeneratedMonster{"monster_gunner", samplePosition(r, *d.Flank)})
		}
		if reason := checkStart(v, world); reason != "" {
			v.Rejections[reason]++
			continue
		}
		return v, nil
	}
	return v, fmt.Errorf("generator exhausted attempts for %s seed %d: %v", ep.ID, seed, v.Rejections)
}
func checkStart(v Instance, world *quake.MapInfo) string {
	points := []quake.Vec3{v.Player}
	for _, m := range v.Monsters {
		points = append(points, m.Position)
	}
	for i, p := range points {
		lo, hi := p, p
		for axis := 0; axis < 2; axis++ {
			lo[axis] -= 16
			hi[axis] += 16
		}
		lo[2] -= 24
		hi[2] += 32
		clear, valid := world.ProjectileBoxClear(lo, hi)
		if !valid || !clear {
			return "static_hull_collision"
		}
		if drop, ok := world.GroundDrop(p, 2); !ok || drop > 1 {
			return "unsupported_floor"
		}
		if i == 0 {
			hi[2] += 10
			clear, valid = world.ProjectileBoxClear(lo, hi)
			if !valid || !clear {
				return "teleport_headroom"
			}
		}
		for j := 0; j < i; j++ {
			if math.Abs(p[0]-points[j][0]) < 40 && math.Abs(p[1]-points[j][1]) < 40 && math.Abs(p[2]-points[j][2]) < 64 {
				return "actor_overlap"
			}
		}
		// Startup entity-lump monsters exist before the player is teleported.
		// Reserve native spawn hulls to prevent a setup telefrag.
		if i >= 2 {
			for _, spawn := range world.Entities {
				if (spawn.Class == "info_player_start" || spawn.Class == "info_player_coop") && math.Abs(p[0]-spawn.Origin[0]) < 40 && math.Abs(p[1]-spawn.Origin[1]) < 40 && math.Abs(p[2]-spawn.Origin[2]) < 64 {
					return "native_spawn_overlap"
				}
			}
		}
	}
	// The current reset exporter requires a visible primary target.
	eye, target := v.Player, v.Monsters[0].Position
	eye[2] += 22
	target[2] += 8
	tr := world.TraceProjectile(eye, target)
	if !tr.Valid || tr.StartSolid || tr.Fraction < 1 {
		return "primary_occluded"
	}
	return ""
}

// Recompute conditions before execution to detect modified plans.
func VerifyPlan(path, root string) error {
	var p Plan
	if err := read(path, &p); err != nil {
		return err
	}
	r, err := Load(p.RegistryPath)
	if err != nil {
		return err
	}
	if p.Version != 1 || p.Workers != 4 || p.RegistrySHA256 != r.SHA256 || len(p.Tasks) == 0 {
		return fmt.Errorf("invalid or stale plan")
	}
	if p.ModelPath != "" {
		h, err := Hash(p.ModelPath)
		if err != nil || h != p.ModelSHA256 {
			return fmt.Errorf("model changed")
		}
	}
	seen := map[string]bool{}
	for _, t := range p.Tasks {
		if len(t.Seeds) == 0 || len(t.Modes) == 0 || seen[t.Episode.ID] {
			return fmt.Errorf("empty or duplicate cohort")
		}
		seen[t.Episode.ID] = true
		mode := t.Modes[0]
		if len(t.Modes) == 2 {
			mode = "both"
		} else if len(t.Modes) != 1 {
			return fmt.Errorf("invalid modes")
		}
		start, ok := t.Episode.Splits[t.Split]
		if !ok {
			return fmt.Errorf("invalid split")
		}
		model := p.ModelPath
		if mode == "rules" {
			model = ""
		}
		expected, err := Build(r, root, []string{t.Episode.ID}, t.Split, mode, model, p.OutputRoot, len(t.Seeds), t.Seeds[0]-start.Start)
		if err != nil {
			return err
		}
		if len(expected.Tasks) != 1 || !reflect.DeepEqual(t, expected.Tasks[0]) {
			return fmt.Errorf("plan conditions or bindings changed: %s", t.Episode.ID)
		}
	}
	return nil
}
