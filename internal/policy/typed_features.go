package policy

import (
	"fmt"
	"math"
	"strings"
)

// Order is part of the v4 weights contract. Types come from client model paths.
var monsterTypes = []string{"berserk", "boss1", "boss2", "brain", "bitch", "flipper", "float", "flyer", "gladiatr", "gunner", "hover", "infantry", "insane", "medic", "mutant", "parasite", "soldier", "tank", "jorg", "makron", "unknown"}

type MonsterCount struct {
	Type  string `json:"type"`
	Count int    `json:"count"`
}

func MonsterType(class, model string) string {
	if model != "" {
		model = strings.ToLower(strings.ReplaceAll(model, "\\", "/"))
		if model == "models/monsters/boss3/jorg/tris.md2" {
			return "jorg"
		}
		if model == "models/monsters/boss3/rider/tris.md2" {
			return "makron"
		}
		for _, kind := range monsterTypes[:18] {
			if model == "models/monsters/"+kind+"/tris.md2" {
				return kind
			}
		}
		return "unknown"
	}
	kind := strings.TrimPrefix(class, "monster_")
	aliases := map[string]string{"berserker": "berserk", "chick": "bitch", "gladiator": "gladiatr", "floater": "float", "supertank": "boss1", "tank_commander": "tank", "soldier_light": "soldier", "soldier_ss": "soldier"}
	if alias, ok := aliases[kind]; ok {
		kind = alias
	}
	for _, known := range monsterTypes {
		if kind == known {
			return kind
		}
	}
	return "unknown"
}

// Appends 8*40 entity values and 24 group values. Position and vector velocity
// already exist in the unchanged v1 prefix. Facing is distinct from movement.
func typedFeatures(o Observation, enemies []Enemy) ([]float64, error) {
	v := make([]float64, 0, 344)
	yaw := degrees(o.ViewAngles[1]) * math.Pi / 180
	for slot := 0; slot < 8; slot++ {
		if slot >= len(enemies) {
			v = append(v, make([]float64, 40)...)
			continue
		}
		e := enemies[slot]
		kind := MonsterType(e.Class, e.ModelPath)
		for _, known := range monsterTypes {
			x := 0.
			if kind == known {
				x = 1
			}
			v = append(v, x)
		}
		motion := make([]float64, 6)
		if e.Velocity != nil {
			vel := *e.Velocity
			speed := math.Hypot(math.Hypot(vel[0], vel[1]), vel[2])
			motion[0], motion[1] = 1, speed/400
			if speed > 1e-9 {
				motion[2] = 1
				motion[3] = (vel[0]*math.Cos(yaw) + vel[1]*math.Sin(yaw)) / speed
				motion[4] = (-vel[0]*math.Sin(yaw) + vel[1]*math.Cos(yaw)) / speed
				motion[5] = vel[2] / speed
			}
		}
		v = append(v, motion...)
		if e.Angles != nil {
			a := *e.Angles
			ry := a[1]*math.Pi/180 - yaw
			pitch := a[0] * math.Pi / 180
			v = append(v, 1, math.Sin(ry), math.Cos(ry), math.Sin(pitch), math.Cos(pitch))
		} else {
			v = append(v, 0, 0, 0, 0, 0)
		}
		if e.Skin != nil {
			v = append(v, 1, float64(*e.Skin)/8)
		} else {
			v = append(v, 0, 0)
		}
		if e.Animation != nil {
			v = append(v, 1, float64(*e.Animation)/512)
		} else {
			v = append(v, 0, 0)
		}
		box := make([]float64, 4)
		if e.Solid != nil {
			s := *e.Solid
			bottom := -float64((s>>5)&31) * 8
			top := float64((s>>10)&63)*8 - 32
			if s != 0 && s != 31 && s&31 != 0 && top > bottom {
				box = []float64{1, float64(s&31) * 8 / 64, bottom / 64, top / 64}
			}
		}
		v = append(v, box...)
	}
	counts := map[string]int{}
	known := 0.
	if o.Composition != nil {
		known = 1
		for _, item := range *o.Composition {
			valid := false
			for _, kind := range monsterTypes {
				if item.Type == kind {
					valid = true
					break
				}
			}
			if !valid || item.Count < 0 {
				return nil, fmt.Errorf("invalid monster composition")
			}
			if _, duplicate := counts[item.Type]; duplicate {
				return nil, fmt.Errorf("duplicate monster composition")
			}
			counts[item.Type] = item.Count
		}
	} else {
		for _, e := range enemies {
			counts[MonsterType(e.Class, e.ModelPath)]++
		}
	}
	v = append(v, known)
	total := 0
	for _, kind := range monsterTypes {
		total += counts[kind]
		v = append(v, float64(counts[kind])/8)
	}
	overflow := 0.
	if known == 1 && total > 8 {
		overflow = 1
	}
	v = append(v, float64(total)/8, overflow)
	for i, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil, fmt.Errorf("nonfinite typed feature")
		}
		v[i] = math.Max(-4, math.Min(4, x))
	}
	return v, nil
}
