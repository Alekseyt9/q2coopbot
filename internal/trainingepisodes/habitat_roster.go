package trainingepisodes

import (
	"fmt"
	"math"
	"reflect"
)

const nativeHabitatRosterKind = "campaign-native-habitat-roster-v1"

func habitatRosterClasses(d MonsterDistribution) []string {
	classes := []string{d.PrimaryClass}
	for _, class := range d.ClassPool {
		if class != d.PrimaryClass {
			classes = append(classes, class)
		}
	}
	return classes
}

func (d MonsterDistribution) validateHabitat() error {
	if d.Version != 2 || d.MinCount < 1 || d.MaxCount > 4 || d.MinCount > d.MaxCount {
		return fmt.Errorf("invalid habitat roster counts")
	}
	if _, ok := monsterProfiles[d.PrimaryClass]; !ok {
		return fmt.Errorf("unsupported primary profile")
	}
	if len(d.ClassPool) == 0 && d.MaxCount != 1 {
		return fmt.Errorf("missing additional native class pool")
	}
	if (d.MaxCount > 1 && d.MinCount < 2) || (d.MaxCount == 1 && len(d.ClassPool) > 0) {
		return fmt.Errorf("ambiguous solo/group habitat contract")
	}
	seen := map[string]bool{}
	for _, class := range d.ClassPool {
		if _, ok := monsterProfiles[class]; !ok || seen[class] {
			return fmt.Errorf("invalid additional profile pool")
		}
		seen[class] = true
	}
	return nil
}

func (g *Generator) validateHabitatRoster(ep Episode) error {
	if g.Version != 1 || g.Roster == nil || g.Roster.validateHabitat() != nil || g.Inventory == nil || g.Inventory.validate() != nil || ep.Recipe.Runner != "combat-baseline" || ep.Recipe.Loadout != "weapons" || ep.Recipe.Mixed != (g.Roster.MaxCount > 1) || !reflect.DeepEqual(ep.Monsters, habitatRosterClasses(*g.Roster)) {
		return fmt.Errorf("invalid native habitat roster contract")
	}
	if g.SeedRevision != nil && (*g.SeedRevision < 1 || *g.SeedRevision > ep.Revision) {
		return fmt.Errorf("invalid habitat sampling revision")
	}
	if g.Site == nil || g.Site.Map != ep.Map || len(g.Site.BSPSHA256) != 64 || len(g.Site.WallDistances) != 8 || len(g.Distributions) != 4 {
		return fmt.Errorf("missing habitat geometry binding")
	}
	for _, split := range splitNames {
		d, ok := g.Distributions[split]
		if !ok || len(d.Health) == 0 || d.Flank != nil || len(d.Additional) > 0 || len(d.AdditionalByClass) != len(g.Roster.ClassPool) {
			return fmt.Errorf("incomplete habitat placement distribution")
		}
		boxes := []PositionRange{d.Player, d.Primary}
		for _, class := range g.Roster.ClassPool {
			positions := d.AdditionalByClass[class]
			if len(positions) == 0 || len(positions) > 16 {
				return fmt.Errorf("missing or excessive profile placements")
			}
			boxes = append(boxes, positions...)
		}
		for _, hp := range d.Health {
			if hp < 1 || hp > 100 {
				return fmt.Errorf("invalid habitat player health")
			}
		}
		for _, box := range boxes {
			for axis, x := range box.Min {
				y := box.Max[axis]
				if math.IsNaN(x) || math.IsNaN(y) || math.IsInf(x, 0) || math.IsInf(y, 0) || x > y || math.Abs(x) > 8192 || math.Abs(y) > 8192 {
					return fmt.Errorf("invalid habitat position bounds")
				}
			}
		}
	}
	for _, split := range splitNames[1:] {
		if overlaps(g.Distributions["train"].Primary, g.Distributions[split].Primary) {
			return fmt.Errorf("habitat train and evaluation primary domains overlap")
		}
	}
	return nil
}
