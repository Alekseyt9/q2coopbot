package trainingepisodes

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math/rand"
)

const nativeRosterKind = "campaign-native-ground-roster-v1"

// The pool is exact map-native classes; sampling is with replacement.
// Its independent RNG must not consume position or inventory randomness.
type MonsterDistribution struct {
	PrimaryClass string   `json:"primary_class,omitempty"`
	Version      int      `json:"version"`
	MinCount     int      `json:"min_count"`
	MaxCount     int      `json:"max_count"`
	ClassPool    []string `json:"class_pool"`
}

func nativeRosterClass(class string) bool {
	switch class {
	case "monster_infantry", "monster_parasite", "monster_soldier", "monster_soldier_light", "monster_soldier_ss", "monster_gunner", "monster_berserk", "monster_brain":
		return true
	}
	return false
}

func (d MonsterDistribution) validate() error {
	if d.Version != 1 || d.PrimaryClass != "" || d.MinCount < 2 || d.MaxCount > 4 || d.MinCount > d.MaxCount || len(d.ClassPool) == 0 {
		return fmt.Errorf("unsupported native roster distribution")
	}
	seen := map[string]bool{}
	for _, class := range d.ClassPool {
		if !nativeRosterClass(class) || seen[class] {
			return fmt.Errorf("unsupported or duplicate native roster pool class %s", class)
		}
		seen[class] = true
	}
	return nil
}

func sampleMonsterRoster(d MonsterDistribution, seed int) []string {
	namespace := "combat-native-roster-v1"
	if d.Version == 2 {
		namespace = "combat-native-habitat-roster-v1"
	}
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s:%d", namespace, seed)))
	r := rand.New(rand.NewSource(int64(binary.LittleEndian.Uint64(sum[:8]))))
	classes := make([]string, d.MinCount+r.Intn(d.MaxCount-d.MinCount+1))
	for i := range classes {
		if d.Version == 2 && i == 0 {
			classes[i] = d.PrimaryClass
			continue
		}
		classes[i] = d.ClassPool[r.Intn(len(d.ClassPool))]
	}
	return classes
}
