package trainingepisodes

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
)

const BulletPacks50Profile = "combat_bullet_packs_50_v1"
const AmmoBoundaryProfile = "combat_ammo_boundary_v1"
const RocketPracticeProfile = "combat_rocket_splash_practice_v1"

// The empty profile reproduces historical conditions. Profiles affect initial
// generated inventories only, not pickups, runtime weapon choices or rewards.
func ApplyInventoryProfile(p *Plan, profile string) error {
	if profile != "" && profile != BulletPacks50Profile && profile != AmmoBoundaryProfile && profile != RocketPracticeProfile {
		return fmt.Errorf("unknown inventory profile %q", profile)
	}
	if p.InventoryProfile != "" {
		return fmt.Errorf("inventory profile already applied")
	}
	for _, task := range p.Tasks {
		if profile == AmmoBoundaryProfile && task.Split == "train" {
			return fmt.Errorf("ammo-boundary profile is evaluation-only")
		}
		if profile == RocketPracticeProfile {
			if len(task.Instances) == 0 {
				return fmt.Errorf("rocket practice requires generated inventory")
			}
			for _, v := range task.Instances {
				if v.Inventory == nil {
					return fmt.Errorf("rocket practice requires generated inventory")
				}
			}
		}
	}
	if profile == BulletPacks50Profile || profile == RocketPracticeProfile {
		for ti := range p.Tasks {
			for ii := range p.Tasks[ti].Instances {
				v := &p.Tasks[ti].Instances[ii]
				if v.Inventory == nil {
					if profile == RocketPracticeProfile {
						return fmt.Errorf("rocket practice requires generated inventory")
					}
					continue
				}
				// Keep zero/nonzero availability and the original sampling stream.
				// Only the positive quantity changes, using an independent seed hash.
				digest := sha256.Sum256([]byte(fmt.Sprintf("combat-bullet-packs-50-v1:%d", v.EngineSeed)))
				if profile == RocketPracticeProfile {
					digest = sha256.Sum256([]byte(fmt.Sprintf("combat-rocket-practice-v1:%d", v.EngineSeed)))
				}
				quantity := 50 * (1 + int(binary.LittleEndian.Uint32(digest[:4])%4))
				inventory := *v.Inventory
				inventory.Items = append([]InventoryCount(nil), inventory.Items...)
				if profile == RocketPracticeProfile {
					inventory.RequestedWeapons = []string{"Rocket Launcher"}
					inventory.InitialWeapon = "Rocket Launcher"
				}
				for i := range inventory.Items {
					if profile == RocketPracticeProfile {
						switch inventory.Items[i].Name {
						case "Blaster", "Rocket Launcher":
							inventory.Items[i].Count = 1
						case "Rockets":
							inventory.Items[i].Count = []int{5, 10, 20}[binary.LittleEndian.Uint32(digest[4:8])%3]
						default:
							inventory.Items[i].Count = 0
						}
						continue
					}
					if inventory.Items[i].Name == "Bullets" && inventory.Items[i].Count > 0 {
						inventory.Items[i].Count = quantity
					}
				}
				v.Inventory = &inventory
			}
		}
	}
	p.InventoryProfile = profile
	return nil
}

func BuildWithInventoryProfile(r *Registry, root string, ids []string, split, mode, model, out string, count, offset int, profile string) (*Plan, error) {
	p, err := Build(r, root, ids, split, mode, model, out, count, offset)
	if err != nil {
		return nil, err
	}
	if err := ApplyInventoryProfile(p, profile); err != nil {
		return nil, err
	}
	return p, nil
}
