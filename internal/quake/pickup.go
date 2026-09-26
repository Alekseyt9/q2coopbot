package quake

// Baseq2 health amounts from the world model, not from a guessed entity name.
func healthModelAmount(path string) int {
	switch path {
	case "models/items/healing/stimpack/tris.md2":
		return 2
	case "models/items/healing/medium/tris.md2":
		return 10
	case "models/items/healing/large/tris.md2":
		return 25
	case "models/items/mega_h/tris.md2":
		return 100
	}
	return 0
}

// Baseq2 world models, distinct from held/view weapon models.
var pickupModels = map[string]string{
	"models/weapons/g_shotg/tris.md2":            "weapon_shotgun",
	"models/weapons/g_shotg2/tris.md2":           "weapon_supershotgun",
	"models/weapons/g_machn/tris.md2":            "weapon_machinegun",
	"models/weapons/g_chain/tris.md2":            "weapon_chaingun",
	"models/weapons/g_launch/tris.md2":           "weapon_grenadelauncher",
	"models/weapons/g_rocket/tris.md2":           "weapon_rocketlauncher",
	"models/weapons/g_hyperb/tris.md2":           "weapon_hyperblaster",
	"models/weapons/g_rail/tris.md2":             "weapon_railgun",
	"models/weapons/g_bfg/tris.md2":              "weapon_bfg",
	"models/items/ammo/shells/medium/tris.md2":   "ammo_shells",
	"models/items/ammo/bullets/medium/tris.md2":  "ammo_bullets",
	"models/items/ammo/cells/medium/tris.md2":    "ammo_cells",
	"models/items/ammo/rockets/medium/tris.md2":  "ammo_rockets",
	"models/items/ammo/slugs/medium/tris.md2":    "ammo_slugs",
	"models/items/ammo/grenades/medium/tris.md2": "ammo_grenades",
	"models/items/armor/jacket/tris.md2":         "item_armor_jacket",
	"models/items/armor/combat/tris.md2":         "item_armor_combat",
	"models/items/armor/body/tris.md2":           "item_armor_body",
	"models/items/armor/shard/tris.md2":          "item_armor_shard",
}
