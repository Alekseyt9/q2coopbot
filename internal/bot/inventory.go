package bot

import (
	"q2coopbot/internal/quake"
	"strings"
)

type inventoryWatch struct {
	mapName              string
	lastFrame, requested int
	warmupUntil          int
}

func (w *inventoryWatch) command(s quake.Snapshot) string {
	if w.mapName != s.Map || s.Frame < w.lastFrame || s.Health <= 0 {
		*w = inventoryWatch{mapName: s.Map}
	}
	w.lastFrame = s.Frame
	if s.Map == "" || s.Frame <= 0 || s.Health <= 0 || s.InventoryOpen {
		return ""
	}
	// Let initial configstrings and weapon precaches drain before adding the
	// inventory stream to the server's bounded outgoing datagrams.
	if w.warmupUntil == 0 {
		w.warmupUntil = s.Frame + 3
	}
	if s.Frame < w.warmupUntil {
		return ""
	}
	if w.requested != 0 && s.Frame-w.requested < 20 {
		return ""
	}
	w.requested = s.Frame
	return "inven"
}

func stockedWeapon(s quake.Snapshot) string {
	if !s.InventoryKnown || s.InventoryAgeFrames > 20 {
		return "weapnext"
	}
	counts := map[string]int{}
	for _, item := range s.Inventory {
		counts[strings.ToLower(item.Name)] = item.Count
	}
	// Command names are local constants, never interpolated server strings.
	for _, item := range []struct {
		name, ammo, model string
		cost              int
	}{
		{"Railgun", "Slugs", "v_rail", 1}, {"HyperBlaster", "Cells", "v_hyperb", 1},
		{"Chaingun", "Bullets", "v_chain", 1}, {"Machinegun", "Bullets", "v_machn", 1},
		{"Super Shotgun", "Shells", "v_shotg2", 2}, {"Shotgun", "Shells", "v_shotg", 1},
		{"Grenade Launcher", "Grenades", "v_launch", 1}, {"Rocket Launcher", "Rockets", "v_rocket", 1},
		{"BFG10K", "Cells", "v_bfg", 50},
	} {
		if counts[strings.ToLower(item.name)] > 0 && counts[strings.ToLower(item.ammo)] >= item.cost && !strings.Contains(s.Weapon, "/"+item.model+"/") {
			return "use " + item.name
		}
	}
	return "use Blaster"
}
