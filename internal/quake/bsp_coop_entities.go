package quake

// Native non-deathmatch spawning excludes entities disabled at every SP skill.
// Keep individual skill restrictions: current server skill is not exposed here.
// Yamagi clears NOT_COOP but does not use it to inhibit entities in g_spawn.c.
func coopMapEntities(entities []MapEntity) []MapEntity {
	var active []MapEntity
	for _, entity := range entities {
		if entity.Class != "worldspawn" && entity.SpawnFlags&1792 == 1792 {
			continue
		}
		active = append(active, entity)
	}
	return active
}
