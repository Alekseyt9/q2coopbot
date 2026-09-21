#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_MAP_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_MAP_H

#include <stdbool.h>

#include "botlib/aas/aas_local.h"

#ifdef __cplusplus
extern "C" {
#endif

aas_bspentity_t *BotInterface_CoopMapEntities(void);

/* Map-only helpers used by the cooperative objective layer. */
bool BotInterface_MapPointInsideChangelevelTrigger(
	const aas_bspentity_t *entities,
	const aas_bspentity_t *entity,
	const vec3_t point,
	int *model_number);

const aas_bspentity_t *BotInterface_MapFindChangelevelTrigger(
	const aas_bspentity_t *entities,
	const vec3_t bot_origin,
	int *model_number);

void BotInterface_ResetMapModel(void);
void BotInterface_DumpCoopMapModel(const aas_bspentity_t *entities);


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_MAP_H */
