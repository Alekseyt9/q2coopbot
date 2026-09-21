#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_AREA_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_AREA_H

#include <stdbool.h>

#include "bot_state.h"

#ifdef __cplusplus
extern "C" {
#endif

void BotInterface_LogCoopAreaTransition(bot_client_state_t *state,
	const char *actor,
	int from_area,
	int to_area,
	const char *area_state);

const char *BotInterface_CoopAreaStateName(bot_coop_area_state_t state);

bot_coop_area_memory_t *BotInterface_FindCoopAreaMemory(
	bot_client_state_t *state,
	int area,
	bool create);

bool BotInterface_CoopAreaIsDangerous(const bot_client_state_t *state,
	int enemy_count);

void BotInterface_RecordCoopSafeArea(bot_client_state_t *state,
	int area,
	int enemy_count);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_AREA_H */

