#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_LIFECYCLE_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_LIFECYCLE_H

#include <stdbool.h>

#include "bot_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

void BotAI_ResetRespawnState(bot_client_state_t *state);
void BotAI_CompleteRespawnAction(bot_client_state_t *state);
bool BotAI_ReplyStandActive(bot_client_state_t *state, float thinktime);
int BotAI_RunStand(bot_client_state_t *state, float thinktime);
void BotAI_SetLifecycleStand(bot_client_state_t *state, float duration);
void BotAI_EnterObserver(bot_client_state_t *state);
void BotAI_EnterIntermission(bot_client_state_t *state);
int BotAI_RunLifecycleFrame(bot_client_state_t *state,
	float thinktime,
	bool request_respawn);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_LIFECYCLE_H */
