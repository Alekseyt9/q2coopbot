#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_COOP_STATE_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_COOP_STATE_H

#include <stdbool.h>

#include "bot_interface.h"
#include "bot_state.h"

typedef struct aas_entityinfo_s aas_entityinfo_t;

#ifdef __cplusplus
extern "C" {
#endif

int BotAI_CoopPlayerEntityInfo(const bot_client_state_t *state,
	 aas_entityinfo_t *player_info);
int BotAI_CoopPlayerEntity(const bot_client_state_t *state, vec3_t origin);
const char *BotAI_CoopPlayerIntentName(bot_coop_player_intent_t intent);
bool BotAI_CoopIntentIsConfident(const bot_client_state_t *state,
	bot_coop_player_intent_t intent);
bool BotAI_CoopPlayerNeedsRescue(const bot_client_state_t *state);
void BotAI_UpdateCoopPlayerIntent(bot_client_state_t *state);
void BotAI_UpdateCoopPlayerStyle(bot_client_state_t *state);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_COOP_STATE_H */
