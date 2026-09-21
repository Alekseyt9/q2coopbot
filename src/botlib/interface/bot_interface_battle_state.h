#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_BATTLE_STATE_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_BATTLE_STATE_H

#include <stdbool.h>

#include "bot_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

void BotAI_ResetFightNavigation(bot_client_state_t *state,
	bool empty_goal_stack);
int BotAI_EntityVisible(const bot_client_state_t *state, int entity);
int BotAI_CurrentEnemyVisible(const bot_client_state_t *state);
int BotAI_ResolveCurrentEnemy(const bot_client_state_t *state,
	ai_dm_enemy_info_t *enemy);
void BotAI_RecordLastEnemyLocation(bot_client_state_t *state,
	const ai_dm_enemy_info_t *enemy);
void BotAI_EnterFoundEnemy(bot_client_state_t *state, bool nearby_goal);
void BotAI_EnterBattleChase(bot_client_state_t *state);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_BATTLE_STATE_H */
