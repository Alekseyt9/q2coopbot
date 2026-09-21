#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_BATTLE_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_BATTLE_H

#include <stdbool.h>

#include "bot_interface.h"
#include "botlib/ai_move/bot_move.h"

#ifdef __cplusplus
extern "C" {
#endif

void BotAI_ConfigureBattleCombat(bot_client_state_t *state);
int BotAI_RunBattleChaseMovement(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input);
int BotAI_RunBattleNBGMovement(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input,
	const ai_dm_enemy_info_t *enemy);
int BotAI_RunBattleRetreatMovement(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input,
	const ai_dm_enemy_info_t *enemy,
	const bot_goal_t *retreat_goal);
int BotAI_RunBattleRetreatIdle(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input);

void BotAI_BuildBattleChaseGoal(const bot_client_state_t *state,
	bot_goal_t *goal);
int BotAI_BattleChaseTravelFlags(const bot_client_state_t *state);
int BotAI_BattleRetreatTravelFlags(void);
int BotAI_LongTermGoalTravelFlags(const bot_client_state_t *state);
void BotAI_SelectBattleWeapon(bot_client_state_t *state);
bool BotAI_HandleBlockedMovement(bot_client_state_t *state,
	bot_moveresult_t *result,
	bool allow_activation,
	vec3_t alternate_direction);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_BATTLE_H */
