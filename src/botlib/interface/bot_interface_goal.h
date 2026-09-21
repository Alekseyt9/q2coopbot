#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_GOAL_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_GOAL_H

#include <stdbool.h>

#include "bot_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

int BotAI_ActivateEntityTravelFlags(void);
bool BotAI_TouchingNearbyGoal(bot_client_state_t *state,
	bot_goal_t *goal);
bool BotAI_NearbyGoalReached(bot_client_state_t *state,
	bot_goal_t *goal);
bool BotAI_GetItemLongTermGoal(bot_client_state_t *state,
	bot_goal_t *goal,
	int travel_flags);
bool BotAI_TryLongTermNearbyGoal(bot_client_state_t *state,
	const bot_goal_t *long_term_goal,
	int travel_flags);
int BotAI_TryBattleChaseNearbyGoal(bot_client_state_t *state,
	const bot_goal_t *chase_goal,
	int travel_flags);
bool BotAI_BuildCoopSafeAreaGoal(const bot_client_state_t *state,
	bot_goal_t *goal);
void BotAI_CTFRetreatGoals(bot_client_state_t *state);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_GOAL_H */

