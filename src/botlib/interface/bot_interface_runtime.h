#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_RUNTIME_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_RUNTIME_H

#include "bot_interface.h"
#include "botlib/ai/goal_move_orchestrator.h"
#include "botlib/ai_move/bot_move.h"

#ifdef __cplusplus
extern "C" {
#endif

void BotInterface_ResetFrameQueues(void);
float BotInterface_CurrentFrameTime(void);
void BotInterface_ResetGoalSnapshot(bot_client_state_t *state);
void BotInterface_UpdateGoalSnapshot(bot_client_state_t *state);
const bot_goal_t *BotInterface_FindSnapshotGoal(const bot_client_state_t *state,
	int number);
int BotInterface_RebuildGoalCandidates(bot_client_state_t *state);
float BotInterface_GoalWeight(void *ctx,
	const ai_goal_candidate_t *candidate);
float BotInterface_GoalTravelTime(void *ctx,
	int start_area,
	const ai_goal_candidate_t *candidate);
void BotInterface_GoalNotify(void *ctx,
	const ai_goal_selection_t *selection);
int BotInterface_PrepareMoveState(bot_client_state_t *state, float thinktime);
void BotInterface_ApplyMoveResult(const bot_moveresult_t *result,
	bot_input_t *out_input);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_RUNTIME_H */
