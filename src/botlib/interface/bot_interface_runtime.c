#include <float.h>
#include <string.h>

#include "botlib/aas/aas_local.h"
#include "botlib/aas/aas_sound.h"
#include "botlib/ai_goal/ai_goal.h"
#include "botlib/ai/goal_move_orchestrator.h"
#include "botlib/ai_move/bot_move.h"
#include "bot_interface_runtime.h"
#include "bot_state.h"

void BotInterface_ResetFrameQueues(void)
{
	AAS_SoundSubsystem_ResetFrameEvents();
}

void BotInterface_ResetGoalSnapshot(bot_client_state_t *state)
{
	if (state == NULL)
	{
		return;
	}

	state->goal_snapshot_count = 0;
	memset(state->goal_snapshot, 0, sizeof(state->goal_snapshot));
}

void BotInterface_UpdateGoalSnapshot(bot_client_state_t *state)
{
	if (state == NULL)
	{
		return;
	}

	BotInterface_ResetGoalSnapshot(state);

	if (state->goal_handle <= 0)
	{
		return;
	}

	bot_goal_t goal = {0};
	if (AI_GoalBotlib_GetTopGoal(state->goal_handle, &goal))
	{
		state->goal_snapshot[state->goal_snapshot_count++] = goal;
	}

	if (state->goal_snapshot_count <
		(int)(sizeof(state->goal_snapshot) / sizeof(state->goal_snapshot[0])) &&
		AI_GoalBotlib_GetSecondGoal(state->goal_handle, &goal))
	{
		if (state->goal_snapshot_count == 0 ||
			state->goal_snapshot[0].number != goal.number)
		{
			state->goal_snapshot[state->goal_snapshot_count++] = goal;
		}
	}
}

const bot_goal_t *BotInterface_FindSnapshotGoal(const bot_client_state_t *state,
	int number)
{
	if (state == NULL || state->goal_snapshot_count <= 0)
	{
		return NULL;
	}

	for (int i = 0; i < state->goal_snapshot_count; ++i)
	{
		if (state->goal_snapshot[i].number == number)
		{
			return &state->goal_snapshot[i];
		}
	}

	return NULL;
}

int BotInterface_RebuildGoalCandidates(bot_client_state_t *state)
{
	if (state == NULL || state->goal_state == NULL)
	{
		return BLERR_INVALIDIMPORT;
	}

	AI_GoalState_ClearCandidates(state->goal_state);

	for (int index = 0; index < state->goal_snapshot_count; ++index)
	{
		const bot_goal_t *goal = &state->goal_snapshot[index];
		ai_goal_candidate_t candidate = {0};
		candidate.item_index = goal->number;
		candidate.area = goal->areanum;
		candidate.travel_flags = TFL_DEFAULT;
		VectorCopy(goal->origin, candidate.origin);

		int start_area = AI_GoalState_GetCurrentArea(state->goal_state);
		int travel_time = 0;
		float weight = BotGoal_EvaluateStackGoal(state->goal_handle,
			goal,
			state->last_client_update.origin,
			start_area,
			state->last_client_update.inventory,
			candidate.travel_flags,
			&travel_time);
		if (weight <= -FLT_MAX)
		{
			continue;
		}

		candidate.base_weight = weight;
		AI_GoalState_AddCandidate(state->goal_state, &candidate);
	}

	return BLERR_NOERROR;
}

float BotInterface_GoalWeight(void *ctx,
	const ai_goal_candidate_t *candidate)
{
	bot_client_state_t *state = (bot_client_state_t *)ctx;
	if (state == NULL || candidate == NULL)
	{
		return 0.0f;
	}

	const bot_goal_t *goal = BotInterface_FindSnapshotGoal(state,
		candidate->item_index);
	if (goal == NULL)
	{
		return candidate->base_weight;
	}

	int start_area = AI_GoalState_GetCurrentArea(state->goal_state);
	int travel_time = 0;
	float weight = BotGoal_EvaluateStackGoal(state->goal_handle,
		goal,
		state->last_client_update.origin,
		start_area,
		state->last_client_update.inventory,
		candidate->travel_flags,
		&travel_time);
	if (weight <= -FLT_MAX)
	{
		return candidate->base_weight;
	}

	return weight;
}

float BotInterface_GoalTravelTime(void *ctx,
	int start_area,
	const ai_goal_candidate_t *candidate)
{
	bot_client_state_t *state = (bot_client_state_t *)ctx;
	if (state == NULL || candidate == NULL)
	{
		return 0.0f;
	}

	const bot_goal_t *goal = BotInterface_FindSnapshotGoal(state,
		candidate->item_index);
	if (goal == NULL)
	{
		return -1.0f;
	}

	if (start_area <= 0 || goal->areanum <= 0)
	{
		int travel_time = 0;
		BotGoal_EvaluateStackGoal(state->goal_handle,
			goal,
			state->last_client_update.origin,
			start_area,
			state->last_client_update.inventory,
			candidate->travel_flags,
			&travel_time);
		return (float)travel_time;
	}

	vec3_t origin;
	VectorCopy(state->last_client_update.origin, origin);
	int travel = AAS_AreaTravelTimeToGoalArea(start_area,
		origin,
		goal->areanum,
		candidate->travel_flags);
	return (float)travel;
}

void BotInterface_GoalNotify(void *ctx,
	const ai_goal_selection_t *selection)
{
	bot_client_state_t *state = (bot_client_state_t *)ctx;
	if (state == NULL)
	{
		return;
	}

	if (selection == NULL || !selection->valid)
	{
		state->active_goal_number = 0;
		return;
	}

	state->active_goal_number = selection->candidate.item_index;
}

int BotInterface_PrepareMoveState(bot_client_state_t *state, float thinktime)
{
	if (state == NULL || state->move_handle <= 0)
	{
		return BLERR_INVALIDIMPORT;
	}

	bot_initmove_t init = {0};
	VectorCopy(state->last_client_update.origin, init.origin);
	VectorCopy(state->last_client_update.velocity, init.velocity);
	VectorCopy(state->last_client_update.viewoffset, init.viewoffset);
	init.entitynum = state->entity_number;
	init.client = state->client_number;
	init.thinktime = thinktime;
	init.presencetype = (state->last_client_update.pm_flags & PMF_DUCKED) ?
		PRESENCE_CROUCH : PRESENCE_NORMAL;

	if (state->last_client_update.pm_flags & PMF_ON_GROUND)
	{
		init.or_moveflags |= MFL_ONGROUND;
	}
	if ((state->last_client_update.pm_flags & PMF_TIME_TELEPORT) &&
		state->last_client_update.pm_time > 0)
	{
		init.or_moveflags |= MFL_TELEPORTED;
	}
	if ((state->last_client_update.pm_flags & PMF_TIME_WATERJUMP) &&
		state->last_client_update.pm_time > 0)
	{
		init.or_moveflags |= MFL_WATERJUMP;
	}

	VectorCopy(state->last_client_update.viewangles, init.viewangles);
	BotInitMoveStateHandle(state->move_handle, &init);

	bot_movestate_t *move_state = BotMoveStateFromHandle(state->move_handle);
	if (move_state != NULL && state->goal_state != NULL)
	{
		AI_GoalState_SetCurrentArea(state->goal_state, move_state->areanum);
	}

	return BLERR_NOERROR;
}

void BotInterface_ApplyMoveResult(const bot_moveresult_t *result,
	bot_input_t *out_input)
{
	if (result == NULL || out_input == NULL)
	{
		return;
	}

	if (result->failure)
	{
		VectorClear(out_input->dir);
		out_input->speed = 0.0f;
	}
}
