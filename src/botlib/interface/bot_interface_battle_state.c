#include <math.h>

#include "botlib/aas/aas_local.h"
#include "botlib/ai/ai_dm.h"
#include "botlib/ai_goal/ai_goal.h"
#include "bot_interface_battle_state.h"
#include "bot_interface_combat.h"
#include "bot_state.h"

void BotAI_EnterNode(bot_client_state_t *state, int node);

/*
=============
BotAI_ResetFightNavigation

Clears the retained movement avoid and goal stack before a non-retreat fight.
=============
*/
void BotAI_ResetFightNavigation(bot_client_state_t *state,
	bool empty_goal_stack)
{
	if (state == NULL)
	{
		return;
	}

	if (state->move_handle > 0)
	{
		BotResetLastAvoidReachHandle(state->move_handle);
	}
	if (empty_goal_stack && state->goal_handle > 0)
	{
		AI_GoalBotlib_EmptyGoalStack(state->goal_handle);
	}
}

/*
=============
BotAI_EntityVisible

Tests a client entity through retail's direct 360-degree long-term-goal
visibility path.
=============
*/
int BotAI_EntityVisible(const bot_client_state_t *state, int entity)
{
	if (state == NULL || entity <= 0 || entity >= aasworld.maxEntities ||
		aasworld.entities == NULL)
	{
		return qfalse;
	}

	vec3_t eye;
	BotInterface_ClientEyePosition(state, eye);
	vec3_t viewangles;
	if (!AI_DMState_GetViewAngles(state->dm_state, viewangles))
	{
		VectorClear(viewangles);
	}

	return AAS_EntityVisible(state->entity_number,
		eye,
		viewangles,
		360.0f,
		entity);
}

/*
=============
BotAI_CurrentEnemyVisible

Tests the retained enemy through retail's direct 360-degree chase visibility.
=============
*/
int BotAI_CurrentEnemyVisible(const bot_client_state_t *state)
{
	if (state == NULL)
	{
		return qfalse;
	}

	return BotAI_EntityVisible(state, state->combat.current_enemy);
}

/*
=============
BotAI_ResolveCurrentEnemy

Builds the DM handoff for the persistent enemy without running BotFindEnemy.
=============
*/
int BotAI_ResolveCurrentEnemy(const bot_client_state_t *state,
	ai_dm_enemy_info_t *enemy)
{
	BotAI_InitEnemyInfo(enemy);
	if (state == NULL || enemy == NULL ||
		state->combat.current_enemy <= 0 ||
		state->combat.current_enemy >= aasworld.maxEntities ||
		aasworld.entities == NULL ||
		!aasworld.entities[state->combat.current_enemy].inuse)
	{
		return qfalse;
	}

	aas_entityinfo_t entity_info;
	AAS_EntityInfo(state->combat.current_enemy, &entity_info);
	if (BotAI_EntityIsDead(&entity_info))
	{
		return qfalse;
	}

	enemy->valid = true;
	enemy->visible = BotAI_CurrentEnemyVisible(state) != 0;
	enemy->entity = entity_info.number;
	VectorCopy(entity_info.origin, enemy->origin);
	VectorSubtract(entity_info.origin, entity_info.old_origin, enemy->velocity);
	VectorCopy(entity_info.lastvisorigin, enemy->lastvisorigin);
	enemy->update_time = entity_info.update_time;
	vec3_t direction;
	VectorSubtract(entity_info.origin,
		state->last_client_update.origin,
		direction);
	enemy->distance = sqrtf(DotProduct(direction, direction));
	enemy->last_seen_time = enemy->visible
		? AAS_Time()
		: state->combat.enemy_last_seen_time;
	enemy->field_of_view = 360.0f;
	enemy->is_shooting = BotAI_EntityIsShooting(&entity_info) != 0;
	enemy->in_field_of_view = enemy->visible;
	enemy->has_line_of_sight = enemy->visible;
	return qtrue;
}

/*
=============
BotAI_RecordLastEnemyLocation

Stores the reachable enemy area and origin consumed by Battle Chase.
=============
*/
void BotAI_RecordLastEnemyLocation(bot_client_state_t *state,
	const ai_dm_enemy_info_t *enemy)
{
	if (state == NULL || enemy == NULL || !enemy->valid)
	{
		return;
	}

	if (aasworld.loaded)
	{
		int area = AAS_PointAreaNum(enemy->origin);
		if (area != 0 && AAS_AreaReachability(area) != 0)
		{
			state->combat.last_enemy_area = area;
			VectorCopy(enemy->origin, state->combat.last_enemy_origin);
		}
	}
}

/*
=============
BotAI_EnterFoundEnemy

Applies the caller-specific retreat and fight transition used after a scan.
=============
*/
void BotAI_EnterFoundEnemy(bot_client_state_t *state, bool nearby_goal)
{
	if (BotAI_WantsToRetreat(state))
	{
		state->ai_node = nearby_goal
			? BOT_AI_NODE_BATTLE_NBG
			: BOT_AI_NODE_BATTLE_RETREAT;
		return;
	}

	BotAI_ResetFightNavigation(state, true);
	BotAI_EnterNode(state, BOT_AI_NODE_BATTLE_FIGHT);
}

/*
=============
BotAI_EnterBattleChase

Mirrors Battle Chase entry's independent ten-second deadline. The deadline
begins when Fight loses visual contact, not when the enemy was last visible.
=============
*/
void BotAI_EnterBattleChase(bot_client_state_t *state)
{
	if (state == NULL)
	{
		return;
	}

	state->combat.chase_time = AAS_Time() + 10.0f;
	if (state->dm_state != NULL)
	{
		AI_DMState_SetChaseDeadline(state->dm_state,
			state->combat.chase_time);
	}
	BotAI_EnterNode(state, BOT_AI_NODE_BATTLE_CHASE);
}
