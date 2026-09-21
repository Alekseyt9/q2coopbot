#include <string.h>

#include "botlib/ai/ai_dm.h"
#include "botlib/ai_goal/ai_goal.h"
#include "botlib/ai_move/bot_move.h"
#include "botlib/ai_weight/bot_weight.h"
#include "botlib/aas/aas_local.h"
#include "botlib/common/l_log.h"
#include "botlib/common/l_utils.h"
#include "botlib/ea/ea_local.h"
#include "bot_interface_assets.h"
#include "bot_interface_battle.h"
#include "bot_interface_combat.h"
#include "bot_interface_entity.h"
#include "bot_interface_runtime.h"
#include "bot_state.h"

#define CHARACTERISTIC_ATTACK_SKILL 4

/*
=============
BotAI_BuildBattleChaseGoal

Builds retail Battle Chase's transient eight-unit box at the last reachable
enemy location.  This goal is intentionally independent of the goal stack.
=============
*/
void BotAI_BuildBattleChaseGoal(const bot_client_state_t *state,
	bot_goal_t *goal)
{
	if (state == NULL || goal == NULL)
	{
		return;
	}

	memset(goal, 0, sizeof(*goal));
	goal->entitynum = state->combat.current_enemy;
	goal->areanum = state->combat.last_enemy_area;
	VectorCopy(state->combat.last_enemy_origin, goal->origin);
	VectorSet(goal->mins, -8.0f, -8.0f, -8.0f);
	VectorSet(goal->maxs, 8.0f, 8.0f, 8.0f);
}

/*
 * Retail composes the battle travel mask from two literals: 102334 on its own
 * and 118718 once `usehook` is set, with the rocket-jump bit ORed in
 * afterwards.  Gladiator's `travelflagfortype` table defines exactly fourteen
 * entries, TFL_INVALID through TFL_GRAPPLEHOOK, and this reconstruction uses
 * bit-identical values for all of them, so those are the only mask bits a
 * reachability can ever match.  The masks below restrict both sides to that
 * set and assert the reconstruction accepts precisely the retail travel types.
 *
 * Inside the mask the two sides are already equal: retail's 102334 reduces to
 * 0xfbe and so does this tree's default.  They differ only above it, where
 * retail carries 0x18000 and this tree carries 0x11c0000 for the jump-pad,
 * air, water, and func_bobbing types Gladiator's reachability writer never
 * emits.  Neither group maps to a Gladiator travel type, so neither can change
 * which retail reachability the battle nodes accept.
 */
#define BOT_GLADIATOR_TRAVELFLAG_MASK                                          \
	(TFL_INVALID | TFL_WALK | TFL_CROUCH | TFL_BARRIERJUMP | TFL_JUMP          \
	 | TFL_LADDER | TFL_WALKOFFLEDGE | TFL_SWIM | TFL_WATERJUMP                \
	 | TFL_TELEPORT | TFL_ELEVATOR | TFL_ROCKETJUMP | TFL_BFGJUMP              \
	 | TFL_GRAPPLEHOOK)
#define BOT_GLADIATOR_BATTLE_TRAVELFLAGS 102334
#define BOT_GLADIATOR_BATTLE_TRAVELFLAGS_HOOK 118718

typedef char bot_assert_battle_travelflags[
	(TFL_DEFAULT & BOT_GLADIATOR_TRAVELFLAG_MASK) ==
	(BOT_GLADIATOR_BATTLE_TRAVELFLAGS & BOT_GLADIATOR_TRAVELFLAG_MASK) ? 1 : -1];
typedef char bot_assert_battle_travelflags_hook[
	((TFL_DEFAULT | TFL_GRAPPLEHOOK) & BOT_GLADIATOR_TRAVELFLAG_MASK) ==
	(BOT_GLADIATOR_BATTLE_TRAVELFLAGS_HOOK & BOT_GLADIATOR_TRAVELFLAG_MASK) ? 1 : -1];
typedef char bot_assert_battle_travelflags_rocketjump[
	((TFL_DEFAULT | TFL_ROCKETJUMP) & BOT_GLADIATOR_TRAVELFLAG_MASK) ==
	((BOT_GLADIATOR_BATTLE_TRAVELFLAGS | 0x1000) &
	 BOT_GLADIATOR_TRAVELFLAG_MASK) ? 1 : -1];

/*
=============
BotAI_BattleChaseTravelFlags

Matches Battle Chase's default travel mask with its independent usehook and
rocketjump gates.
=============
*/
int BotAI_BattleChaseTravelFlags(const bot_client_state_t *state)
{
	int travel_flags = TFL_DEFAULT;
	if (BotAI_LibVarOrderedNonZero("usehook"))
	{
		travel_flags |= TFL_GRAPPLEHOOK;
	}
	if (BotAI_LibVarOrderedNonZero("rocketjump") &&
		BotAI_CanAndWantsToRocketJump(state))
	{
		travel_flags |= TFL_ROCKETJUMP;
	}
	return travel_flags;
}

/*
=============
BotAI_LongTermGoalTravelFlags

Builds Seek LTG/NBG's retail travel mask from its three terms. Retail builds it
from 0x18FBE, the usehook grapple bit and the rocket-jump bit only
(0x1001f813-0x1001f865 and 0x1001f301-0x1001f34b); it never reads the point
contents and never adds the lava or slime bits Q3's later ai_dmnet.c does.
=============
*/
int BotAI_LongTermGoalTravelFlags(const bot_client_state_t *state)
{
	int travel_flags = TFL_DEFAULT;
	if (BotAI_LibVarOrderedNonZero("usehook"))
	{
		travel_flags |= TFL_GRAPPLEHOOK;
	}
	if (BotAI_LibVarOrderedNonZero("rocketjump") &&
		BotAI_CanAndWantsToRocketJump(state))
	{
		travel_flags |= TFL_ROCKETJUMP;
	}
	return travel_flags;
}

/*
=============
BotAI_BattleRetreatTravelFlags

Matches Battle Retreat's shorter travel mask: unlike Chase and Battle NBG it
does not append the rocket-jump gate.
=============
*/
int BotAI_BattleRetreatTravelFlags(void)
{
	int travel_flags = TFL_DEFAULT;
	if (BotAI_LibVarOrderedNonZero("usehook"))
	{
		travel_flags |= TFL_GRAPPLEHOOK;
	}
	return travel_flags;
}

/*
=============
BotAI_ApplyBattleMoveResultView

Preserves BotMoveToGoal's explicit movement view in the input record before a
battle node optionally applies its private accelerated view turn. A mover-set
view is already a direct EA view and must survive the input bridge unchanged.
=============
*/
static void BotAI_ApplyBattleMoveResultView(const bot_client_state_t *state,
	const bot_moveresult_t *result,
	bot_input_t *input)
{
	if (state == NULL || result == NULL || input == NULL)
	{
		return;
	}

	if ((result->flags & MOVERESULT_MOVEMENTVIEWSET) != 0)
	{
		/* Preserve the EA_View captured from the movement helper. */
		return;
	}

	VectorCopy(state->last_client_update.viewangles, input->viewangles);
	if ((result->flags & (MOVERESULT_MOVEMENTVIEW |
		MOVERESULT_SWIMVIEW)) != 0)
	{
		VectorCopy(result->ideal_viewangles, input->viewangles);
	}
}

/*
=============
BotAI_SetBattleResultIdealView

Transfers a movement result's view target to the private Battle AI state.
=============
*/
static void BotAI_SetBattleResultIdealView(bot_client_state_t *state,
	const bot_moveresult_t *result)
{
	if (state == NULL || state->dm_state == NULL || result == NULL)
	{
		return;
	}

	AI_DMState_SetIdealViewAngles(state->dm_state, result->ideal_viewangles);
}

/*
=============
BotAI_SetBattleMovementGoalView

Uses Gladiator's fixed 300-unit movement lookahead when the mover did not
provide its own movement or swim view.

Battle Chase (0x100203f5) and Battle Retreat (0x1002093e) both fall through to
a shared vectoangles tail when BotMovementViewTarget fails, using
moveresult.movedir as the direction, so ideal_viewangles is always written.
Retail's `ideal_viewangles[2] *= 0.5` at 0x1002044f / 0x100209a0 is a dead
multiply - vectoangles (sub_10041790) already stores 0 into ROLL, as
Vector2Angles does - so it is deliberately not reproduced.
=============
*/
static int BotAI_SetBattleMovementGoalView(bot_client_state_t *state,
	const bot_goal_t *goal,
	int travel_flags,
	const bot_moveresult_t *result)
{
	if (state == NULL || state->dm_state == NULL || result == NULL)
	{
		return qfalse;
	}

	vec3_t direction;
	vec3_t target;
	if (goal != NULL &&
		BotMovementViewTargetHandle(state->move_handle,
			goal,
			travel_flags,
			300.0f,
			target))
	{
		VectorSubtract(target, state->last_client_update.origin, direction);
	}
	else
	{
		VectorCopy(result->movedir, direction);
	}

	vec3_t viewangles;
	Vector2Angles(direction, viewangles);
	AI_DMState_SetIdealViewAngles(state->dm_state, viewangles);
	return qtrue;
}

/*
=============
BotAI_ConfigureBattleCombat

Supplies retained enemy ownership to the combat helpers used after a direct
battle-node movement result.
=============
*/
void BotAI_ConfigureBattleCombat(bot_client_state_t *state)
{
	if (state == NULL || state->dm_state == NULL)
	{
		return;
	}

	AI_DMState_SetEnemyContext(state->dm_state,
		state->combat.current_enemy,
		state->combat.enemy_sight_time,
		state->combat.last_enemy_area,
		state->combat.last_enemy_origin);
}

/*
=============
BotAI_SynchroniseBattleWeaponState

Builds retail's per-frame weapon-selection input from the current client state.
=============
*/
static int BotAI_SynchroniseBattleWeaponState(bot_client_state_t *state)
{
	if (state == NULL || state->weapon_state <= 0)
	{
		return qfalse;
	}

	BotWeaponStateSyncFrame(state->weapon_state,
		state->client_number,
		state->last_client_update.inventory,
		BotInterface_ModelNameForIndex(state->last_client_update.gunindex));
	return qtrue;
}

/*
=============
BotAI_ChooseBattleWeapon

Runs retail's weapon score/select operation after its frame input is current.
=============
*/
static void BotAI_ChooseBattleWeapon(bot_client_state_t *state)
{
	const bot_weaponstate_t *weapon_state;
	int weapon_count = 0;
	int weight_count = 0;
	float best_weight = 0.0f;

	if (state == NULL || state->weapon_state <= 0)
	{
		return;
	}

	state->current_weapon = BotSelectBestFightWeapon(state->client_number,
		state->weapon_state,
		state->last_client_update.inventory,
		BotInterface_CurrentFrameTime());
	weapon_state = BotWeaponStatePeek(state->weapon_state);
	if (weapon_state != NULL)
	{
		weapon_count = weapon_state->config != NULL
			? weapon_state->config->num_weapons : 0;
		weight_count = weapon_state->weights != NULL
			? weapon_state->weights->index_count : 0;
		best_weight = weapon_state->last_best_weight;
	}
	BotLib_LogWriteTimeStamped(
		"weapon_select client=%d current=%d blaster=%d health=%d "
		"enemy_height=%d aggression=%.1f retreat=%d best_weight=%.1f "
		"weapon_count=%d weight_count=%d model=\"%s\"",
		state->client_number,
		state->current_weapon,
		state->last_client_update.inventory[BOT_BATTLE_INVENTORY_BLASTER],
		state->last_client_update.inventory[BOT_BATTLE_INVENTORY_HEALTH],
		state->last_client_update.inventory[BOT_BATTLE_ENEMY_HEIGHT],
		BotAI_Aggression(state),
		BotAI_WantsToRetreat(state),
		best_weight,
		weapon_count,
		weight_count,
		BotInterface_ModelNameForIndex(state->last_client_update.gunindex));
}

/*
=============
BotAI_SelectBattleWeapon

Synchronises and selects a weapon only at the retail active-combat call sites.
=============
*/
void BotAI_SelectBattleWeapon(bot_client_state_t *state)
{
	if (!BotAI_SynchroniseBattleWeaponState(state))
	{
		return;
	}

	BotAI_ChooseBattleWeapon(state);
}

/*
=============
BotAI_BattleAttackSkill

Reads Battle Retreat's bounded attack-skill branch with the same no-character
fallback used by the shared combat implementation.
=============
*/
static float BotAI_BattleAttackSkill(const bot_client_state_t *state)
{
	if (state == NULL || state->character == NULL)
	{
		return 1.0f;
	}

	return Characteristic_BFloat(state->character,
		CHARACTERISTIC_ATTACK_SKILL,
		0.0f,
		1.0f);
}

/*
=============
BotAI_RunBattleChaseMovement

Executes Battle Chase's direct BotMoveToGoal path, preserving its move result
for the input bridge instead of using the predictive visible-enemy fallback.
=============
*/
int BotAI_RunBattleChaseMovement(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input)
{
	if (state == NULL || input == NULL)
	{
		return BLERR_INVALIDIMPORT;
	}

	int status = BotInterface_PrepareMoveState(state, thinktime);
	if (status != BLERR_NOERROR)
	{
		return status;
	}

	bot_goal_t chase_goal;
	BotAI_BuildBattleChaseGoal(state, &chase_goal);
	bot_moveresult_t result;
	BotClearMoveResult(&result);
	BotMoveToGoalHandle(&result,
		state->move_handle,
		&chase_goal,
		BotAI_BattleChaseTravelFlags(state));
	if (result.failure)
	{
		BotResetAvoidReachHandle(state->move_handle);
		state->long_term_goal_time = 0.0f;
	}
	vec3_t alternate_direction;
	BotAI_HandleBlockedMovement(state,
		&result,
		false,
		alternate_direction);

	status = EA_GetInput(state->client_number, thinktime, input);
	if (status != BLERR_NOERROR)
	{
		return status;
	}
	BotInterface_ApplyMoveResult(&result, input);
	BotAI_ApplyBattleMoveResultView(state, &result, input);
	state->last_move_result = result;
	state->has_move_result = true;
	status = EA_SubmitInput(state->client_number, input);
	if (status != BLERR_NOERROR)
	{
		return status;
	}

	/*
	 * 0x100200a0 runs the ideal-view selection unconditionally and gates only
	 * BotChangeViewAngles on MOVERESULT_MOVEMENTVIEWSET (0x10020471).  The
	 * chase_time reset sits between the two (0x10020531 between 0x10020529
	 * and 0x10020534).
	 */
	if ((result.flags & (MOVERESULT_MOVEMENTVIEW |
		MOVERESULT_SWIMVIEW)) != 0)
	{
		BotAI_SetBattleResultIdealView(state, &result);
	}
	else
	{
		BotAI_SetBattleMovementGoalView(state,
			&chase_goal,
			BotAI_BattleChaseTravelFlags(state),
			&result);
	}

	bot_movestate_t *move_state = BotMoveStateFromHandle(state->move_handle);
	if (move_state != NULL &&
		move_state->areanum == state->combat.last_enemy_area)
	{
		state->combat.chase_time = 0.0f;
	}

	if ((result.flags & MOVERESULT_MOVEMENTVIEWSET) == 0)
	{
		AI_DMState_ChangeViewAngles(state->dm_state, state, thinktime);
	}
	return BLERR_NOERROR;
}

/*
=============
BotAI_RunBattleNBGMovement

Runs the retained nearby-item goal directly until Battle NBG's five-second
deadline removes it, avoiding the ordinary LTG/NBG stack refresh path.
=============
*/
int BotAI_RunBattleNBGMovement(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input,
	const ai_dm_enemy_info_t *enemy)
{
	if (state == NULL || input == NULL || enemy == NULL || state->goal_handle <= 0)
	{
		return BLERR_INVALIDIMPORT;
	}

	bot_goal_t nearby_goal;
	if (!AI_GoalBotlib_GetTopGoal(state->goal_handle, &nearby_goal))
	{
		return BLERR_INVALIDIMPORT;
	}

	int status = BotInterface_PrepareMoveState(state, thinktime);
	if (status != BLERR_NOERROR)
	{
		return status;
	}

	bot_moveresult_t result;
	BotClearMoveResult(&result);
	BotMoveToGoalHandle(&result,
		state->move_handle,
		&nearby_goal,
		BotAI_BattleChaseTravelFlags(state));
	if (result.failure)
	{
		BotResetAvoidReachHandle(state->move_handle);
		state->nearby_goal_time = 0.0f;
	}
	vec3_t alternate_direction;
	BotAI_HandleBlockedMovement(state,
		&result,
		false,
		alternate_direction);

	status = EA_GetInput(state->client_number, thinktime, input);
	if (status != BLERR_NOERROR)
	{
		return status;
	}
	BotInterface_ApplyMoveResult(&result, input);
	BotAI_ApplyBattleMoveResultView(state, &result, input);
	state->last_move_result = result;
	state->has_move_result = true;

	BotAI_SynchroniseBattleWeaponState(state);
	BotAI_UpdateEnemyBattleInventory(state, state->combat.current_enemy);
	BotAI_ChooseBattleWeapon(state);
	status = EA_SubmitInput(state->client_number, input);
	if (status != BLERR_NOERROR)
	{
		return status;
	}

	BotAI_ConfigureBattleCombat(state);
	if ((result.flags & MOVERESULT_MOVEMENTVIEW) != 0)
	{
		BotAI_SetBattleResultIdealView(state, &result);
	}
	else
	{
		AI_DMState_AimAtEnemy(state->dm_state, state, enemy, thinktime);
	}
	AI_DMState_CheckAttack(state->dm_state,
		state,
		enemy,
		BotInterface_CurrentFrameTime());
	if ((result.flags & MOVERESULT_MOVEMENTVIEWSET) == 0)
	{
		AI_DMState_ChangeViewAngles(state->dm_state, state, thinktime);
	}
	return BLERR_NOERROR;
}

/*
=============
BotAI_RunBattleRetreatMovement

Moves directly to Battle Retreat's retained long-term goal after the node has
applied its enemy-visibility and nearby-goal checks.
=============
*/
int BotAI_RunBattleRetreatMovement(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input,
	const ai_dm_enemy_info_t *enemy,
	const bot_goal_t *retreat_goal)
{
	if (state == NULL || input == NULL || enemy == NULL || retreat_goal == NULL)
	{
		return BLERR_INVALIDIMPORT;
	}

	BotAI_UseItems(state);
	int status = BotInterface_PrepareMoveState(state, thinktime);
	if (status != BLERR_NOERROR)
	{
		return status;
	}

	bot_moveresult_t result;
	BotClearMoveResult(&result);
	BotMoveToGoalHandle(&result,
		state->move_handle,
		retreat_goal,
		BotAI_BattleRetreatTravelFlags());
	if (result.failure)
	{
		BotResetAvoidReachHandle(state->move_handle);
		state->long_term_goal_time = 0.0f;
	}
	vec3_t alternate_direction;
	BotAI_HandleBlockedMovement(state,
		&result,
		false,
		alternate_direction);

	status = EA_GetInput(state->client_number, thinktime, input);
	if (status != BLERR_NOERROR)
	{
		return status;
	}
	BotInterface_ApplyMoveResult(&result, input);
	BotAI_ApplyBattleMoveResultView(state, &result, input);
	state->last_move_result = result;
	state->has_move_result = true;
	state->active_goal_number = retreat_goal->number;

	BotAI_SelectBattleWeapon(state);
	status = EA_SubmitInput(state->client_number, input);
	if (status != BLERR_NOERROR)
	{
		return status;
	}

	BotAI_ConfigureBattleCombat(state);
	if ((result.flags & MOVERESULT_MOVEMENTVIEW) != 0)
	{
		BotAI_SetBattleResultIdealView(state, &result);
	}
	else if ((result.flags & MOVERESULT_MOVEMENTVIEWSET) == 0)
	{
		if (BotAI_BattleAttackSkill(state) < 0.3f)
		{
			/* 0x100209a6 turns on both arms of the view-target if/else. */
			(void)BotAI_SetBattleMovementGoalView(state,
				retreat_goal,
				BotAI_BattleRetreatTravelFlags(),
				&result);
			AI_DMState_ChangeViewAngles(state->dm_state, state, thinktime);
		}
		else
		{
			AI_DMState_AimAtEnemy(state->dm_state, state, enemy, thinktime);
		}
	}
	AI_DMState_CheckAttack(state->dm_state,
		state,
		enemy,
		BotInterface_CurrentFrameTime());
	return BLERR_NOERROR;
}

/*
=============
BotAI_RunBattleRetreatIdle

Handles Battle Retreat's no-long-term-goal return: retail leaves the node
active, submits no movement, and advances the retained private view turn.
=============
*/
int BotAI_RunBattleRetreatIdle(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input)
{
	if (state == NULL || input == NULL)
	{
		return BLERR_INVALIDIMPORT;
	}

	memset(input, 0, sizeof(*input));
	input->thinktime = thinktime;
	VectorCopy(state->last_client_update.viewangles, input->viewangles);
	int status = EA_SubmitInput(state->client_number, input);
	if (status != BLERR_NOERROR)
	{
		return status;
	}

	BotAI_ConfigureBattleCombat(state);
	AI_DMState_ChangeViewAngles(state->dm_state, state, thinktime);
	return BLERR_NOERROR;
}
