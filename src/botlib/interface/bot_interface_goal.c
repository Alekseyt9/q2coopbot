#include <string.h>

#include "botlib/aas/aas_local.h"
#include "botlib/ai_goal/ai_goal.h"
#include "botlib/common/l_libvar.h"
#include "botlib/common/l_log.h"
#include "botlib/ea/ea_local.h"
#include "bot_interface_assets.h"
#include "bot_interface_battle.h"
#include "bot_interface_battle_state.h"
#include "bot_interface_combat.h"
#include "bot_interface_console.h"
#include "bot_interface_entity.h"
#include "bot_interface_goal.h"
#include "bot_state.h"

void BotAI_EnterNode(bot_client_state_t *state, int node);

/*
=============
BotAI_ActivateEntityTravelFlags

Matches the activate-entity node's two-term travel mask. Retail builds it from
0x18FBE plus the usehook grapple bit only (0x1001efad-0x1001efd7); unlike Seek
NBG and Seek LTG it never reads the rocketjump libvar or sub_10022990.
=============
*/
int BotAI_ActivateEntityTravelFlags(void)
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
BotAI_DropUnwantedCTFTech

Replays sub_100262c0 after a runes-enabled goal contact. When a touched CTF
tech differs from the held tech, retail drops the held one. Its tech-four
comparison deliberately keeps the raw Haste-model exception rather than the
otherwise expected Regeneration-model exception.
=============
*/
static void BotAI_DropUnwantedCTFTech(const bot_client_state_t *state,
	const bot_goal_t *goal)
{
	bot_updateentity_t cached_entity;
	if (state == NULL || goal == NULL ||
		!BotAI_LibVarOrderedNonZero("ctf") ||
		goal->entitynum <= 0 ||
		!BotInterface_ReadEntityCache(goal->entitynum, &cached_entity))
	{
		return;
	}

	const char *model_name = BotInterface_ModelNameForIndex(
		cached_entity.modelindex);
	if (model_name == NULL)
	{
		return;
	}

	bool resistance = strcmp(model_name,
		"models/ctf/resistance/tris.md2") == 0;
	bool strength = strcmp(model_name,
		"models/ctf/strength/tris.md2") == 0;
	bool haste = strcmp(model_name, "models/ctf/haste/tris.md2") == 0;
	bool regeneration = strcmp(model_name,
		"models/ctf/regeneration/tris.md2") == 0;
	if (!resistance && !strength && !haste && !regeneration)
	{
		return;
	}

	const int *inventory = state->last_client_update.inventory;
	if ((inventory[BOT_BATTLE_INVENTORY_TECH1] > 0 && !resistance) ||
		(inventory[BOT_BATTLE_INVENTORY_TECH2] > 0 && !strength) ||
		(inventory[BOT_BATTLE_INVENTORY_TECH3] > 0 && !haste) ||
		(inventory[BOT_BATTLE_INVENTORY_TECH4] > 0 && !haste))
	{
		EA_DropItem(state->client_number, "tech");
	}
}

/*
=============
BotAI_TouchingNearbyGoal

Checks the retail contact branch shared by item LTG, Seek NBG, and Battle NBG.
=============
*/
bool BotAI_TouchingNearbyGoal(bot_client_state_t *state,
	bot_goal_t *goal)
{
	if (state == NULL || goal == NULL ||
		!BotTouchingGoal(state->last_client_update.origin, goal))
	{
		return false;
	}

	if ((goal->flags & GFL_ITEM) != 0 && BotAI_LibVarOrderedNonZero("runes"))
	{
		BotAI_DropUnwantedCTFTech(state, goal);
	}
	return true;
}

/*
=============
BotAI_NearbyGoalReached

Applies BotReachedGoal's retail item-specific contact and visibility tests
before an LTG or NBG goal is replaced. The selectors exclusively own retail
avoid-slot insertion.
=============
*/
bool BotAI_NearbyGoalReached(bot_client_state_t *state,
	bot_goal_t *goal)
{
	if (state == NULL || goal == NULL)
	{
		return false;
	}

	if (BotAI_TouchingNearbyGoal(state, goal))
	{
		return true;
	}
	if ((goal->flags & GFL_ITEM) == 0)
	{
		return false;
	}

	vec3_t eye;
	BotInterface_ClientEyePosition(state, eye);
	if (BotItemGoalInVisButNotVisible(state->entity_number,
		eye,
		state->last_client_update.viewangles,
		goal) != 0)
	{
		return true;
	}

	return false;
}

/*
=============
BotAI_GetItemLongTermGoal

Reconstructs the ordinary-item tail of BotLongTermGoal: retain a selected
item for twenty seconds, replace it on contact/expiry, and recover from a
fully avoided item set by clearing both avoid layers.
=============
*/
bool BotAI_GetItemLongTermGoal(bot_client_state_t *state,
	bot_goal_t *goal,
	int travel_flags)
{
	if (state == NULL || goal == NULL || state->goal_handle <= 0)
	{
		return false;
	}

	if (AI_GoalBotlib_GetTopGoal(state->goal_handle, goal) == 0)
	{
		state->long_term_goal_time = 0.0f;
	}
	else if (BotAI_NearbyGoalReached(state, goal))
	{
		state->long_term_goal_time = 0.0f;
	}

	if (state->long_term_goal_time < AAS_Time())
	{
		AI_GoalBotlib_PopGoal(state->goal_handle);
		/* 0x1001e6b1 passes the tfl BotLongTermGoal was called with, which
		   for Battle Retreat is its own mask, not Seek LTG's. */
		if (AI_GoalBotlib_ChooseLTG(state->goal_handle,
			state->last_client_update.origin,
			state->last_client_update.inventory,
			travel_flags) != 0)
		{
			state->long_term_goal_time = AAS_Time() + 20.0f;
		}
		else
		{
			AI_GoalBotlib_ResetAvoidGoals(state->goal_handle);
			BotResetAvoidReachHandle(state->move_handle);
		}
	}

	return AI_GoalBotlib_GetTopGoal(state->goal_handle, goal) != 0;
}

/*
=============
BotAI_TryLongTermNearbyGoal

Reconstructs Seek LTG's half-second nearby-item trial. Gladiator uses the
larger 1500 travel-time budget only while directly defending a key area and
gives a selected nearby goal exactly five seconds before returning to LTG.
=============
*/
bool BotAI_TryLongTermNearbyGoal(bot_client_state_t *state,
	const bot_goal_t *long_term_goal,
	int travel_flags)
{
	if (state == NULL || long_term_goal == NULL || state->goal_handle <= 0 ||
		AAS_Time() <= state->nearby_goal_check_time)
	{
		return false;
	}

	state->nearby_goal_check_time = AAS_Time() + 0.5f;
	float max_travel_time = state->ltg_type == 3 ? 1500.0f : 700.0f;
	bot_goal_t goal = *long_term_goal;
	if (!AI_GoalBotlib_ChooseNBG(state->goal_handle,
		state->last_client_update.origin,
		state->last_client_update.inventory,
		travel_flags,
		&goal,
		max_travel_time))
	{
		return false;
	}

	state->nearby_goal_time = AAS_Time() + 5.0f;
	BotAI_EnterNode(state, BOT_AI_NODE_SEEK_NBG);
	BotResetLastAvoidReachHandle(state->move_handle);
	return true;
}

/*
=============
BotAI_TryBattleChaseNearbyGoal

Performs the retail once-per-second nearby-item search against the retained
last-enemy goal and transfers a selected item to Battle NBG for five seconds.
The 0xaec/0xaf8 lease and probe clocks are shared with Seek LTG/NBG rather
than belonging to combat state.
=============
*/
int BotAI_TryBattleChaseNearbyGoal(bot_client_state_t *state,
	const bot_goal_t *chase_goal,
	int travel_flags)
{
	if (state == NULL || chase_goal == NULL ||
		AAS_Time() <= state->nearby_goal_check_time)
	{
		return qfalse;
	}

	state->nearby_goal_check_time = AAS_Time() + 1.0f;
	bot_goal_t goal = *chase_goal;
	if (state->goal_handle <= 0 || !AI_GoalBotlib_ChooseNBG(state->goal_handle,
		state->last_client_update.origin,
		state->last_client_update.inventory,
		travel_flags,
		&goal,
		500.0f))
	{
		return qfalse;
	}

	state->nearby_goal_time = AAS_Time() + 5.0f;
	BotAI_ResetFightNavigation(state, false);
	BotAI_EnterNode(state, BOT_AI_NODE_BATTLE_NBG);
	return qtrue;
}

/*
=============
BotAI_BuildCoopSafeAreaGoal

Turn the last observed safe position into a retreat goal only for a genuine
danger retreat.  It is a fallback below team/item retreat goals, and the AAS
travel-time check prevents a stale cross-area memory from trapping the bot.
=============
*/
bool BotAI_BuildCoopSafeAreaGoal(const bot_client_state_t *state,
	bot_goal_t *goal)
{
	vec3_t start_origin;
	int start_area;
	int travel_time;

	if (state == NULL || goal == NULL ||
		LibVarGetValue("coopbot_safe_area_retreat") == 0.0f ||
		!state->coop_last_safe_valid ||
		!BotAI_CoopMode() ||
		(state->coop_area_state != BOT_COOP_AREA_DANGEROUS &&
		 !BotAI_CoopDangerRequiresRetreat(state)) ||
		!AAS_Initialized())
	{
		return false;
	}

	start_area = AAS_PointAreaNum(state->last_client_update.origin);
	if (start_area <= 0 || state->coop_last_safe_area <= 0)
	{
		return false;
	}

	VectorCopy(state->last_client_update.origin, start_origin);
	travel_time = AAS_AreaTravelTimeToGoalArea(start_area,
		start_origin,
		state->coop_last_safe_area,
		BotAI_BattleRetreatTravelFlags());
	if (travel_time <= 0 && start_area != state->coop_last_safe_area)
	{
		return false;
	}

	memset(goal, 0, sizeof(*goal));
	VectorCopy(state->coop_last_safe_origin, goal->origin);
	VectorSet(goal->mins, -16.0f, -16.0f, -24.0f);
	VectorSet(goal->maxs, 16.0f, 16.0f, 32.0f);
	goal->areanum = state->coop_last_safe_area;
	goal->entitynum = -1;
	if (LibVarGetValue("coopbot_log") >= 1.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_safe_area_retreat client=%d from_area=%d to_area=%d "
			"safe_time=%.2f",
			state->client_number,
			start_area,
			state->coop_last_safe_area,
			state->coop_last_safe_time);
	}
	return true;
}

/*
=============
BotAI_CTFRetreatGoals

Retail BotCTFRetreatGoals (sub_100263d0, ref be_ai2_dmq2.c:2124-2139).  It only
promotes a flag carrier to the rush-base LTG and resets that branch's timers -
it resolves no goal.  Battle Retreat takes the goal itself from BotLongTermGoal
on the very next line (0x10020795 then 0x100207a6), whose ltgtype-5 arm does
the flag lookup.
=============
*/
void BotAI_CTFRetreatGoals(bot_client_state_t *state)
{
	if (state == NULL || BotAI_CarryingFlag(state) == 0)
	{
		return;
	}

	if (state->ltg_type != BOT_LTG_RUSH_BASE)
	{
		state->ltg_type = BOT_LTG_RUSH_BASE;
		state->team_goal_time = AAS_Time() + 120.0f;
		state->rush_base_away_time = 0.0f;
	}
}
