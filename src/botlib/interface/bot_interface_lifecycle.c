#include <float.h>
#include <string.h>

#include "botlib/aas/aas_local.h"
#include "botlib/ai/ai_dm.h"
#include "botlib/ai_goal/ai_goal.h"
#include "botlib/ai_chat/ai_chat.h"
#include "botlib/common/l_libvar.h"
#include "botlib/ea/ea_local.h"
#include "bot_interface_battle.h"
#include "bot_interface_console.h"
#include "bot_interface_lifecycle.h"
#include "bot_state.h"

/*
=============
BotAI_ReplyStandActive

Dispatches a completed pending reply strictly after the retail stand deadline
after first advancing Stand's private view turn. The private retail guard
keeps the bot standing while it emits its removebot command.
=============
*/
bool BotAI_ReplyStandActive(bot_client_state_t *state, float thinktime)
{
	if (state == NULL || !state->chat_standing)
	{
		return false;
	}

	if (AAS_Time() <= state->stand_time)
	{
		return true;
	}

	BotAI_ConfigureBattleCombat(state);
	AI_DMState_ChangeViewAngles(state->dm_state, state, thinktime);
	if (LibVarGetValue("__squatt") != 0.0f)
	{
		EA_Say(state->client_number, "I never hacked your brain...\n");
		/* Retail 0x1001ed0a pushes ClientName(bs->client) as EA_Command's
		   single argument before the NULL sentinel, so the host removes the
		   bot that spoke rather than the first in-use bot edict. */
		EA_Command(state->client_number,
			"removebot",
			(char *)BotState_ClientName(state->client_number),
			(char *)NULL);
		return true;
	}
	if (state->chat_state != NULL)
	{
		BotEnterChat(state->chat_state, state->client_number, 0);
	}
	state->chat_standing = false;
	return false;
}

/*
=============
BotAI_RunStand

Submits Stand's stationary input and advances the retained private view turn
before its pending-chat deadline is handled by the node scheduler.
=============
*/
int BotAI_RunStand(bot_client_state_t *state, float thinktime)
{
	bot_input_t input;
	memset(&input, 0, sizeof(input));
	input.thinktime = thinktime;
	VectorCopy(state->last_client_update.viewangles, input.viewangles);

	int status = EA_SubmitInput(state->client_number, &input);
	if (status != BLERR_NOERROR)
	{
		return status;
	}

	BotAI_ConfigureBattleCombat(state);
	AI_DMState_ChangeViewAngles(state->dm_state, state, thinktime);
	BotState_EmitPendingClientCommands(state);
	status = EA_EndRegular(state->client_number, thinktime);
	if (status == BLERR_NOERROR)
	{
		state->client_update_valid = false;
	}
	return status;
}

/*
=============
BotAI_ResetRespawnState

Mirrors the reset sequence at the retail respawn-node entry before the engine
accepts the respawn action.
=============
*/
void BotAI_ResetRespawnState(bot_client_state_t *state)
{
	if (state == NULL)
	{
		return;
	}

	if (state->goal_handle > 0)
	{
		BotResetGoalState(state->goal_handle);
	}
	if (state->goal_state != NULL)
	{
		AI_GoalState_Reset(state->goal_state);
	}
	if (state->move_handle > 0)
	{
		BotResetMoveStateHandle(state->move_handle);
	}
	if (state->weapon_state > 0)
	{
		BotResetWeaponState(state->weapon_state);
	}
	if (state->dm_state != NULL)
	{
		AI_DMState_Reset(state->dm_state);
	}

	state->goal_snapshot_count = 0;
	memset(state->goal_snapshot, 0, sizeof(state->goal_snapshot));
	memset(&state->last_move_result, 0, sizeof(state->last_move_result));
	state->has_move_result = false;
	state->active_goal_number = 0;
	state->current_weapon = 0;
	state->combat.last_enemy_area = 0;
	state->combat.enemy_visible = false;
	state->combat.enemy_visible_time = -FLT_MAX;
	state->combat.enemy_sight_time = -FLT_MAX;
	state->combat.enemy_death_time = -FLT_MAX;
	state->combat.enemy_last_seen_time = -FLT_MAX;
	state->combat.chase_time = -FLT_MAX;
	state->combat.revenge_enemy = -1;
	state->combat.revenge_kills = 0;
	state->combat.last_known_health = 0;
	state->combat.last_damage_amount = 0;
	state->combat.last_damage_time = -FLT_MAX;
	state->combat.last_health_valid = false;
	state->combat.took_damage = false;
	VectorClear(state->combat.last_enemy_origin);
	VectorClear(state->combat.last_enemy_velocity);
}

/*
=============
BotAI_CompleteRespawnAction

Applies the side effects immediately after retail's EA Respawn call: mark the
one-shot action, enter any pending death chat for the retained killer context,
then release that retained enemy.
=============
*/
void BotAI_CompleteRespawnAction(bot_client_state_t *state)
{
	if (state == NULL)
	{
		return;
	}

	state->respawn_action_sent = true;
	if (state->combat.current_enemy != 0)
	{
		if (state->chat_state != NULL)
		{
			BotEnterChat(state->chat_state, state->client_number, 0);
		}
		state->combat.current_enemy = 0;
	}
}

/*
=============
BotAI_SetLifecycleStand

Adapts Gladiator's generic stand node to the reconstructed reply-stand state.
Stand always enters the pending chat at expiry; an empty buffer is a no-op.
=============
*/
void BotAI_SetLifecycleStand(bot_client_state_t *state, float duration)
{
	if (state == NULL)
	{
		return;
	}

	state->stand_time = AAS_Time() + duration;
	state->chat_standing = true;
	BotAI_EnterNode(state, BOT_AI_NODE_STAND);
}

/*
=============
BotAI_EnterObserver

Mirrors AIEnter_Observer: discard the level-local bot state once, then leave
the dedicated observer node passive until the pmove state changes.
=============
*/
void BotAI_EnterObserver(bot_client_state_t *state)
{
	if (state == NULL)
	{
		return;
	}

	BotState_ResetForNewMap(state);
	BotAI_EnterNode(state, BOT_AI_NODE_OBSERVER);
}

/*
=============
BotAI_EnterIntermission

Mirrors AIEnter_Intermission's reset and immediate end-level initial chat.
=============
*/
void BotAI_EnterIntermission(bot_client_state_t *state)
{
	if (state == NULL)
	{
		return;
	}

	BotState_ResetForNewMap(state);
	if (BotAI_ConstructLifecycleChat(state,
		"end_level",
		CHARACTERISTIC_CHAT_STARTENDLEVEL,
		false))
	{
		BotEnterChat(state->chat_state, state->client_number, 0);
	}
	BotAI_EnterNode(state, BOT_AI_NODE_INTERMISSION);
}

/*
=============
BotAI_RunLifecycleFrame

Submits the retail passive/respawn action frame while excluding ordinary
combat, item, and movement work for spectator, intermission, dead, and gib
pmove states.
=============
*/
int BotAI_RunLifecycleFrame(bot_client_state_t *state,
	float thinktime,
	bool request_respawn)
{
	if (state == NULL)
	{
		return BLERR_AIUPDATEINACTIVECLIENT;
	}

	EA_ResetInput(state->client_number);
	if (request_respawn)
	{
		EA_Respawn(state->client_number);
		BotAI_CompleteRespawnAction(state);
	}

	int status = EA_EndRegular(state->client_number, thinktime);
	if (status == BLERR_NOERROR)
	{
		state->client_update_valid = false;
	}
	return status;
}
