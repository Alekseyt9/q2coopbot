#include <math.h>
#include <string.h>

#include "botlib/aas/aas_local.h"
#include "botlib/common/l_libvar.h"
#include "botlib/common/l_log.h"
#include "bot_interface_combat.h"
#include "bot_interface_coop_role.h"
#include "bot_interface_coop_state.h"
#include "bot_state.h"

const char *BotAI_CoopRoleName(bot_coop_role_t role)
{
	switch (role)
	{
		case BOT_COOP_ROLE_SUPPORT:
			return "SUPPORT";
		case BOT_COOP_ROLE_ANCHOR:
			return "ANCHOR";
		case BOT_COOP_ROLE_COVER:
			return "COVER";
		case BOT_COOP_ROLE_RESCUER:
			return "RESCUER";
		case BOT_COOP_ROLE_FINISHER:
			return "FINISHER";
		case BOT_COOP_ROLE_VANGUARD:
			return "VANGUARD";
		case BOT_COOP_ROLE_REGROUP:
			return "REGROUP";
		case BOT_COOP_ROLE_FOLLOWER:
		default:
			return "FOLLOWER";
	}
}

/*
===============
BotAI_UpdateCoopJointRetreat

Keep a short cooperative retreat commitment after the human has clearly
fallen back. Intent is sampled every AI frame, so without this lease a
stationary frame immediately after the retreat would let Battle Retreat hand
control back to Chase. The lease is deliberately short and configurable;
hard leash/elevator and danger overlays still have priority over it.
===============
*/
void BotAI_UpdateCoopJointRetreat(bot_client_state_t *state)
{
	float now;
	float duration;
	bool enabled;

	if (state == NULL)
	{
		return;
	}

	now = AAS_Time();
	enabled = BotAI_CoopMode() != 0 &&
		LibVarGetValue("coopbot_player_intent") != 0.0f &&
		LibVarGetValue("coopbot_joint_retreat") != 0.0f;
	if (!enabled)
	{
		state->coop_joint_retreat_until = 0.0f;
		state->coop_joint_retreat_active = false;
		return;
	}

	duration = LibVarGetValue("coopbot_joint_retreat_duration");
	if (duration <= 0.0f)
	{
		duration = 1.5f;
	}

	if (BotAI_CoopIntentIsConfident(state, BOT_COOP_INTENT_RETREAT))
	{
		if (!state->coop_joint_retreat_active &&
			LibVarGetValue("coopbot_log") >= 1.0f)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_joint_retreat client=%d phase=start duration=%.2f",
				state->client_number, duration);
		}
		state->coop_joint_retreat_active = true;
		if (state->coop_joint_retreat_until < now + duration)
		{
			state->coop_joint_retreat_until = now + duration;
		}
		return;
	}

	if (state->coop_joint_retreat_active &&
		now >= state->coop_joint_retreat_until)
	{
		if (LibVarGetValue("coopbot_log") >= 1.0f)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_joint_retreat client=%d phase=end",
				state->client_number);
		}
		state->coop_joint_retreat_active = false;
		state->coop_joint_retreat_until = 0.0f;
	}
}

static const char *BotAI_CoopActionName(bot_coop_action_t action)
{
	switch (action)
	{
		case BOT_COOP_ACTION_POSITION:
			return "POSITION";
		case BOT_COOP_ACTION_RESCUE:
			return "RESCUE";
		case BOT_COOP_ACTION_NONE:
		default:
			return "NONE";
	}
}

void BotAI_CoopClearActionCommitment(bot_client_state_t *state)
{
	if (state == NULL)
	{
		return;
	}
	if (state->coop_action_valid && LibVarGetValue("coopbot_log") >= 1.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_action client=%d action=%s phase=end started=%.2f",
			state->client_number,
			BotAI_CoopActionName(state->coop_action),
			state->coop_action_started);
	}
	state->coop_action = BOT_COOP_ACTION_NONE;
	state->coop_action_started = 0.0f;
	state->coop_action_until = 0.0f;
	VectorClear(state->coop_action_direction);
	state->coop_action_valid = false;
}

void BotAI_CoopStartAction(bot_client_state_t *state,
	bot_coop_action_t action,
	const vec3_t direction)
{
	float now;
	float commitment;

	if (state == NULL)
	{
		return;
	}
	now = AAS_Time();
	commitment = LibVarGetValue("coopbot_action_commitment");
	if (commitment <= 0.0f)
	{
		commitment = 0.75f;
	}
	if (state->coop_action_valid && state->coop_action == action &&
		state->coop_action_until > now)
	{
		/* Rescue remains active while the critical-health signal persists. */
		if (action == BOT_COOP_ACTION_RESCUE &&
			state->coop_action_until < now + commitment)
		{
			state->coop_action_until = now + commitment;
		}
		return;
	}
	BotAI_CoopClearActionCommitment(state);
	state->coop_action = action;
	state->coop_action_started = now;
	state->coop_action_until = now + commitment;
	if (direction != NULL)
	{
		VectorCopy(direction, state->coop_action_direction);
	}
	else
	{
		VectorClear(state->coop_action_direction);
	}
	state->coop_action_valid = true;
	if (LibVarGetValue("coopbot_log") >= 1.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_action client=%d action=%s phase=start until=%.2f",
			state->client_number,
			BotAI_CoopActionName(action),
			state->coop_action_until);
	}
}

bool BotAI_CoopRoleBlocksNewGroup(const bot_client_state_t *state)
{
	if (state == NULL || LibVarGetValue("coopbot_roles") == 0.0f)
	{
		return false;
	}
	return state->coop_role == BOT_COOP_ROLE_REGROUP ||
		state->coop_role == BOT_COOP_ROLE_COVER ||
		state->coop_role == BOT_COOP_ROLE_ANCHOR ||
		state->coop_role == BOT_COOP_ROLE_RESCUER;
}

bool BotAI_CoopRoleBreaksChase(const bot_client_state_t *state)
{
	if (state == NULL || LibVarGetValue("coopbot_roles") == 0.0f)
	{
		return false;
	}
	return state->coop_role == BOT_COOP_ROLE_REGROUP ||
		state->coop_role == BOT_COOP_ROLE_COVER ||
		state->coop_role == BOT_COOP_ROLE_ANCHOR ||
		state->coop_role == BOT_COOP_ROLE_RESCUER;
}

/*
=============
BotAI_UpdateCoopRole

Select a short-lived role modifier from the intent and hard safety gates.  The
role does not replace the retail AI node; it only changes how much initiative
the coop overlay may take and whether a distant/new fight is admissible.
=============
*/
void BotAI_UpdateCoopRole(bot_client_state_t *state)
{
	aas_entityinfo_t player_info;
	bot_coop_role_t previous_role;
	bot_coop_role_t role = BOT_COOP_ROLE_FOLLOWER;
	float budget = 0.0f;
	float confidence = 0.0f;
	float distance;
	float vertical;
	float hard_leash;
	vec3_t direction;
	int player_entity;
	int player_area;
	int bot_area;

	if (state == NULL)
	{
		return;
	}
	if (LibVarGetValue("coopbot_roles") == 0.0f || !BotAI_CoopMode())
	{
		BotAI_CoopClearActionCommitment(state);
		state->coop_role = BOT_COOP_ROLE_FOLLOWER;
		state->coop_initiative_budget = 0.0f;
		state->coop_role_confidence = 0.0f;
		return;
	}

	memset(&player_info, 0, sizeof(player_info));
	player_entity = BotAI_CoopPlayerEntityInfo(state, &player_info);
	if (player_entity < 0)
	{
		BotAI_CoopClearActionCommitment(state);
		state->coop_role = BOT_COOP_ROLE_FOLLOWER;
		state->coop_initiative_budget = 0.0f;
		state->coop_role_confidence = 0.0f;
		return;
	}

	VectorSubtract(player_info.origin,
		state->last_client_update.origin,
		direction);
	distance = sqrtf(DotProduct(direction, direction));
	vertical = fabsf(direction[2]);
	hard_leash = LibVarGetValue("coopbot_hard_leash");
	player_area = AAS_PointAreaNum(player_info.origin);
	bot_area = AAS_PointAreaNum(state->last_client_update.origin);
	previous_role = state->coop_role;

	if (hard_leash > 0.0f &&
		(distance > hard_leash ||
			(bot_area > 0 && player_area > 0 && bot_area != player_area &&
				vertical >= 64.0f)))
	{
		role = BOT_COOP_ROLE_REGROUP;
		budget = 0.0f;
		confidence = 1.0f;
	}
	else if (BotAI_CoopPlayerNeedsRescue(state))
	{
		role = BOT_COOP_ROLE_RESCUER;
		budget = 0.85f;
		confidence = 0.90f;
	}
	else if (state->coop_joint_retreat_active)
	{
		role = BOT_COOP_ROLE_COVER;
		budget = 0.75f;
		confidence = 0.90f;
	}
	else if (BotAI_CoopIntentIsConfident(state,
		BOT_COOP_INTENT_RETREAT))
	{
		role = BOT_COOP_ROLE_COVER;
		budget = 0.75f;
		confidence = state->coop_player_intent_confidence;
	}
	else if (BotAI_CoopIntentIsConfident(state,
		BOT_COOP_INTENT_ENGAGE_TARGET))
	{
		role = BOT_COOP_ROLE_SUPPORT;
		budget = 0.35f;
		confidence = state->coop_player_intent_confidence;
	}
	else if (BotAI_CoopIntentIsConfident(state,
		BOT_COOP_INTENT_ADVANCE) ||
		BotAI_CoopIntentIsConfident(state, BOT_COOP_INTENT_EXPLORE))
	{
		/* VANGUARD is intentionally low initiative: advance only briefly. */
		role = BOT_COOP_ROLE_VANGUARD;
		budget = state->coop_player_intent == BOT_COOP_INTENT_ADVANCE
			? 0.20f
			: 0.15f;
		confidence = state->coop_player_intent_confidence;
	}
	else if (BotAI_CoopIntentIsConfident(state,
		BOT_COOP_INTENT_HOLD))
	{
		role = BOT_COOP_ROLE_ANCHOR;
		budget = 0.45f;
		confidence = state->coop_player_intent_confidence;
	}
	else if (state->coop_player_intent == BOT_COOP_INTENT_SEARCH)
	{
		role = BOT_COOP_ROLE_SUPPORT;
		budget = 0.25f;
		confidence = state->coop_player_intent_confidence;
	}
	else
	{
		role = BOT_COOP_ROLE_FOLLOWER;
		budget = 0.10f;
		confidence = state->coop_player_intent_confidence;
	}

	/* Adapt only the soft role choice; regroup, rescue and other safety gates
	 * above remain authoritative regardless of the learned style. */
	if (state->coop_player_style_valid &&
		state->coop_player_style_confidence >= 0.25f)
	{
		if (state->coop_player_style_aggression >= 0.68f &&
			role == BOT_COOP_ROLE_VANGUARD)
		{
			/* An aggressive player already creates forward pressure; support
			 * them instead of trying to lead the encounter. */
			role = BOT_COOP_ROLE_SUPPORT;
			budget = 0.35f;
		}
		else if (role == BOT_COOP_ROLE_VANGUARD &&
			state->coop_player_style_risk_tolerance >= 0.55f)
		{
			/* A confident, fast player gives the companion a little more room
			 * to take the next useful position. */
			budget = fminf(budget + 0.10f, 0.35f);
		}
		if (state->coop_player_style_risk_tolerance < 0.30f)
		{
			budget = fminf(budget, 0.20f);
		}
	}

	if (role == BOT_COOP_ROLE_REGROUP)
	{
		if (state->coop_action == BOT_COOP_ACTION_RESCUE)
		{
			BotAI_CoopClearActionCommitment(state);
		}
	}
	else if (role == BOT_COOP_ROLE_RESCUER)
	{
		BotAI_CoopStartAction(state, BOT_COOP_ACTION_RESCUE, NULL);
	}
	else if (state->coop_action == BOT_COOP_ACTION_RESCUE)
	{
		if (state->coop_action_valid &&
			AAS_Time() < state->coop_action_until)
		{
			/* Keep the short rescue decision stable after a transient sample. */
			role = BOT_COOP_ROLE_RESCUER;
			budget = 0.85f;
			confidence = 0.90f;
		}
		else
		{
			BotAI_CoopClearActionCommitment(state);
		}
	}
	if (role != BOT_COOP_ROLE_COVER && role != BOT_COOP_ROLE_SUPPORT &&
		state->coop_action == BOT_COOP_ACTION_POSITION)
	{
		BotAI_CoopClearActionCommitment(state);
	}
	state->coop_role = role;
	state->coop_initiative_budget = budget;
	state->coop_role_confidence = confidence;
	state->coop_role_time = AAS_Time();
	if (previous_role != role && LibVarGetValue("coopbot_log") >= 1.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_role client=%d player=%d role=%s confidence=%.2f "
			"initiative=%.2f distance=%.1f intent=%s",
			state->client_number,
			player_entity,
			BotAI_CoopRoleName(role),
			confidence,
			budget,
			distance,
			BotAI_CoopPlayerIntentName(state->coop_player_intent));
		if (role == BOT_COOP_ROLE_REGROUP)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_action client=%d action=REGROUP phase=start",
				state->client_number);
		}
		if (previous_role == BOT_COOP_ROLE_REGROUP)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_action client=%d action=REGROUP phase=end",
				state->client_number);
		}
	}
}

