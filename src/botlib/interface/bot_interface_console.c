#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "shared/q_platform.h"
#include "q2bridge/botlib.h"
#include "botlib/common/l_libvar.h"
#include "botlib/aas/aas_local.h"
#include "botlib/ai/ai_dm.h"
#include "botlib/ai_chat/ai_chat.h"
#include "botlib/ai_character/bot_character.h"
#include "botlib/ai_goal/ai_goal.h"
#include "botlib/ai_goal/bot_goal.h"
#include "botlib/ea/ea_local.h"
#include "bot_interface.h"
#include "bot_interface_behavior.h"
#include "bot_interface_combat.h"
#include "bot_interface_console.h"
#include "bot_interface_entity.h"
#include "bot_state.h"

/*
=============
BotAI_IsOwnConsoleChat

Applies Gladiator's two bounded netname comparisons for plain and parenthesised
chat prefixes.
=============
*/
static bool BotAI_IsOwnConsoleChat(const bot_client_state_t *state,
	const char *message,
	const char *colon)
{
	if (state == NULL || message == NULL || colon == NULL || colon < message)
	{
		return false;
	}

	const char *bot_name = BotState_ClientName(state->client_number);
	if (bot_name == NULL)
	{
		bot_name = "";
	}

	size_t prefix_length = (size_t)(colon - message);
	if (strncmp(message, bot_name, prefix_length) == 0)
	{
		return true;
	}

	if (prefix_length >= 2U &&
		strncmp(message + 1, bot_name, prefix_length - 2U) == 0)
	{
		return true;
	}

	return false;
}

/*
=============
BotAI_ValidChatPosition

Reconstructs Gladiator's dead/ground, hazardous-contents, and world-floor
checks before a bot may stop to answer chat.
=============
*/
static bool BotAI_ValidChatPosition(const bot_client_state_t *state)
{
	if (state == NULL)
	{
		return false;
	}

	if (state->last_client_update.pm_type == PM_DEAD ||
		state->last_client_update.pm_type == PM_GIB)
	{
		return true;
	}

	if ((state->last_client_update.pm_flags & PMF_ON_GROUND) == 0)
	{
		return false;
	}

	vec3_t point;
	VectorCopy(state->last_client_update.origin, point);
	point[2] -= 24.0f;
	if ((AAS_PointContents(point) & (CONTENTS_LAVA | CONTENTS_SLIME)) != 0)
	{
		return false;
	}

	VectorCopy(state->last_client_update.origin, point);
	point[2] += 32.0f;
	if ((AAS_PointContents(point) & MASK_WATER) != 0)
	{
		return false;
	}

	vec3_t start;
	vec3_t end;
	vec3_t mins;
	vec3_t maxs;
	VectorCopy(state->last_client_update.origin, start);
	VectorCopy(state->last_client_update.origin, end);
	start[2] += 1.0f;
	end[2] -= 100.0f;
	AAS_PresenceTypeBoundingBox(PRESENCE_CROUCH, mins, maxs);

	/* Gladiator passes the literal PRESENCE_CROUCH/client pair to Trace. */
	bsp_trace_t trace = AAS_Trace(start,
		mins,
		maxs,
		end,
		PRESENCE_CROUCH,
		state->client_number);
	return trace.ent == 0;
}

/*
=============
BotAI_ConsoleTeamPlayIsOn

Reconstructs Gladiator's team-command gate from the model/skin team dmflags,
CTF, and explicit teamplay libvars.
=============
*/
static bool BotAI_ConsoleTeamPlayIsOn(void)
{
	int dmflags = (int)LibVarGetValue("dmflags");
	return (dmflags & BOT_CONSOLE_TEAM_DMFLAGS) != 0 ||
		LibVarGetValue("ctf") != 0.0f ||
		LibVarGetValue("teamplay") != 0.0f;
}

/*
=============
BotAI_FindExactConsoleClientByName

Finds the exact case-sensitive NETNAME source used by Gladiator's addressed
team-command gate.
=============
*/
static int BotAI_FindExactConsoleClientByName(const char *name)
{
	return ClientFromName(name);
}

/*
=============
BotAI_ConsoleTeamPlayerCount

Counts named same-team clients for the retail unaddressed-command probability.
=============
*/
static int BotAI_ConsoleTeamPlayerCount(const bot_client_state_t *state)
{
	int count = 0;
	for (int client = 0; client < BotState_ClientCapacity(); ++client)
	{
		const char *name = BotState_ClientName(client);
		/* ref BotNumTeamMates (be_ai2_dmq2.c:1230-1236) calls BotSameTeam
		   per client with the entity number, i.e. client + 1. */
		if (name != NULL && name[0] != '\0' &&
			BotAI_SameTeam(state, client + 1))
		{
			++count;
		}
	}
	return count;
}

/*
=============
BotAI_ConsoleNameAddressesBot

Tests the case-insensitive substring relation against the bot name and its
current 32-byte subteam name.
=============
*/
static bool BotAI_ConsoleNameAddressesBot(const bot_client_state_t *state,
	const char *name)
{
	if (state == NULL || name == NULL || name[0] == '\0')
	{
		return false;
	}

	return StringContainsIndex(BotState_ClientName(state->client_number), name, 0) >= 0 ||
		StringContainsIndex(state->subteam, name, 0) >= 0;
}

/*
=============
BotAI_ConsoleAddressedToBot

Reconstructs sub_10026be0: validates the NETNAME as a teammate, resolves the
context-32 addressee list against the bot/subteam names, and applies the retail
one-over-other-teammates probability to unaddressed commands.
=============
*/
static bool BotAI_ConsoleAddressedToBot(bot_client_state_t *state,
	const bot_match_t *match)
{
	if (state == NULL || match == NULL)
	{
		return false;
	}

	char netname[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	BotMatchVariableSized(match,
		BOT_CONSOLE_MATCH_NETNAME,
		netname,
		(int)sizeof(netname));
	int source_client = BotAI_FindExactConsoleClientByName(netname);
	if (source_client < 0 ||
		/* 0x10026be0 gates on j_sub_10023550(arg1, eax + 1). */
		!BotAI_SameTeam(state, source_client + 1))
	{
		return false;
	}

	if ((match->subtype & BOT_CONSOLE_MATCH_SUBTYPE_ADDRESSED) == 0)
	{
		int teammate_count = BotAI_ConsoleTeamPlayerCount(state);
		if (teammate_count <= 1)
		{
			return true;
		}
		return BotAI_ConsoleRandom() <= 1.0f / (float)(teammate_count - 1);
	}

	char addressee[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	BotMatchVariableSized(match,
		BOT_CONSOLE_MATCH_ADDRESSEE,
		addressee,
		(int)sizeof(addressee));
	while (addressee[0] != '\0')
	{
		char addressee_source[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
		memcpy(addressee_source, addressee, sizeof(addressee_source));
		bot_match_t addressee_match;
		memset(&addressee_match, 0, sizeof(addressee_match));
		if (!BotFindMatch(addressee,
			&addressee_match,
			BOT_CONSOLE_ADDRESSEE_CONTEXT))
		{
			break;
		}

		if (addressee_match.type == BOT_CONSOLE_ADDRESSEE_EVERYONE)
		{
			return true;
		}

		char name[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
		BotMatchVariableSized(&addressee_match,
			BOT_CONSOLE_MATCH_TEAMMATE,
			name,
			(int)sizeof(name));
		if (BotAI_ConsoleNameAddressesBot(state, name))
		{
			return true;
		}

		if (addressee_match.type != BOT_CONSOLE_ADDRESSEE_MULTIPLE_NAMES)
		{
			break;
		}
		BotMatchVariableSized(&addressee_match,
			BOT_CONSOLE_MATCH_MORE,
			addressee,
			(int)sizeof(addressee));
		if (addressee[0] == '\0')
		{
			const char *remainder = addressee_source + strlen(name);
			if (strncmp(remainder, " and ", 5U) == 0)
			{
				remainder += 5;
			}
			else if (strncmp(remainder, ", ", 2U) == 0)
			{
				remainder += 2;
			}
			else
			{
				break;
			}
			strncpy(addressee, remainder, sizeof(addressee));
			addressee[sizeof(addressee) - 1U] = '\0';
		}
	}

	return false;
}

/*
=============
BotAI_FindConsoleClientByName

Finds the first case-insensitive exact client name, then the first client name
containing the requested text, matching Gladiator's team-command lookup.
=============
*/
static int BotAI_FindConsoleClientByName(const char *name)
{
	return FindClientByName((char *)name);
}

/*
=============
BotAI_UpdateConsoleLeadership

Applies Gladiator's paired start/stop team-leadership match side effects,
including its fuzzy client lookup and ST_I variable-selection behavior.
=============
*/
static void BotAI_UpdateConsoleLeadership(bot_client_state_t *state,
	const bot_match_t *match)
{
	if (state == NULL || match == NULL)
	{
		return;
	}

	char teammate[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	BotMatchVariableSized(match,
		BOT_CONSOLE_MATCH_TEAMMATE,
		teammate,
		(int)sizeof(teammate));

	if (match->type == BOT_CONSOLE_MATCH_START_TEAM_LEADERSHIP)
	{
		if ((match->subtype & BOT_CONSOLE_MATCH_SUBTYPE_I) != 0)
		{
			strncpy(state->team_leader, teammate, sizeof(state->team_leader));
			state->team_leader[sizeof(state->team_leader) - 1U] = '\0';
			return;
		}

		int client = BotAI_FindConsoleClientByName(teammate);
		if (client >= 0)
		{
			strcpy(state->team_leader, BotState_ClientName(client));
		}
		return;
	}

	int client;
	if ((match->subtype & BOT_CONSOLE_MATCH_SUBTYPE_I) != 0)
	{
		char netname[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
		BotMatchVariableSized(match,
			BOT_CONSOLE_MATCH_NETNAME,
			netname,
			(int)sizeof(netname));
		client = BotAI_FindConsoleClientByName(netname);
	}
	else
	{
		client = BotAI_FindConsoleClientByName(teammate);
	}

	if (client >= 0 &&
		Q_stricmp(state->team_leader, BotState_ClientName(client)) == 0)
	{
		state->team_leader[0] = '\0';
	}
}

/*
=============
BotAI_ConsoleEnterInitialTeamChat

Constructs a single-variable initial chat, then enters Gladiator's team-chat
path. Retail BotEnterChat is a no-op when the named template was unavailable.
=============
*/
void BotAI_ConsoleEnterInitialTeamChat(bot_client_state_t *state,
	const char *type,
	const char *variable)
{
	if (state == NULL || state->chat_state == NULL || type == NULL)
	{
		return;
	}

	BotInitialChat(state->chat_state, type, variable, NULL);
	BotEnterChat(state->chat_state,
		state->client_number,
		BOT_CONSOLE_CHAT_TEAM);
}

/*
=============
BotAI_ConsoleEnterInitialTeamChat2

Constructs a two-variable retail initial chat and enters the team destination.
=============
*/
void BotAI_ConsoleEnterInitialTeamChat2(bot_client_state_t *state,
	const char *type,
	const char *first,
	const char *second)
{
	if (state == NULL || state->chat_state == NULL || type == NULL)
	{
		return;
	}

	BotInitialChat(state->chat_state,
		type,
		first,
		second,
		NULL);
	BotEnterChat(state->chat_state,
		state->client_number,
		BOT_CONSOLE_CHAT_TEAM);
}

/*
=============
BotAI_ConsoleCreateWaypoint

Allocates the compact semantic mirror of Gladiator's named waypoint object.
=============
*/
static bot_console_waypoint_t *BotAI_ConsoleCreateWaypoint(const char *name,
	const vec3_t origin,
	int areanum)
{
	const char *waypoint_name = name != NULL ? name : "";
	size_t name_bytes = strlen(waypoint_name) + 1U;
	bot_console_waypoint_t *waypoint = GetMemory(sizeof(*waypoint) + name_bytes);
	if (waypoint == NULL)
	{
		return NULL;
	}

	waypoint->name = waypoint->name_storage;
	memcpy(waypoint->name, waypoint_name, name_bytes);
	if (origin != NULL)
	{
		VectorCopy(origin, waypoint->goal.origin);
	}
	else
	{
		VectorClear(waypoint->goal.origin);
	}
	waypoint->goal.areanum = areanum;
	VectorSet(waypoint->goal.mins, -8.0f, -8.0f, -8.0f);
	VectorSet(waypoint->goal.maxs, 8.0f, 8.0f, 8.0f);
	waypoint->next = NULL;
	waypoint->prev = NULL;
	return waypoint;
}

/*
=============
BotAI_ConsoleFindWaypoint

Finds a named checkpoint with Gladiator's case-insensitive comparison.
=============
*/
static bot_console_waypoint_t *BotAI_ConsoleFindWaypoint(
	bot_console_waypoint_t *waypoints,
	const char *name)
{
	if (name == NULL)
	{
		return NULL;
	}

	for (bot_console_waypoint_t *waypoint = waypoints;
		waypoint != NULL;
		waypoint = waypoint->next)
	{
		if (Q_stricmp(waypoint->name, name) == 0)
		{
			return waypoint;
		}
	}
	return NULL;
}

/*
=============
BotAI_ConsoleUnlinkCheckpoint

Unlinks one checkpoint before retail replaces a duplicate name.
=============
*/
static void BotAI_ConsoleUnlinkCheckpoint(bot_client_state_t *state,
	bot_console_waypoint_t *waypoint)
{
	if (state == NULL || waypoint == NULL)
	{
		return;
	}

	if (waypoint->prev != NULL)
	{
		waypoint->prev->next = waypoint->next;
	}
	else
	{
		state->checkpoints = waypoint->next;
	}
	if (waypoint->next != NULL)
	{
		waypoint->next->prev = waypoint->prev;
	}
	waypoint->next = NULL;
	waypoint->prev = NULL;
}

/*
=============
BotAI_ConsoleResolveTeamGoal

Resolves a key-area name through static level items and then remembered
checkpoints, matching sub_10026770's lookup order.
=============
*/
static bool BotAI_ConsoleResolveTeamGoal(bot_client_state_t *state,
	char *name,
	bot_goal_t *goal)
{
	if (state == NULL || name == NULL || goal == NULL)
	{
		return false;
	}

	int index = 0;
	while (name[0] != '\0')
	{
		bot_goal_t candidate;
		memset(&candidate, 0, sizeof(candidate));
		int next = BotGetLevelItemGoal(index, name, &candidate);
		if (next < 0)
		{
			break;
		}
		if ((candidate.flags & GFL_DROPPED) == 0)
		{
			memcpy(goal, &candidate, sizeof(*goal));
			return true;
		}
		if (next <= index)
		{
			break;
		}
		index = next;
	}

	bot_console_waypoint_t *checkpoint = BotAI_ConsoleFindWaypoint(
		state->checkpoints,
		name);
	if (checkpoint == NULL)
	{
		return false;
	}
	memcpy(goal, &checkpoint->goal, sizeof(*goal));
	return true;
}

/*
=============
BotAI_ConsoleTeamGoalTime

Parses an optional minutes/seconds match and returns its absolute deadline.
=============
*/
static float BotAI_ConsoleTeamGoalTime(const bot_match_t *match)
{
	if (match == NULL ||
		(match->subtype & BOT_CONSOLE_MATCH_SUBTYPE_TIME) == 0)
	{
		return 0.0f;
	}

	char time_text[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	BotMatchVariableSized(match,
		BOT_CONSOLE_MATCH_TIME,
		time_text,
		(int)sizeof(time_text));
	bot_match_t time_match;
	memset(&time_match, 0, sizeof(time_match));
	if (!BotFindMatch(time_text, &time_match, BOT_CONSOLE_TIME_CONTEXT))
	{
		return 0.0f;
	}

	char number[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	BotMatchVariableSized(&time_match,
		BOT_CONSOLE_MATCH_TIME,
		number,
		(int)sizeof(number));
	float duration = (float)atof(number);
	if (time_match.type == BOT_CONSOLE_MATCH_MINUTES)
	{
		duration *= 60.0f;
	}
	else if (time_match.type != BOT_CONSOLE_MATCH_SECONDS)
	{
		return 0.0f;
	}
	return duration > 0.0f ? AAS_Time() + duration : 0.0f;
}

/*
=============
BotAI_ConsoleSetPointGoal

Builds the point-sized team goal used by camp-here and camp-there.
=============
*/
void BotAI_ConsoleSetPointGoal(bot_goal_t *goal,
	const vec3_t origin,
	int areanum,
	int entitynum)
{
	VectorCopy(origin, goal->origin);
	goal->areanum = areanum;
	VectorSet(goal->mins, -8.0f, -8.0f, -8.0f);
	VectorSet(goal->maxs, 8.0f, 8.0f, 8.0f);
	goal->entitynum = entitynum;
}

/*
=============
BotAI_ConsoleClientOrigin

Reads a team-command source from the live entity cache, with a bot-client
snapshot fallback for reconstructed semantic peers.
=============
*/
static bool BotAI_ConsoleClientOrigin(int client, vec3_t origin)
{
	int entity = client + 1;
	if (origin == NULL || client < 0 || entity >= BOT_INTERFACE_MAX_ENTITIES)
	{
		return false;
	}
	bot_updateentity_t cached = {0};
	if (BotInterface_ReadEntityCache(entity, &cached))
	{
		VectorCopy(cached.origin, origin);
		return true;
	}

	bot_client_state_t *source = BotState_Get(client);
	if (source == NULL || !source->client_update_valid)
	{
		return false;
	}
	VectorCopy(source->last_client_update.origin, origin);
	return true;
}

/*
=============
BotAI_ConsoleHandleHelpAccompany

Reconstructs cases 3 and 4, including teammate-pronoun parsing, fuzzy target
lookup, the live-target/near-item goal fallback, and their distinct LTG times.
=============
*/
static void BotAI_ConsoleHandleHelpAccompany(bot_client_state_t *state,
	const bot_match_t *match)
{
	char teammate[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	BotMatchVariableSized(match,
		BOT_CONSOLE_MATCH_TEAMMATE,
		teammate,
		(int)sizeof(teammate));

	bot_match_t teammate_match;
	memset(&teammate_match, 0, sizeof(teammate_match));
	bool other = true;
	int target_client;
	char netname[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	netname[0] = '\0';
	if (BotFindMatch(teammate,
		&teammate_match,
		BOT_CONSOLE_TEAMMATE_CONTEXT) &&
		teammate_match.type == BOT_CONSOLE_MATCH_ME)
	{
		BotMatchVariableSized(match,
			BOT_CONSOLE_MATCH_NETNAME,
			netname,
			(int)sizeof(netname));
		target_client = BotAI_FindExactConsoleClientByName(netname);
		other = false;
	}
	else
	{
		target_client = BotAI_FindConsoleClientByName(teammate);
		if (target_client == state->client_number)
		{
			return;
		}
	}

	if (target_client < 0)
	{
		BotAI_ConsoleEnterInitialTeamChat(state,
			"whois",
			other ? teammate : netname);
		return;
	}

	state->team_goal.entitynum = 0;
	vec3_t target_origin;
	if (BotAI_ConsoleClientOrigin(target_client, target_origin))
	{
		int areanum = AAS_PointAreaNum(target_origin);
		if (areanum != 0 && AAS_AreaReachability(areanum) != 0)
		{
			BotAI_ConsoleSetPointGoal(&state->team_goal,
				target_origin,
				areanum,
				target_client + 1);
		}
	}

	if (state->team_goal.entitynum == 0 &&
		(match->subtype & BOT_CONSOLE_MATCH_SUBTYPE_NEARITEM) != 0)
	{
		char item[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
		BotMatchVariableSized(match,
			BOT_CONSOLE_MATCH_ITEM,
			item,
			(int)sizeof(item));
		if (!BotAI_ConsoleResolveTeamGoal(state, item, &state->team_goal))
		{
			BotAI_ConsoleEnterInitialTeamChat(state, "cannotfind", item);
			return;
		}
	}

	if (state->team_goal.entitynum == 0)
	{
		BotAI_ConsoleEnterInitialTeamChat(state,
			other ? "whereis" : "whereareyou",
			other ? teammate : netname);
		return;
	}

	state->ltg_teammate = target_client;
	state->team_goal_number = state->team_goal.number;
	state->teammate_visible_time = AAS_Time();
	state->team_message_time = AAS_Time() +
		2.0f * BotAI_ConsoleRandom();
	state->team_goal_time = BotAI_ConsoleTeamGoalTime(match);
	if (match->type == BOT_CONSOLE_MATCH_HELP)
	{
		state->ltg_type = 1;
		if (state->team_goal_time == 0.0f)
		{
			state->team_goal_time = AAS_Time() +
				BOT_CONSOLE_HELP_DURATION;
		}
		return;
	}

	state->ltg_type = 2;
	if (state->team_goal_time == 0.0f)
	{
		state->team_goal_time = AAS_Time() +
			BOT_CONSOLE_ACCOMPANY_DURATION;
	}
	state->formation_dist = BOT_CONSOLE_ACCOMPANY_DISTANCE;
	state->arrive_time = 0.0f;
}

/*
=============
BotAI_ConsoleHandleDefendKeyArea

Reconstructs case 5's key-area resolution, failure chat, LTG deadline, and
defend-away reset.
=============
*/
static void BotAI_ConsoleHandleDefendKeyArea(bot_client_state_t *state,
	const bot_match_t *match)
{
	char keyarea[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	BotMatchVariableSized(match,
		BOT_CONSOLE_MATCH_KEYAREA,
		keyarea,
		(int)sizeof(keyarea));
	if (!BotAI_ConsoleResolveTeamGoal(state, keyarea, &state->team_goal))
	{
		BotAI_ConsoleEnterInitialTeamChat(state, "cannotfind", keyarea);
		return;
	}

	state->team_goal_number = state->team_goal.number;
	state->ltg_type = 3;
	state->team_message_time = AAS_Time() +
		2.0f * BotAI_ConsoleRandom();
	state->team_goal_time = BotAI_ConsoleTeamGoalTime(match);
	if (state->team_goal_time == 0.0f)
	{
		state->team_goal_time = AAS_Time() +
			BOT_CONSOLE_DEFEND_DURATION;
	}
	state->defend_away_time = 0.0f;
}

/*
=============
BotAI_ConsoleCTFFlagGoals

Loads the static Red and Blue Flag goals that the CTF command gate and direct
long-term-goal branches share.
=============
*/
bool BotAI_ConsoleCTFFlagGoals(bot_goal_t *red_flag, bot_goal_t *blue_flag)
{
	if (red_flag == NULL || blue_flag == NULL)
	{
		return false;
	}

	char red_name[] = "Red Flag";
	char blue_name[] = "Blue Flag";
	memset(red_flag, 0, sizeof(*red_flag));
	memset(blue_flag, 0, sizeof(*blue_flag));
	return BotGetLevelItemGoal(-1, red_name, red_flag) >= 0 &&
		red_flag->areanum != 0 &&
		BotGetLevelItemGoal(-1, blue_name, blue_flag) >= 0 &&
		blue_flag->areanum != 0;
}

/*
=============
BotAI_ConsoleCTFFlagsAvailable

Mirrors the case 6/7 gate on ctf plus non-zero cached Red and Blue Flag goal
areas. The reconstruction resolves the static goals on demand from botlib's
level-item table because that table owns the semantic cache.
=============
*/
static bool BotAI_ConsoleCTFFlagsAvailable(void)
{
	if (LibVarGetValue("ctf") == 0.0f)
	{
		return false;
	}

	bot_goal_t red_flag;
	bot_goal_t blue_flag;
	return BotAI_ConsoleCTFFlagGoals(&red_flag, &blue_flag);
}

/*
=============
BotAI_SelectAutomaticCTFGoal

Reconstructs sub_10026440's Seek-LTG CTF choice: a carrier rushes home;
otherwise an unassigned aggressive bot selects the enemy flag, defends its
home flag, or waits through the fixed get-flag-away interval.
=============
*/
void BotAI_SelectAutomaticCTFGoal(bot_client_state_t *state)
{
	if (state == NULL)
	{
		return;
	}

	float ctf = LibVarGetValue("ctf");
	if (ctf == 0.0f || isnan(ctf))
	{
		return;
	}

	float now = AAS_Time();
	if (BotAI_CarryingFlag(state) != 0)
	{
		if (state->ltg_type != BOT_LTG_RUSH_BASE)
		{
			state->ltg_type = BOT_LTG_RUSH_BASE;
			state->rush_base_away_time = 0.0f;
			state->team_goal_time = now + 120.0f;
		}
		return;
	}

	if (now < state->get_flag_away_time ||
		(state->ltg_type >= 1 && state->ltg_type <= 7) ||
		BotAI_Aggression(state) < 50.0f)
	{
		return;
	}

	state->team_message_time = now + 2.0f * BotAI_LongTermGoalRandom();
	float selection = BotAI_LongTermGoalRandom();
	bot_goal_t red_flag;
	bot_goal_t blue_flag;
	bool flags_available = BotAI_ConsoleCTFFlagGoals(&red_flag, &blue_flag);
	if (selection < 0.33f && flags_available)
	{
		state->ltg_type = BOT_LTG_GET_FLAG;
		state->team_goal_time = now + 180.0f;
		return;
	}

	if (selection < 0.66f && flags_available)
	{
		state->team_goal = BotAI_CTFTeam(state) == 1 ? red_flag : blue_flag;
		state->team_goal_number = state->team_goal.number;
		state->ltg_type = 3;
		state->defend_away_time = 0.0f;
		state->team_goal_time = now + 120.0f;
		return;
	}

	state->ltg_type = 0;
	state->get_flag_away_time = now + 60.0f;
}

/*
=============
BotAI_ConsoleCommitCTFOrder

Applies the retail case 6 rush-base or case 7 get-flag LTG, message deadline,
fixed goal duration, and rush-away reset boundary.
=============
*/
static void BotAI_ConsoleCommitCTFOrder(bot_client_state_t *state,
	int match_type)
{
	state->team_message_time = AAS_Time() +
		2.0f * BotAI_ConsoleRandom();
	if (match_type == BOT_CONSOLE_MATCH_RUSH_BASE)
	{
		state->ltg_type = 5;
		state->rush_base_away_time = 0.0f;
		state->team_goal_time = AAS_Time() +
			BOT_CONSOLE_RUSH_BASE_DURATION;
		return;
	}

	state->ltg_type = 4;
	state->team_goal_time = AAS_Time() +
		BOT_CONSOLE_GET_FLAG_DURATION;
}

/*
=============
BotAI_ConsoleCommitCampGoal

Commits case 19's resolved goal, teammate, LTG type, and retail deadlines.
=============
*/
static void BotAI_ConsoleCommitCampGoal(bot_client_state_t *state,
	const bot_match_t *match,
	const bot_goal_t *goal,
	int source_client)
{
	memcpy(&state->team_goal, goal, sizeof(state->team_goal));
	state->team_goal_number = goal->number;
	state->team_message_time = AAS_Time() + 2.0f * BotAI_ConsoleRandom();
	state->ltg_type = 6;
	state->ltg_teammate = source_client;
	state->team_goal_time = BotAI_ConsoleTeamGoalTime(match);
	if (state->team_goal_time == 0.0f)
	{
		state->team_goal_time = AAS_Time() +
			BOT_CONSOLE_DEFAULT_TEAM_GOAL_DURATION;
	}
	state->arrive_time = 0.0f;
}

/*
=============
BotAI_ConsoleHandleCamp

Reconstructs match case 19's sender lookup and there/here/key-area branches.
=============
*/
static void BotAI_ConsoleHandleCamp(bot_client_state_t *state,
	const bot_match_t *match)
{
	char netname[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	BotMatchVariableSized(match,
		BOT_CONSOLE_MATCH_NETNAME,
		netname,
		(int)sizeof(netname));
	int source_client = BotAI_FindConsoleClientByName(netname);
	if (source_client < 0)
	{
		BotAI_ConsoleEnterInitialTeamChat(state, "whois", netname);
		return;
	}

	char keyarea[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	BotMatchVariableSized(match,
		BOT_CONSOLE_MATCH_KEYAREA,
		keyarea,
		(int)sizeof(keyarea));
	bot_goal_t goal;
	memcpy(&goal, &state->team_goal, sizeof(goal));
	if ((match->subtype & BOT_CONSOLE_MATCH_SUBTYPE_THERE) != 0)
	{
		BotAI_ConsoleSetPointGoal(&goal,
			state->last_client_update.origin,
			AAS_PointAreaNum(state->last_client_update.origin),
			state->entity_number);
	}
	else if ((match->subtype & BOT_CONSOLE_MATCH_SUBTYPE_HERE) != 0)
	{
		if (source_client == state->client_number)
		{
			return;
		}
		state->team_goal.entitynum = 0;
		goal.entitynum = 0;

		vec3_t source_origin;
		int areanum = 0;
		bool visible = false;
		if (BotAI_ConsoleClientOrigin(source_client, source_origin))
		{
			areanum = AAS_PointAreaNum(source_origin);
			vec3_t eye;
			BotInterface_ClientEyePosition(state, eye);
			visible = areanum != 0 &&
				AAS_AreaReachability(areanum) != 0 &&
				BotInterface_HasLineOfSight(eye,
					source_origin,
					state->entity_number,
					source_client + 1);
		}
		if (!visible)
		{
			BotAI_ConsoleEnterInitialTeamChat(state,
				"whereareyou",
				netname);
			return;
		}

		BotAI_ConsoleSetPointGoal(&goal,
			source_origin,
			areanum,
			source_client + 1);
	}
	else if (!BotAI_ConsoleResolveTeamGoal(state, keyarea, &goal))
	{
		BotAI_ConsoleEnterInitialTeamChat(state, "cannotfind", keyarea);
		return;
	}

	BotAI_ConsoleCommitCampGoal(state,
		match,
		&goal,
		source_client);
}

/*
=============
BotAI_ConsoleHandleCheckpoint

Stores case 20 checkpoints independently of addressing and emits only the
addressed invalid/confirmation team chats.
=============
*/
static void BotAI_ConsoleHandleCheckpoint(bot_client_state_t *state,
	const bot_match_t *match)
{
	char position[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	BotMatchVariableSized(match,
		BOT_CONSOLE_MATCH_POSITION,
		position,
		(int)sizeof(position));
	vec3_t origin;
	VectorClear(origin);
	(void)sscanf(position,
		"%f %f %f",
		&origin[0],
		&origin[1],
		&origin[2]);
	origin[2] += 0.5f;
	int areanum = AAS_PointAreaNum(origin);
	if (areanum == 0)
	{
		if (BotAI_ConsoleAddressedToBot(state, match))
		{
			BotAI_ConsoleEnterInitialTeamChat(state,
				"checkpoint_invalid",
				NULL);
		}
		return;
	}

	char name[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	BotMatchVariableSized(match,
		BOT_CONSOLE_MATCH_NAME,
		name,
		(int)sizeof(name));
	bot_console_waypoint_t *old = BotAI_ConsoleFindWaypoint(
		state->checkpoints,
		name);
	if (old != NULL)
	{
		BotAI_ConsoleUnlinkCheckpoint(state, old);
		FreeMemory(old);
	}

	bot_console_waypoint_t *checkpoint = BotAI_ConsoleCreateWaypoint(name,
		origin,
		areanum);
	if (checkpoint == NULL)
	{
		return;
	}
	checkpoint->next = state->checkpoints;
	if (state->checkpoints != NULL)
	{
		state->checkpoints->prev = checkpoint;
	}
	state->checkpoints = checkpoint;

	if (BotAI_ConsoleAddressedToBot(state, match))
	{
		char gps[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
		snprintf(gps,
			sizeof(gps),
			"%1.0f %1.0f %1.0f",
			checkpoint->goal.origin[0],
			checkpoint->goal.origin[1],
			checkpoint->goal.origin[2]);
		BotAI_ConsoleEnterInitialTeamChat2(state,
			"checkpoint_confirm",
			name,
			gps);
	}
}

/*
=============
BotAI_ConsoleClearPatrolPoints

Mirrors case 21's failure store, which clears only the patrol-list head and
leaves the current-point and flag fields untouched.
=============
*/
static void BotAI_ConsoleClearPatrolPoints(bot_client_state_t *state)
{
	state->patrol_points = NULL;
}

/*
=============
BotAI_ConsoleRecoverPatrolVariables

Recovers patrol variables when an empty optional match alternative leaves the
reconstructed matcher without a KEYAREA or MORE public span.
=============
*/
static int BotAI_ConsoleRecoverPatrolVariables(const char *text,
	char *keyarea,
	size_t keyarea_size,
	char *more,
	size_t more_size)
{
	if (keyarea == NULL || keyarea_size == 0U ||
		more == NULL || more_size == 0U)
	{
		return 0;
	}
	keyarea[0] = '\0';
	more[0] = '\0';
	if (text == NULL)
	{
		return 0;
	}

	const char *cursor = text;
	const char *prefixes[] = {"the ", "checkpoint ", "waypoint "};
	for (size_t index = 0; index < sizeof(prefixes) / sizeof(prefixes[0]); ++index)
	{
		size_t prefix_length = strlen(prefixes[index]);
		if (Q_strnicmp(cursor, prefixes[index], prefix_length) == 0)
		{
			cursor += prefix_length;
			break;
		}
	}

	strncpy(keyarea, cursor, keyarea_size - 1U);
	keyarea[keyarea_size - 1U] = '\0';
	int separator = StringContainsIndex(keyarea, " to ", 0);
	int back_to_start = StringContainsIndex(keyarea, " and back to the start", 0);
	if (separator >= 0 &&
		(back_to_start < 0 || separator < back_to_start))
	{
		const char *remainder = keyarea + separator + 4;
		strncpy(more, remainder, more_size - 1U);
		more[more_size - 1U] = '\0';
		keyarea[separator] = '\0';
		return BOT_CONSOLE_MATCH_SUBTYPE_MORE;
	}

	const struct
	{
		const char *suffix;
		int subtype;
	} suffixes[] = {
		{" and back to the start", BOT_CONSOLE_MATCH_SUBTYPE_BACK},
		{" and back", BOT_CONSOLE_MATCH_SUBTYPE_BACK},
		{" and reverse", BOT_CONSOLE_MATCH_SUBTYPE_REVERSE},
	};
	for (size_t index = 0; index < sizeof(suffixes) / sizeof(suffixes[0]); ++index)
	{
		size_t text_length = strlen(keyarea);
		size_t suffix_length = strlen(suffixes[index].suffix);
		if (text_length >= suffix_length &&
			Q_stricmp(keyarea + text_length - suffix_length,
				suffixes[index].suffix) == 0)
		{
			keyarea[text_length - suffix_length] = '\0';
			return suffixes[index].subtype;
		}
	}

	return 0;
}

/*
=============
BotAI_ConsoleGetPatrolPoints

Parses and resolves the chained patrol-key-area grammar used by sub_10026990.
=============
*/
static bool BotAI_ConsoleGetPatrolPoints(bot_client_state_t *state,
	const bot_match_t *match)
{
	char remaining[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	BotMatchVariableSized(match,
		BOT_CONSOLE_MATCH_KEYAREA,
		remaining,
		(int)sizeof(remaining));
	bot_console_waypoint_t *points = NULL;
	bot_console_waypoint_t *tail = NULL;
	int point_count = 0;
	int patrol_flags = 0;

	for (;;)
	{
		bot_match_t point_match;
		memset(&point_match, 0, sizeof(point_match));
		if (!BotFindMatch(remaining,
			&point_match,
			BOT_CONSOLE_PATROL_CONTEXT) ||
			point_match.type != BOT_CONSOLE_MATCH_PATROL_KEYAREA)
		{
			EA_SayTeam(state->client_number, "what do you say?");
			BotState_FreeConsoleWaypoints(points);
			BotAI_ConsoleClearPatrolPoints(state);
			return false;
		}

		char keyarea[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
		BotMatchVariableSized(&point_match,
			BOT_CONSOLE_MATCH_KEYAREA,
			keyarea,
			(int)sizeof(keyarea));
		int point_subtype = point_match.subtype;
		char recovered_more[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
		BotMatchVariableSized(&point_match,
			BOT_CONSOLE_MATCH_MORE,
			recovered_more,
			(int)sizeof(recovered_more));
		if (keyarea[0] == '\0' ||
			((point_subtype & BOT_CONSOLE_MATCH_SUBTYPE_MORE) != 0 &&
				recovered_more[0] == '\0'))
		{
			char fallback_keyarea[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
			char fallback_more[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
			int fallback_subtype = BotAI_ConsoleRecoverPatrolVariables(remaining,
				fallback_keyarea,
				sizeof(fallback_keyarea),
				fallback_more,
				sizeof(fallback_more));
			if (keyarea[0] == '\0')
			{
				strncpy(keyarea,
					fallback_keyarea,
					sizeof(keyarea) - 1U);
				keyarea[sizeof(keyarea) - 1U] = '\0';
			}
			if (recovered_more[0] == '\0')
			{
				strncpy(recovered_more,
					fallback_more,
					sizeof(recovered_more) - 1U);
				recovered_more[sizeof(recovered_more) - 1U] = '\0';
			}
			if (point_subtype == 0 && fallback_subtype != 0)
			{
				point_subtype = fallback_subtype;
			}
		}
		if (keyarea[0] == '\0')
		{
			EA_SayTeam(state->client_number, "what do you say?");
			BotState_FreeConsoleWaypoints(points);
			BotAI_ConsoleClearPatrolPoints(state);
			return false;
		}
		bot_goal_t goal;
		memset(&goal, 0, sizeof(goal));
		if (!BotAI_ConsoleResolveTeamGoal(state, keyarea, &goal))
		{
			BotAI_ConsoleEnterInitialTeamChat(state, "cannotfind", keyarea);
			BotState_FreeConsoleWaypoints(points);
			BotAI_ConsoleClearPatrolPoints(state);
			return false;
		}

		bot_console_waypoint_t *point = BotAI_ConsoleCreateWaypoint(keyarea,
			goal.origin,
			goal.areanum);
		if (point == NULL)
		{
			BotState_FreeConsoleWaypoints(points);
			return false;
		}
		if (tail != NULL)
		{
			tail->next = point;
			point->prev = tail;
		}
		else
		{
			points = point;
		}
		tail = point;
		++point_count;

		if ((point_subtype & BOT_CONSOLE_MATCH_SUBTYPE_BACK) != 0)
		{
			patrol_flags = BOT_CONSOLE_PATROL_LOOP;
			break;
		}
		if ((point_subtype & BOT_CONSOLE_MATCH_SUBTYPE_REVERSE) != 0)
		{
			patrol_flags = BOT_CONSOLE_PATROL_REVERSE;
			break;
		}
		if ((point_subtype & BOT_CONSOLE_MATCH_SUBTYPE_MORE) == 0)
		{
			break;
		}
		if (recovered_more[0] != '\0')
		{
			strncpy(remaining, recovered_more, sizeof(remaining) - 1U);
			remaining[sizeof(remaining) - 1U] = '\0';
		}
		else
		{
			remaining[0] = '\0';
		}
	}

	if (point_count < 2)
	{
		EA_SayTeam(state->client_number,
			"I need more key points to patrol\n");
		BotState_FreeConsoleWaypoints(points);
		return false;
	}

	BotState_FreeConsoleWaypoints(state->patrol_points);
	state->patrol_points = points;
	state->current_patrol_point = points;
	state->patrol_flags = patrol_flags;
	return true;
}

/*
=============
BotAI_ConsoleCommitPatrol

Commits case 21's LTG and message/goal deadlines after waypoint parsing.
=============
*/
static void BotAI_ConsoleCommitPatrol(bot_client_state_t *state,
	const bot_match_t *match)
{
	state->team_message_time = AAS_Time() + 2.0f * BotAI_ConsoleRandom();
	state->ltg_type = 7;
	state->team_goal_time = BotAI_ConsoleTeamGoalTime(match);
	if (state->team_goal_time == 0.0f)
	{
		state->team_goal_time = AAS_Time() +
			BOT_CONSOLE_DEFAULT_TEAM_GOAL_DURATION;
	}
}

/*
=============
BotAI_ConsoleEasyClientName

Reconstructs sub_10021860's teammate-name cleanup: high-bit stripping,
space/clan-tag/Mr removal, lower-casing, and alphanumeric/underscore filtering.
=============
*/
void BotAI_ConsoleEasyClientName(int client,
	char *output,
	size_t output_size)
{
	if (output == NULL || output_size == 0U)
	{
		return;
	}

	char name[128];
	snprintf(name, sizeof(name), "%s", BotState_ClientName(client));
	for (char *character = name; *character != '\0'; ++character)
	{
		*character = (char)((unsigned char)*character & 0x7fU);
	}

	char *space = strstr(name, " ");
	while (space != NULL)
	{
		memmove(space, space + 1, strlen(space + 1) + 1U);
		space = strstr(name, " ");
	}

	char *open_bracket = strstr(name, "[");
	char *close_bracket = strstr(name, "]");
	if (open_bracket != NULL && close_bracket != NULL)
	{
		if (close_bracket > open_bracket)
		{
			memmove(open_bracket,
				close_bracket + 1,
				strlen(close_bracket + 1) + 1U);
		}
		else
		{
			memmove(close_bracket,
				open_bracket + 1,
				strlen(open_bracket + 1) + 1U);
		}
	}

	if ((name[0] == 'm' || name[0] == 'M') &&
		(name[1] == 'r' || name[1] == 'R'))
	{
		memmove(name, name + 2, strlen(name + 2) + 1U);
	}

	char *character = name;
	while (*character != '\0')
	{
		if ((*character >= 'a' && *character <= 'z') ||
			(*character >= '0' && *character <= '9') ||
			*character == '_')
		{
			++character;
		}
		else if (*character >= 'A' && *character <= 'Z')
		{
			*character = (char)(*character + ('a' - 'A'));
			++character;
		}
		else
		{
			memmove(character,
				character + 1,
				strlen(character + 1) + 1U);
		}
	}

	strncpy(output, name, output_size - 1U);
	output[output_size - 1U] = '\0';
}

/*
=============
BotAI_ConsoleReportLongTermGoal

Reports Gladiator long-term-goal types one through seven through the exact
initial-chat types and team-chat destination used by match case 11.
=============
*/
static void BotAI_ConsoleReportLongTermGoal(bot_client_state_t *state)
{
	if (state == NULL)
	{
		return;
	}

	char teammate[BOT_CONSOLE_EASY_NAME_CHARS];
	char goal_name[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	switch (state->ltg_type)
	{
	case 1:
		BotAI_ConsoleEasyClientName(state->ltg_teammate,
			teammate,
			sizeof(teammate));
		BotAI_ConsoleEnterInitialTeamChat(state, "helping", teammate);
		return;
	case 2:
		BotAI_ConsoleEasyClientName(state->ltg_teammate,
			teammate,
			sizeof(teammate));
		BotAI_ConsoleEnterInitialTeamChat(state, "accompanying", teammate);
		return;
	case 3:
		BotGoalName(state->team_goal_number,
			goal_name,
			(int)sizeof(goal_name));
		BotAI_ConsoleEnterInitialTeamChat(state, "defending", goal_name);
		return;
	case 4:
		BotAI_ConsoleEnterInitialTeamChat(state, "capturingflag", NULL);
		return;
	case 5:
		BotAI_ConsoleEnterInitialTeamChat(state, "rushingbase", NULL);
		return;
	case 6:
		BotAI_ConsoleEnterInitialTeamChat(state, "camping", NULL);
		return;
	case 7:
		BotAI_ConsoleEnterInitialTeamChat(state, "patrolling", NULL);
		return;
	default:
		return;
	}
}

/*
=============
BotAI_ConsoleJoinSubteam

Copies TEAMNAME into Gladiator's 32-byte subteam slot, pins byte 31 to NUL,
then announces the untruncated captured team name.
=============
*/
static void BotAI_ConsoleJoinSubteam(bot_client_state_t *state,
	const bot_match_t *match)
{
	char teamname[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	BotMatchVariableSized(match,
		BOT_CONSOLE_MATCH_TEAMMATE,
		teamname,
		(int)sizeof(teamname));
	strncpy(state->subteam, teamname, sizeof(state->subteam));
	state->subteam[sizeof(state->subteam) - 1U] = '\0';
	BotAI_ConsoleEnterInitialTeamChat(state, "joinedteam", teamname);
}

/*
=============
BotAI_ConsoleLeaveSubteam

Announces a non-empty current subteam and clears the first retail dword of the
32-byte slot regardless of whether a message was constructed.
=============
*/
static void BotAI_ConsoleLeaveSubteam(bot_client_state_t *state)
{
	if (state == NULL || state->chat_state == NULL)
	{
		return;
	}

	/*
	 * The inline-strlen guard at 0x10027a7c covers only BotInitialChat
	 * (0x10027a8c); BotEnterChat at 0x10027aa1 and the slot clear at
	 * 0x10027aaf sit at the outer indentation and run unconditionally.  The
	 * fused helper would skip the flush, leaving a staged chat pending so it
	 * later goes out through EA_Say instead of EA_SayTeam.
	 */
	if (state->subteam[0] != '\0')
	{
		BotInitialChat(state->chat_state, "leftteam", state->subteam, NULL);
	}
	BotEnterChat(state->chat_state,
		state->client_number,
		BOT_CONSOLE_CHAT_TEAM);
	memset(state->subteam, 0, sizeof(int));
}

/*
=============
BotAI_ConsoleSetFormationSpace

Converts NUMBER using the retail feet/metre factors and replaces values outside
the inclusive 48..500 range with the 100-unit default.
=============
*/
static void BotAI_ConsoleSetFormationSpace(bot_client_state_t *state,
	const bot_match_t *match)
{
	char number[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	BotMatchVariableSized(match,
		BOT_CONSOLE_MATCH_NUMBER,
		number,
		(int)sizeof(number));
	double space = atof(number) *
		((match->subtype & BOT_CONSOLE_MATCH_SUBTYPE_FEET) != 0
			? 9.7536000000000005
			: 32.0);
	if (space < 48.0 || space > 500.0)
	{
		space = 100.0;
	}
	state->formation_dist = (float)space;
}

/*
=============
BotAI_MatchConsoleMessage

Classifies normalised console text in Gladiator's combined obituary,
enter-game, and initial-team-chat context, applying the reconstructed death,
team command, subteam, formation, and dismissal effects.
=============
*/
static bool BotAI_MatchConsoleMessage(bot_client_state_t *state, const char *message)
{
	if (state == NULL || message == NULL)
	{
		return false;
	}

	bot_match_t match;
	memset(&match, 0, sizeof(match));
	if (!BotFindMatch(message, &match, BOT_CONSOLE_MATCH_CONTEXT))
	{
		return false;
	}

	if (match.type < 1 || match.type > 21)
	{
		BotInterface_Printf(PRT_MESSAGE, "unknown match type\n");
		return true;
	}

	switch (match.type)
	{
	case BOT_CONSOLE_MATCH_DEATH:
	{
		char victim[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
		BotMatchVariableSized(&match,
			BOT_CONSOLE_MATCH_VICTIM,
			victim,
			(int)sizeof(victim));
		int victim_client = ClientFromName(victim);
		if (victim_client == state->client_number)
		{
			state->bot_death_type = match.subtype;
		}
		else if (victim_client + 1 == state->combat.current_enemy)
		{
			state->enemy_death_type = match.subtype;
			state->combat.enemy_death_time = AAS_Time();
		}
		break;
	}
	case BOT_CONSOLE_MATCH_HELP:
	case BOT_CONSOLE_MATCH_ACCOMPANY:
		if (BotAI_ConsoleTeamPlayIsOn() &&
			BotAI_ConsoleAddressedToBot(state, &match))
		{
			BotAI_ConsoleHandleHelpAccompany(state, &match);
		}
		break;
	case BOT_CONSOLE_MATCH_DEFEND_KEY_AREA:
		if (BotAI_ConsoleTeamPlayIsOn() &&
			BotAI_ConsoleAddressedToBot(state, &match))
		{
			BotAI_ConsoleHandleDefendKeyArea(state, &match);
		}
		break;
	case BOT_CONSOLE_MATCH_RUSH_BASE:
	case BOT_CONSOLE_MATCH_GET_FLAG:
		if (BotAI_ConsoleCTFFlagsAvailable() &&
			BotAI_ConsoleAddressedToBot(state, &match))
		{
			BotAI_ConsoleCommitCTFOrder(state, match.type);
		}
		break;
	case BOT_CONSOLE_MATCH_START_TEAM_LEADERSHIP:
	case BOT_CONSOLE_MATCH_STOP_TEAM_LEADERSHIP:
		if (BotAI_ConsoleTeamPlayIsOn())
		{
			BotAI_UpdateConsoleLeadership(state, &match);
		}
		break;
	case BOT_CONSOLE_MATCH_WAIT:
		BotInterface_Printf(PRT_MESSAGE, "unknown match type\n");
		break;
	case BOT_CONSOLE_MATCH_WHAT_ARE_YOU_DOING:
		if (BotAI_ConsoleAddressedToBot(state, &match))
		{
			BotAI_ConsoleReportLongTermGoal(state);
		}
		break;
	case BOT_CONSOLE_MATCH_JOIN_SUBTEAM:
		if (BotAI_ConsoleTeamPlayIsOn() &&
			BotAI_ConsoleAddressedToBot(state, &match))
		{
			BotAI_ConsoleJoinSubteam(state, &match);
		}
		break;
	case BOT_CONSOLE_MATCH_LEAVE_SUBTEAM:
		if (BotAI_ConsoleTeamPlayIsOn() &&
			BotAI_ConsoleAddressedToBot(state, &match))
		{
			BotAI_ConsoleLeaveSubteam(state);
		}
		break;
	case BOT_CONSOLE_MATCH_CREATE_FORMATION:
	case BOT_CONSOLE_MATCH_FORMATION_POSITION:
		EA_SayTeam(state->client_number,
			"the part of my brain to create formations has been damaged");
		break;
	case BOT_CONSOLE_MATCH_FORMATION_SPACE:
		if (BotAI_ConsoleTeamPlayIsOn() &&
			BotAI_ConsoleAddressedToBot(state, &match))
		{
			BotAI_ConsoleSetFormationSpace(state, &match);
		}
		break;
	case BOT_CONSOLE_MATCH_DO_FORMATION:
		break;
	case BOT_CONSOLE_MATCH_DISMISS:
		if (BotAI_ConsoleTeamPlayIsOn() &&
			BotAI_ConsoleAddressedToBot(state, &match) &&
			(state->ltg_type == 1 || state->ltg_type == 2))
		{
			state->ltg_type = 0;
		}
		break;
	case BOT_CONSOLE_MATCH_CAMP:
		if (BotAI_ConsoleTeamPlayIsOn() &&
			BotAI_ConsoleAddressedToBot(state, &match))
		{
			BotAI_ConsoleHandleCamp(state, &match);
		}
		break;
	case BOT_CONSOLE_MATCH_CHECKPOINT:
		if (BotAI_ConsoleTeamPlayIsOn())
		{
			BotAI_ConsoleHandleCheckpoint(state, &match);
		}
		break;
	case BOT_CONSOLE_MATCH_PATROL:
		if (BotAI_ConsoleTeamPlayIsOn() &&
			BotAI_ConsoleAddressedToBot(state, &match) &&
			BotAI_ConsoleGetPatrolPoints(state, &match))
		{
			BotAI_ConsoleCommitPatrol(state, &match);
		}
		break;
	default:
		break;
	}

	return true;
}

/*
=============
BotAI_ChatTime

Returns Gladiator's pending-message length converted through the character's
chat characters-per-minute characteristic.
=============
*/
float BotAI_ChatTime(const bot_client_state_t *state)
{
	if (state == NULL || state->chat_state == NULL || state->character == NULL)
	{
		return 0.0f;
	}

	int cpm = Characteristic_BInteger(state->character,
		CHARACTERISTIC_CHAT_CPM,
		1,
		4000);
	return (float)BotChatLength(state->chat_state) * 30.0f / (float)cpm;
}

/*
=============
BotAI_ConstructRandomChat

Reconstructs sub_10022470's Seek-LTG random-chat gate.  It retains the
ordered team-goal exclusions, think-time trial, fast-chat bypass, valid
position predicate, and misc-versus-insult selection before leaving a
pending initial chat for the Stand node.
=============
*/
bool BotAI_ConstructRandomChat(bot_client_state_t *state,
	float thinktime)
{
	if (state == NULL || state->chat_state == NULL ||
		state->character == NULL || LibVarGetValue("nochat") != 0.0f)
	{
		return false;
	}
	if (state->ltg_type == 1 || state->ltg_type == 2 ||
		state->ltg_type == 5)
	{
		return false;
	}

	float random_chat = Characteristic_BFloat(state->character,
		CHARACTERISTIC_CHAT_RANDOM,
		0.0f,
		1.0f);
	if (BotAI_ConsoleRandom() > thinktime * 0.1f)
	{
		return false;
	}
	if (LibVarGetValue("fastchat") == 0.0f &&
		(BotAI_ConsoleRandom() > random_chat ||
			BotAI_ConsoleRandom() > 0.25f))
	{
		return false;
	}
	if (!BotAI_ValidChatPosition(state))
	{
		return false;
	}

	float miscellaneous = Characteristic_BFloat(state->character,
		CHARACTERISTIC_CHAT_MISC,
		0.0f,
		1.0f);
	const char *type = BotAI_ConsoleRandom() < miscellaneous
		? "random_misc"
		: "random_insult";
	BotInitialChat(state->chat_state, type, NULL);
	return true;
}

/*
=============
BotAI_ConstructLifecycleChat

Applies the Gladiator initial-chat gates shared by enter-game and level
transition events. The gate result is independent of template availability.
=============
*/
bool BotAI_ConstructLifecycleChat(bot_client_state_t *state,
	const char *type,
	int characteristic,
	bool require_valid_position)
{
	if (state == NULL || type == NULL || state->chat_state == NULL ||
		state->character == NULL || LibVarGetValue("nochat") != 0.0f)
	{
		return false;
	}

	float chance = Characteristic_BFloat(state->character,
		characteristic,
		0.0f,
		1.0f);
	if (LibVarGetValue("fastchat") == 0.0f &&
		BotAI_ConsoleRandom() > chance)
	{
		return false;
	}
	if (require_valid_position && !BotAI_ValidChatPosition(state))
	{
		return false;
	}

	char name[0x20];
	BotAI_ConsoleEasyClientName(state->client_number, name, sizeof(name));
	BotInitialChat(state->chat_state,
		type,
		name,
		NULL);
	return true;
}

/*
=============
BotAI_ConstructDeathChat

Reconstructs the compact retail death-chat selector.  A disabled nochat
libvar vetoes the event; otherwise fastchat bypasses the character death-chat
probability.  The chosen pending message is not emitted until Respawn reaches
past its typing deadline.
=============
*/
bool BotAI_ConstructDeathChat(bot_client_state_t *state)
{
	if (state == NULL || state->chat_state == NULL ||
		state->character == NULL || LibVarGetValue("nochat") != 0.0f)
	{
		return false;
	}

	float death_chat = Characteristic_BFloat(state->character,
		CHARACTERISTIC_CHAT_DEATH,
		0.0f,
		1.0f);
	if (LibVarGetValue("fastchat") == 0.0f &&
		BotAI_ConsoleRandom() > death_chat)
	{
		return false;
	}

	/*
	 * 0x10022160 builds variable 0 with EasyClientName (sub_10021860), not the
	 * raw netname, and leaves it empty when bs->enemy is 0.  Retail's caller
	 * buffer is the 32-byte var_20 slot.
	 */
	char killer_name[0x20];
	killer_name[0] = '\0';
	if (state->combat.current_enemy != 0 &&
		state->combat.current_enemy <= aasworld.maxClients)
	{
		BotAI_ConsoleEasyClientName(state->combat.current_enemy - 1,
			killer_name,
			sizeof(killer_name));
	}

	const char *chat_type = "death_bfg";
	if (state->bot_death_type != 12)
	{
		float insult = Characteristic_BFloat(state->character,
			CHARACTERISTIC_CHAT_INSULT,
			0.0f,
			1.0f);
		chat_type = BotAI_ConsoleRandom() < insult
			? "death_insult"
			: "death_praise";
	}

	BotInitialChat(state->chat_state,
		chat_type,
		killer_name,
		NULL);
	return true;
}

/*
=============
BotAI_ConstructKillChat

Reconstructs `sub_100222e0` for Battle Fight's dead-enemy exit.  It leaves a
pending kill chat for Stand only after the retail nochat, fastchat,
characteristic-19, and valid-position gates; the obituary subtype selects the
telefrag form before the normal insult/praise trial.
=============
*/
bool BotAI_ConstructKillChat(bot_client_state_t *state)
{
	if (state == NULL || state->chat_state == NULL ||
		state->character == NULL || LibVarGetValue("nochat") != 0.0f)
	{
		return false;
	}

	float kill_chat = Characteristic_BFloat(state->character,
		CHARACTERISTIC_CHAT_KILL,
		0.0f,
		1.0f);
	if (LibVarGetValue("fastchat") == 0.0f &&
		BotAI_ConsoleRandom() > kill_chat)
	{
		return false;
	}
	if (!BotAI_ValidChatPosition(state))
	{
		return false;
	}

	/* 0x100222e0 uses EasyClientName for variable 0, as the death chat does. */
	char victim_name[0x20];
	victim_name[0] = '\0';
	if (state->combat.current_enemy != 0 &&
		state->combat.current_enemy <= aasworld.maxClients)
	{
		BotAI_ConsoleEasyClientName(state->combat.current_enemy - 1,
			victim_name,
			sizeof(victim_name));
	}

	const char *chat_type = "kill_telefrag";
	if (state->enemy_death_type != 13)
	{
		float insult = Characteristic_BFloat(state->character,
			CHARACTERISTIC_CHAT_INSULT,
			0.0f,
			1.0f);
		chat_type = BotAI_ConsoleRandom() < insult
			? "kill_insult"
			: "kill_praise";
	}

	BotInitialChat(state->chat_state,
		chat_type,
		victim_name,
		NULL);
	return true;
}

/*
=============
BotAI_SelectConsoleReply

Applies Gladiator's nochat, stand-node, position, population, and character
probability gates before constructing a reply from the text after the colon.
=============
*/
static bool BotAI_SelectConsoleReply(bot_client_state_t *state,
	char *message)
{
	if (state == NULL || message == NULL || state->chat_state == NULL ||
		state->ai_node == BOT_AI_NODE_STAND ||
		LibVarGetValue("nochat") != 0.0f ||
		!BotAI_ValidChatPosition(state) || state->character == NULL)
	{
		return false;
	}

	float chat_reply = Characteristic_BFloat(state->character,
		CHARACTERISTIC_CHAT_REPLY,
		0.0f,
		1.0f);
	float population_gate = 1.5f / (float)(BotState_ActiveClientCount() + 1);
	if (BotAI_ConsoleRandom() >= population_gate ||
		BotAI_ConsoleRandom() >= chat_reply)
	{
		return false;
	}

	char *colon = strchr(message, ':');
	if (colon == NULL)
	{
		return false;
	}

	memmove(message, colon + 1, strlen(colon + 1) + 1U);
	UnifyWhiteSpaces(message);
	return BotReplyChat(state->chat_state, message) != 0;
}

/*
=============
BotCheckConsoleMessages

Reads the retail node queue in FIFO order, defers a recent chat head while the
queue is below the flood threshold, classifies normalised text, and removes
each non-deferred node at the same decision points as Gladiator.
=============
*/
void BotCheckConsoleMessages(bot_client_state_t *state)
{
	if (state == NULL || state->chat_state == NULL)
	{
		return;
	}

	bot_console_message_node_t *node = BotNextConsoleMessage(
		state->chat_state);
	while (node != NULL)
	{
		if (BotNumConsoleMessages(state->chat_state) < 10 &&
			node->type == CMS_CHAT)
		{
			float read_time = 1.0f + BotAI_ConsoleRandom();
			if (node->time > AAS_Time() - read_time)
			{
				break;
			}
		}

		if (node->type == CMS_CHAT)
		{
			const char *colon = strchr(node->message, ':');
			if (colon == NULL || BotAI_IsOwnConsoleChat(state, node->message, colon))
			{
				BotRemoveConsoleMessage(state->chat_state, node);
				node = BotNextConsoleMessage(state->chat_state);
				continue;
			}
		}

		char message[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
		memcpy(message, node->message, sizeof(message));
		message[sizeof(message) - 1U] = '\0';
		UnifyWhiteSpaces(message);
		unsigned long synonym_context = BotAI_ConsoleSynonymContext(state);
		BotReplaceSynonyms(message, synonym_context);

		bool matched = BotAI_MatchConsoleMessage(state, message);

		if (!matched &&
			node->type == CMS_CHAT &&
			BotAI_SelectConsoleReply(state, message))
		{
			BotRemoveConsoleMessage(state->chat_state, node);
			state->stand_time = AAS_Time() + BotAI_ChatTime(state);
			state->chat_standing = true;
			BotAI_EnterNode(state, BOT_AI_NODE_STAND);
			return;
		}

		BotRemoveConsoleMessage(state->chat_state, node);
		node = BotNextConsoleMessage(state->chat_state);
	}
}
