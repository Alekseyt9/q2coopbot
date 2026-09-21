#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "q2bridge/botlib.h"
#include "botlib_interface.h"
#include "botlib/ai_chat/ai_chat.h"
#include "botlib/ai_character/bot_character.h"
#include "botlib/ai_goal/bot_goal.h"
#include "botlib/ai_move/bot_move.h"
#include "botlib/ai_move/mover_catalogue.h"
#include "botlib/ai_weight/bot_weight.h"
#include "bot_interface.h"
#include "bot_interface_exports.h"

/*
=============
BotInterface_Test

Preserves the retail test export as a pure no-op.
=============
*/
int BotInterface_Test(int parm0, char *parm1, vec3_t parm2, vec3_t parm3)
{
	(void)parm0;
	(void)parm1;
	(void)parm2;
	(void)parm3;

	return BLERR_NOERROR;
}

/*
=============
BotInterface_BotWeightIndex

Guards the extended goal-weight lookup behind the botlib setup contract.
=============
*/
int BotInterface_BotWeightIndex(int handle, const char *classname)
{
    if (!BotInterface_EnsureLibraryReady("BotWeightIndex"))
    {
        return -1;
    }

    return BotWeightIndex(handle, classname);
}

/*
=============
BotInterface_BotItemGoalInVisButNotVisible

Checks item goals against AAS visibility and trace results.
=============
*/
int BotInterface_BotItemGoalInVisButNotVisible(int viewer,
	vec3_t eye,
	vec3_t viewangles,
	bot_goal_t *goal)
{
	if (!BotInterface_EnsureLibraryReady("BotItemGoalInVisButNotVisible"))
	{
		return 0;
	}

	return BotItemGoalInVisButNotVisible(viewer, eye, viewangles, goal);
}

/*
=============
BotInterface_BotTouchingGoal

Guards the extended retail goal-bounds query behind library setup.
=============
*/
int BotInterface_BotTouchingGoal(const vec3_t origin, const bot_goal_t *goal)
{
    if (!BotInterface_EnsureLibraryReady("BotTouchingGoal"))
    {
        return 0;
    }

    return BotTouchingGoal(origin, goal);
}

int BotInterface_BotAllocWeightConfig(void)
{
    if (!BotLibraryEnsureSetup("BotAllocWeightConfig"))
    {
        return 0;
    }

    return BotAllocWeightConfig();
}

void BotInterface_BotFreeWeightConfig(int handle)
{
    if (!BotInterface_EnsureLibraryReady("BotFreeWeightConfig"))
    {
        return;
    }

    BotFreeWeightConfig(handle);
}

void BotInterface_BotFreeWeightConfig2(bot_weight_config_t *config)
{
    if (!BotInterface_EnsureLibraryReady("BotFreeWeightConfig2"))
    {
        return;
    }

    BotFreeWeightConfig2(config);
}

int BotInterface_BotLoadWeights(int handle, const char *filename)
{
    if (!BotInterface_EnsureLibraryReady("BotLoadWeights"))
    {
        return 0;
    }

    return BotLoadWeights(handle, filename);
}

int BotInterface_BotWriteWeights(int handle, const char *filename)
{
    if (!BotInterface_EnsureLibraryReady("BotWriteWeights"))
    {
        return 0;
    }

    return BotWriteWeights(handle, filename);
}

int BotInterface_BotSetWeight(int handle, const char *name, float value)
{
    if (!BotInterface_EnsureLibraryReady("BotSetWeight"))
    {
        return 0;
    }

    return BotSetWeight(handle, name, value);
}

/*
=============
BotInterface_BotFindFuzzyWeight

Bridge fuzzy weight name lookup through the weight handle table.
=============
*/
int BotInterface_BotFindFuzzyWeight(int handle, const char *name)
{
	if (!BotInterface_EnsureLibraryReady("BotFindFuzzyWeight"))
	{
		return -1;
	}

	return BotFindFuzzyWeight(handle, name);
}

/*
=============
BotInterface_BotFuzzyWeightHandle

Bridge fuzzy weight evaluation through the weight handle table.
=============
*/
float BotInterface_BotFuzzyWeightHandle(int handle,
											   const int *inventory,
											   int weight_index)
{
	if (!BotInterface_EnsureLibraryReady("BotFuzzyWeightHandle"))
	{
		return 0.0f;
	}

	return BotFuzzyWeightHandle(handle, inventory, weight_index);
}

bot_weight_config_t *BotInterface_BotReadWeightsFile(const char *filename)
{
    if (!BotInterface_EnsureLibraryReady("BotReadWeightsFile"))
    {
        return NULL;
    }

    return BotReadWeightsFile(filename);
}

/*
=============
BotInterface_BotAllocMoveState

Guards allocation of an exported movement-state handle.
=============
*/
int BotInterface_BotAllocMoveState(void)
{
	if (!BotInterface_EnsureLibraryReady("BotAllocMoveState"))
	{
		return 0;
	}

	return BotAllocMoveStateHandle();
}

/*
=============
BotInterface_BotFreeMoveState

Guards release of an exported movement-state handle.
=============
*/
void BotInterface_BotFreeMoveState(int handle)
{
	if (!BotInterface_EnsureLibraryReady("BotFreeMoveState"))
	{
		return;
	}

	BotFreeMoveStateHandle(handle);
}

/*
=============
BotInterface_BotResetMoveState

Guards reset of an exported movement-state handle.
=============
*/
void BotInterface_BotResetMoveState(int handle)
{
	if (!BotInterface_EnsureLibraryReady("BotResetMoveState"))
	{
		return;
	}

	BotResetMoveStateHandle(handle);
}

/*
=============
BotInterface_BotInitMoveState

Guards initialization of an exported movement-state handle.
=============
*/
void BotInterface_BotInitMoveState(int handle, const bot_initmove_t *initmove)
{
	if (!BotInterface_EnsureLibraryReady("BotInitMoveState"))
	{
		return;
	}

	BotInitMoveStateHandle(handle, initmove);
}

/*
=============
BotInterface_BotMoveToGoal

Guards the exported move-to-goal operation.
=============
*/
void BotInterface_BotMoveToGoal(bot_moveresult_t *result,
	int movestate,
	const bot_goal_t *goal,
	int travelflags)
{
	if (!BotInterface_EnsureLibraryReady("BotMoveToGoal"))
	{
		if (result != NULL)
		{
			memset(result, 0, sizeof(*result));
		}
		return;
	}

	BotMoveToGoalHandle(result, movestate, goal, travelflags);
}

/*
=============
BotInterface_BotMoveInDirection

Guards the exported directional movement operation.
=============
*/
int BotInterface_BotMoveInDirection(int movestate, const vec3_t dir, float speed, int type)
{
	if (!BotInterface_EnsureLibraryReady("BotMoveInDirection"))
	{
		return 0;
	}

	return BotMoveInDirectionHandle(movestate, dir, speed, type);
}

/*
=============
BotInterface_BotResetAvoidReach

Guards reset of movement reachability avoidance.
=============
*/
void BotInterface_BotResetAvoidReach(int movestate)
{
	if (!BotInterface_EnsureLibraryReady("BotResetAvoidReach"))
	{
		return;
	}

	BotResetAvoidReachHandle(movestate);
}

/*
=============
BotInterface_BotResetLastAvoidReach

Bridge reset helper for the most recent avoided reachability.
=============
*/
void BotInterface_BotResetLastAvoidReach(int movestate)
{
	if (!BotInterface_EnsureLibraryReady("BotResetLastAvoidReach"))
	{
		return;
	}

	BotResetLastAvoidReachHandle(movestate);
}

/*
=============
BotInterface_BotReachabilityArea

Bridge reachability area lookup through the botlib API.
=============
*/
int BotInterface_BotReachabilityArea(vec3_t origin, int client)
{
	if (!BotInterface_EnsureLibraryReady("BotReachabilityArea"))
	{
		return 0;
	}

	return BotReachabilityArea(origin, client);
}

/*
=============
BotInterface_BotMovementViewTarget

Bridge movement lookahead targeting through the botlib API.
=============
*/
int BotInterface_BotMovementViewTarget(int movestate,
											  const bot_goal_t *goal,
											  int travelflags,
											  float lookahead,
											  vec3_t target)
{
	if (!BotInterface_EnsureLibraryReady("BotMovementViewTarget"))
	{
		return 0;
	}

	return BotMovementViewTargetHandle(movestate, goal, travelflags, lookahead, target);
}

/*
=============
BotInterface_BotPredictVisiblePosition

Bridge visibility prediction through the botlib API.
=============
*/
int BotInterface_BotPredictVisiblePosition(vec3_t origin,
												  int areanum,
												  const bot_goal_t *goal,
												  int travelflags,
												  vec3_t target)
{
	if (!BotInterface_EnsureLibraryReady("BotPredictVisiblePosition"))
	{
		return 0;
	}

	return BotPredictVisiblePosition(origin, areanum, goal, travelflags, target);
}

/*
=============
BotInterface_BotAddAvoidSpot

Bridge avoid-spot updates through the botlib movement API.
=============
*/
void BotInterface_BotAddAvoidSpot(int movestate, vec3_t origin, float radius, int type)
{
	if (!BotInterface_EnsureLibraryReady("BotAddAvoidSpot"))
	{
		return;
	}

	BotAddAvoidSpot(movestate, origin, radius, type);
}

int BotInterface_BotAllocWeaponState(void)
{
    if (!BotInterface_EnsureLibraryReady("BotAllocWeaponState"))
    {
        return 0;
    }

    return BotAllocWeaponState();
}

void BotInterface_BotFreeWeaponState(int handle)
{
    if (!BotInterface_EnsureLibraryReady("BotFreeWeaponState"))
    {
        return;
    }

    BotFreeWeaponState(handle);
}

void BotInterface_BotResetWeaponState(int handle)
{
    if (!BotInterface_EnsureLibraryReady("BotResetWeaponState"))
    {
        return;
    }

    BotResetWeaponState(handle);
}

int BotInterface_BotLoadWeaponWeights(int weaponstate, const char *filename)
{
    if (!BotInterface_EnsureLibraryReady("BotLoadWeaponWeights"))
    {
        return BLERR_LIBRARYNOTSETUP;
    }

    return BotLoadWeaponWeights(weaponstate, filename);
}

void BotInterface_BotFreeWeaponWeights(int weaponstate)
{
    if (!BotInterface_EnsureLibraryReady("BotFreeWeaponWeights"))
    {
        return;
    }

    BotFreeWeaponWeights(weaponstate);
}

int BotInterface_BotChooseBestFightWeapon(int weaponstate, const int *inventory)
{
    if (!BotInterface_EnsureLibraryReady("BotChooseBestFightWeapon"))
    {
        return 0;
    }

    return BotChooseBestFightWeapon(weaponstate, inventory);
}

int BotInterface_BotGetTopRankedWeapon(int weaponstate)
{
    if (!BotInterface_EnsureLibraryReady("BotGetTopRankedWeapon"))
    {
        return 0;
    }

    return BotGetTopRankedWeapon(weaponstate);
}

void BotInterface_BotGetWeaponInfo(int weaponstate, int weapon, bot_weapon_info_t *weaponinfo)
{
    if (!BotInterface_EnsureLibraryReady("BotGetWeaponInfo"))
    {
        if (weaponinfo != NULL)
        {
            memset(weaponinfo, 0, sizeof(*weaponinfo));
        }
        return;
    }

    BotGetWeaponInfo(weaponstate, weapon, weaponinfo);
}

/*
=============
BotInterface_BotLoadCharacter

Guards and forwards Q3-style character loading.
=============
*/
int BotInterface_BotLoadCharacter(const char *character_file, float skill)
{
	if (!BotInterface_EnsureLibraryReady("BotLoadCharacter"))
	{
		return 0;
	}

	return BotLoadCharacterHandle(character_file, skill);
}

/*
=============
BotInterface_BotFreeCharacter

Guards and forwards character handle release.
=============
*/
void BotInterface_BotFreeCharacter(int handle)
{
	if (!BotInterface_EnsureLibraryReady("BotFreeCharacter"))
	{
		return;
	}

	BotFreeCharacterHandle(handle);
}

/*
=============
BotInterface_BotLoadCharacterSkill

Guards and forwards exact skill-specific character loading.
=============
*/
int BotInterface_BotLoadCharacterSkill(const char *character_file, float skill)
{
	if (!BotInterface_EnsureLibraryReady("BotLoadCharacterSkill"))
	{
		return 0;
	}

	return BotLoadCharacterSkillHandle(character_file, skill);
}

/*
=============
BotInterface_BotFreeCharacterStrings

Guards and forwards transient character profile cleanup.
=============
*/
void BotInterface_BotFreeCharacterStrings(ai_character_profile_t *profile)
{
	if (!BotInterface_EnsureLibraryReady("BotFreeCharacterStrings"))
	{
		return;
	}

	BotFreeCharacterStringsHandle(profile);
}

/*
=============
BotInterface_Characteristic_Float

Guards and forwards float characteristic queries.
=============
*/
float BotInterface_Characteristic_Float(int handle, int index)
{
	if (!BotInterface_EnsureLibraryReady("Characteristic_Float"))
	{
		return 0.0f;
	}

	return Characteristic_FloatHandle(handle, index);
}

/*
=============
BotInterface_Characteristic_BFloat

Guards and forwards bounded float characteristic queries.
=============
*/
float BotInterface_Characteristic_BFloat(int handle,
	int index,
	float minimum,
	float maximum)
{
	if (!BotInterface_EnsureLibraryReady("Characteristic_BFloat"))
	{
		return 0.0f;
	}

	return Characteristic_BFloatHandle(handle, index, minimum, maximum);
}

/*
=============
BotInterface_Characteristic_Integer

Guards and forwards integer characteristic queries.
=============
*/
int BotInterface_Characteristic_Integer(int handle, int index)
{
	if (!BotInterface_EnsureLibraryReady("Characteristic_Integer"))
	{
		return 0;
	}

	return Characteristic_IntegerHandle(handle, index);
}

/*
=============
BotInterface_Characteristic_BInteger

Guards and forwards bounded integer characteristic queries.
=============
*/
int BotInterface_Characteristic_BInteger(int handle,
	int index,
	int minimum,
	int maximum)
{
	if (!BotInterface_EnsureLibraryReady("Characteristic_BInteger"))
	{
		return 0;
	}

	return Characteristic_BIntegerHandle(handle, index, minimum, maximum);
}

/*
=============
BotInterface_Characteristic_String

Guards and forwards string characteristic queries.
=============
*/
void BotInterface_Characteristic_String(int handle,
	int index,
	char *buffer,
	int buffer_size)
{
	if (!BotInterface_EnsureLibraryReady("Characteristic_String"))
	{
		if (buffer != NULL && buffer_size > 0)
		{
			buffer[0] = '\0';
		}
		return;
	}

	Characteristic_StringHandle(handle, index, buffer, buffer_size);
}

bot_chatstate_t *BotInterface_BotAllocChatState(void)
{
    if (!BotInterface_EnsureLibraryReady("BotAllocChatState"))
    {
        return NULL;
    }

    return BotAllocChatState();
}

/*
=============
BotInterface_BotFreeChatState

Guards release of an exported chat state.
=============
*/
void BotInterface_BotFreeChatState(bot_chatstate_t *state)
{
	if (!BotInterface_EnsureLibraryReady("BotFreeChatState"))
	{
		return;
	}

	BotDestroyChatState(state);
}

/*
=============
BotInterface_BotLoadChatFile

Guards loading a chat file into an exported chat state.
=============
*/
int BotInterface_BotLoadChatFile(bot_chatstate_t *state, const char *chatfile, const char *chatname)
{
	if (!BotInterface_EnsureLibraryReady("BotLoadChatFile"))
	{
		return BLERR_LIBRARYNOTSETUP;
	}

	return BotLoadChatFile(state, chatfile, chatname);
}

void BotInterface_BotFreeChatFile(bot_chatstate_t *state)
{
    if (!BotInterface_EnsureLibraryReady("BotFreeChatFile"))
    {
        return;
    }

    BotFreeChatFile(state);
}

void BotInterface_BotQueueConsoleMessage(bot_chatstate_t *state, int type, const char *message)
{
    if (!BotInterface_EnsureLibraryReady("BotQueueConsoleMessage"))
    {
        return;
    }

    BotQueueConsoleMessage(state, type, message);
}

/*
=============
BotInterface_BotRemoveConsoleMessage

Guards removal of the first queued console message of a given type.
=============
*/
int BotInterface_BotRemoveConsoleMessage(bot_chatstate_t *state, int type)
{
	if (!BotInterface_EnsureLibraryReady("BotRemoveConsoleMessage"))
	{
		return 0;
	}

	return BotRemoveConsoleMessageType(state, type);
}

/*
=============
BotInterface_BotNextConsoleMessage

Guards copying and consuming the next queued console message.
=============
*/
int BotInterface_BotNextConsoleMessage(bot_chatstate_t *state,
	int *type,
	char *buffer,
	size_t buffer_size)
{
	if (!BotInterface_EnsureLibraryReady("BotNextConsoleMessage"))
	{
		if (type != NULL)
		{
			*type = 0;
		}
		if (buffer != NULL && buffer_size > 0)
		{
			buffer[0] = '\0';
		}
		return 0;
	}

	return BotNextConsoleMessageCopy(state, type, buffer, buffer_size);
}

size_t BotInterface_BotNumConsoleMessages(const bot_chatstate_t *state)
{
    if (!BotInterface_EnsureLibraryReady("BotNumConsoleMessages"))
    {
        return 0U;
    }

    return BotNumConsoleMessages(state);
}

/*
=============
BotInterface_BotNumInitialChats

Guards the Q3-shaped initial-chat count adapter, including successor aliases.
=============
*/
int BotInterface_BotNumInitialChats(const bot_chatstate_t *state, const char *type)
{
	if (!BotInterface_EnsureLibraryReady("BotNumInitialChats"))
	{
		return 0;
	}

	return BotNumInitialChatsWithAliases(state, type);
}

void BotInterface_BotEnterChat(bot_chatstate_t *state, int client, int sendto)
{
    if (!BotInterface_EnsureLibraryReady("BotEnterChat"))
    {
        return;
    }

    BotEnterChat(state, client, sendto);
}

/*
=============
BotInterface_BotReplyChat

Guards the compatibility export and forwards its folded context to the named
host adapter, leaving retail BotReplyChat's two-argument contract untouched.
=============
*/
int BotInterface_BotReplyChat(bot_chatstate_t *state,
	const char *message,
	unsigned long int context)
{
	if (!BotInterface_EnsureLibraryReady("BotReplyChat"))
	{
		return 0;
	}

	return BotReplyChatWithContext(state, message, context);
}

/*
=============
BotInterface_BotReplyChatWithContexts

Guards and forwards the Q3-shaped split-context reply export.
=============
*/
int BotInterface_BotReplyChatWithContexts(bot_chatstate_t *state,
	const char *message,
	unsigned long int mcontext,
	unsigned long int vcontext,
	const char *var0,
	const char *var1,
	const char *var2,
	const char *var3,
	const char *var4,
	const char *var5,
	const char *var6,
	const char *var7)
{
	if (!BotInterface_EnsureLibraryReady("BotReplyChatWithContexts"))
	{
		return 0;
	}

	return BotReplyChatWithContexts(state,
		message,
		mcontext,
		vcontext,
		var0,
		var1,
		var2,
		var3,
		var4,
		var5,
		var6,
		var7);
}

/*
=============
BotInterface_BotInitialChat

Guards the compatibility export and forwards at most ten variables to the
explicit-context host adapter.
=============
*/
int BotInterface_BotInitialChat(bot_chatstate_t *state,
	const char *type,
	unsigned long context,
	...)
{
	const char *variables[10] = {0};
	if (!BotInterface_EnsureLibraryReady("BotInitialChat"))
	{
		return 0;
	}

	va_list args;
	va_start(args, context);
	for (size_t i = 0; i < sizeof(variables) / sizeof(variables[0]); ++i)
	{
		const char *value = va_arg(args, const char *);
		if (value == NULL)
		{
			break;
		}
		variables[i] = value;
	}
	va_end(args);

	return BotInitialChatWithContext(state,
		type,
		context,
		variables[0],
		variables[1],
		variables[2],
		variables[3],
		variables[4],
		variables[5],
		variables[6],
		variables[7],
		variables[8],
		variables[9],
		NULL);
}

/*
=============
BotInterface_BotChatLength

Returns the constructed chat message length for a chat state.
=============
*/
int BotInterface_BotChatLength(const bot_chatstate_t *state)
{
	if (!BotInterface_EnsureLibraryReady("BotChatLength"))
	{
		return 0;
	}

	return BotChatLength(state);
}

void BotInterface_BotGetChatMessage(bot_chatstate_t *state, char *buffer, int buffer_size)
{
	if (!BotInterface_EnsureLibraryReady("BotGetChatMessage"))
	{
		if (buffer != NULL && buffer_size > 0)
		{
			buffer[0] = '\0';
		}
		return;
	}

	BotGetChatMessage(state, buffer, buffer_size);
}

void BotInterface_BotSetChatGender(bot_chatstate_t *state, int gender)
{
	if (!BotInterface_EnsureLibraryReady("BotSetChatGender"))
	{
		return;
	}

	BotSetChatGender(state, gender);
}

/*
=============
BotInterface_BotSetChatName

Guards assignment of an exported chat state's client name.
=============
*/
void BotInterface_BotSetChatName(bot_chatstate_t *state, const char *name, int client)
{
	if (!BotInterface_EnsureLibraryReady("BotSetChatName"))
	{
		return;
	}

	BotSetChatNameWithClient(state, name, client);
}

/*
=============
BotInterface_StringContains

Guards and forwards the compatibility substring-index helper.
=============
*/
int BotInterface_StringContains(const char *str1,
	const char *str2,
	int casesensitive)
{
	if (!BotInterface_EnsureLibraryReady("StringContains"))
	{
		return -1;
	}

	return StringContainsIndex(str1, str2, casesensitive);
}

/*
=============
BotInterface_BotFindMatch

Guards and forwards the reconstructed setup match-template export.
=============
*/
int BotInterface_BotFindMatch(const char *str,
	bot_match_t *match,
	unsigned long int context)
{
	if (!BotInterface_EnsureLibraryReady("BotFindMatch"))
	{
		if (match != NULL)
		{
			memset(match, 0, sizeof(*match));
		}
		return 0;
	}

	return BotFindMatch(str, match, context);
}

/*
=============
BotInterface_BotMatchVariable

Guards and forwards bounds-checked captured match-variable extraction.
=============
*/
void BotInterface_BotMatchVariable(const bot_match_t *match,
	int variable,
	char *buffer,
	int buffer_size)
{
	if (!BotInterface_EnsureLibraryReady("BotMatchVariable"))
	{
		if (buffer != NULL && buffer_size > 0)
		{
			buffer[0] = '\0';
		}
		return;
	}

	BotMatchVariableSized(match, variable, buffer, buffer_size);
}

/*
=============
BotInterface_UnifyWhiteSpaces

Guards and forwards retail chat whitespace canonicalization.
=============
*/
void BotInterface_UnifyWhiteSpaces(char *string)
{
	if (!BotInterface_EnsureLibraryReady("UnifyWhiteSpaces"))
	{
		return;
	}

	UnifyWhiteSpaces(string);
}

/*
=============
BotInterface_BotReplaceSynonyms

Guards and forwards setup-cache synonym replacement.
=============
*/
void BotInterface_BotReplaceSynonyms(char *string, unsigned long int context)
{
	if (!BotInterface_EnsureLibraryReady("BotReplaceSynonyms"))
	{
		return;
	}

	BotReplaceSynonyms(string, context);
}
void BotInterface_FillExportedWrappers(bot_export_extended_t *export_table)
{
    if (export_table == NULL)
    {
        return;
    }

    export_table->Test = BotInterface_Test;
    export_table->BotWeightIndex = BotInterface_BotWeightIndex;
    export_table->BotItemGoalInVisButNotVisible = BotInterface_BotItemGoalInVisButNotVisible;
    export_table->BotTouchingGoal = BotInterface_BotTouchingGoal;
    export_table->BotAllocWeightConfig = BotInterface_BotAllocWeightConfig;
    export_table->BotFreeWeightConfig = BotInterface_BotFreeWeightConfig;
    export_table->BotFreeWeightConfig2 = BotInterface_BotFreeWeightConfig2;
    export_table->BotLoadWeights = BotInterface_BotLoadWeights;
    export_table->BotWriteWeights = BotInterface_BotWriteWeights;
    export_table->BotSetWeight = BotInterface_BotSetWeight;
    export_table->BotFindFuzzyWeight = BotInterface_BotFindFuzzyWeight;
    export_table->BotFuzzyWeightHandle = BotInterface_BotFuzzyWeightHandle;
    export_table->BotReadWeightsFile = BotInterface_BotReadWeightsFile;
    export_table->BotAllocMoveState = BotInterface_BotAllocMoveState;
    export_table->BotFreeMoveState = BotInterface_BotFreeMoveState;
    export_table->BotResetMoveState = BotInterface_BotResetMoveState;
    export_table->BotInitMoveState = BotInterface_BotInitMoveState;
    export_table->BotMoveToGoal = BotInterface_BotMoveToGoal;
    export_table->BotMoveInDirection = BotInterface_BotMoveInDirection;
    export_table->BotResetAvoidReach = BotInterface_BotResetAvoidReach;
    export_table->BotResetLastAvoidReach = BotInterface_BotResetLastAvoidReach;
    export_table->BotReachabilityArea = BotInterface_BotReachabilityArea;
    export_table->BotMovementViewTarget = BotInterface_BotMovementViewTarget;
    export_table->BotPredictVisiblePosition = BotInterface_BotPredictVisiblePosition;
    export_table->BotAddAvoidSpot = BotInterface_BotAddAvoidSpot;
    export_table->BotLoadCharacter = BotInterface_BotLoadCharacter;
    export_table->BotFreeCharacter = BotInterface_BotFreeCharacter;
    export_table->BotLoadCharacterSkill = BotInterface_BotLoadCharacterSkill;
    export_table->BotFreeCharacterStrings = BotInterface_BotFreeCharacterStrings;
    export_table->Characteristic_Float = BotInterface_Characteristic_Float;
    export_table->Characteristic_BFloat = BotInterface_Characteristic_BFloat;
    export_table->Characteristic_Integer = BotInterface_Characteristic_Integer;
    export_table->Characteristic_BInteger = BotInterface_Characteristic_BInteger;
    export_table->Characteristic_String = BotInterface_Characteristic_String;
    export_table->BotAllocWeaponState = BotInterface_BotAllocWeaponState;
    export_table->BotFreeWeaponState = BotInterface_BotFreeWeaponState;
    export_table->BotResetWeaponState = BotInterface_BotResetWeaponState;
    export_table->BotLoadWeaponWeights = BotInterface_BotLoadWeaponWeights;
    export_table->BotFreeWeaponWeights = BotInterface_BotFreeWeaponWeights;
    export_table->BotChooseBestFightWeapon = BotInterface_BotChooseBestFightWeapon;
    export_table->BotGetTopRankedWeapon = BotInterface_BotGetTopRankedWeapon;
    export_table->BotGetWeaponInfo = BotInterface_BotGetWeaponInfo;
    export_table->BotAllocChatState = BotInterface_BotAllocChatState;
    export_table->BotFreeChatState = BotInterface_BotFreeChatState;
    export_table->BotLoadChatFile = BotInterface_BotLoadChatFile;
    export_table->BotFreeChatFile = BotInterface_BotFreeChatFile;
    export_table->BotQueueConsoleMessage = BotInterface_BotQueueConsoleMessage;
    export_table->BotRemoveConsoleMessage = BotInterface_BotRemoveConsoleMessage;
    export_table->BotNextConsoleMessage = BotInterface_BotNextConsoleMessage;
    export_table->BotNumConsoleMessages = BotInterface_BotNumConsoleMessages;
    export_table->BotEnterChat = BotInterface_BotEnterChat;
    export_table->BotReplyChat = BotInterface_BotReplyChat;
    export_table->BotReplyChatWithContexts = BotInterface_BotReplyChatWithContexts;
    export_table->BotChatLength = BotInterface_BotChatLength;
    export_table->BotNumInitialChats = BotInterface_BotNumInitialChats;
    export_table->BotInitialChat = BotInterface_BotInitialChat;
    export_table->BotGetChatMessage = BotInterface_BotGetChatMessage;
    export_table->BotSetChatGender = BotInterface_BotSetChatGender;
    export_table->BotSetChatName = BotInterface_BotSetChatName;
    export_table->StringContains = BotInterface_StringContains;
    export_table->BotFindMatch = BotInterface_BotFindMatch;
    export_table->BotMatchVariable = BotInterface_BotMatchVariable;
    export_table->UnifyWhiteSpaces = BotInterface_UnifyWhiteSpaces;
    export_table->BotReplaceSynonyms = BotInterface_BotReplaceSynonyms;
}
