#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <stdarg.h>
#include <string.h>
#include <float.h>
#include <math.h>

#include "shared/gladiator_version.h"
#include "shared/q_platform.h"
#include "q2bridge/aas_translation.h"
#include "q2bridge/botlib.h"
#include "q2bridge/bridge.h"
#include "q2bridge/bridge_config.h"
#include "q2bridge/update_translator.h"
#include "botlib/common/l_crc.h"
#include "botlib/common/l_libvar.h"
#include "botlib/common/l_log.h"
#include "botlib/common/l_memory.h"
#include "botlib/common/l_utils.h"
#include "botlib/aas/aas_map.h"
#include "botlib/aas/aas_local.h"
#include "botlib/aas/aas_sound.h"
#include "botlib/ai_chat/ai_chat.h"
#include "botlib/ai_character/bot_character.h"
#include "botlib/ai/ai_dm.h"
#include "botlib/ai_weight/bot_weight.h"
#include "botlib/ai_goal/ai_goal.h"
#include "botlib/ai/goal_move_orchestrator.h"
#include "botlib/ai_move/mover_catalogue.h"
#include "bot_interface_assets.h"
#include "bot_interface_area.h"
#include "bot_interface_elevator.h"
#include "bot_interface_map.h"
#include "bot_interface_objective.h"
#include "bot_interface_combat.h"
#include "bot_interface_entity.h"
#include "bot_interface_behavior.h"
#include "bot_interface_import.h"
#include "bot_interface_exports.h"
#include "bot_interface_console.h"
#include "bot_interface_lifecycle.h"
#include "bot_interface_runtime.h"
#include "bot_interface_battle.h"
#include "bot_interface_battle_state.h"
#include "bot_interface_goal.h"
#include "bot_interface_coop_state.h"
#include "bot_interface_coop_role.h"
#include "bot_interface_coop_overlay.h"
#include "botlib/ea/ea_local.h"
#include "botlib/precomp/l_precomp.h"
#include "botlib_interface.h"
#include "bot_interface.h"
#include "bot_state.h"

void BotInterface_Printf(int priority, const char *fmt, ...);
void BotAI_EnterNode(bot_client_state_t *state, int node);
int BotAI_CoopPlayerEntity(const bot_client_state_t *state,
	vec3_t origin);
bool BotAI_CoopObjectiveControlEnabled(void);
void BotAI_CoopSetControlObjectivePhase(
	bot_client_state_t *state,
	bot_coop_control_phase_t phase,
	int control_entity,
	int goal_area);
bool BotAI_ApplyCoopControlWait(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input);
bool BotAI_ApplyCoopAreaAdvanceGate(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input);


static bot_chatstate_t *g_botInterfaceConsoleChat = NULL;
static bot_export_t *g_botRetailExportTable = NULL;
static bot_export_extended_t *g_botExtendedExportTable = NULL;

static aas_bspentity_t *g_botInterfaceCoopMapEntities = NULL;
static float g_botInterfaceFrameTime = 0.0f;
static unsigned int g_botInterfaceFrameNumber = 0;
static bool g_botInterfaceDebugDrawEnabled = false;
static bool g_botInterfaceStartFrameLogged = false;

float BotInterface_CurrentFrameTime(void)
{
	return g_botInterfaceFrameTime;
}

#define CHARACTERISTIC_CROUCHER 24



static void BotInterface_ResetMapCache(void)
{
	if (g_botInterfaceCoopMapEntities != NULL)
	{
		AAS_FreeBSPEntities(g_botInterfaceCoopMapEntities);
		g_botInterfaceCoopMapEntities = NULL;
	}
	BotMove_MoverCatalogueReset();
	BotInterface_ResetMapAssets();
	BotInterface_ResetMapModel();
	g_botInterfaceStartFrameLogged = false;
}

/*
=============
BotInterface_CoopMapEntities

Keep one parsed copy of the map's BSP entity/control graph.  The objective
overlay and the diagnostic dump must use the same immutable source so a
blocked inline model cannot be resolved against a different entity snapshot
than the one reported for the map.
=============
*/
aas_bspentity_t *BotInterface_CoopMapEntities(void)
{
	if (g_botInterfaceCoopMapEntities == NULL && AAS_Initialized())
	{
		g_botInterfaceCoopMapEntities = AAS_LoadBSPEntities();
	}
	return g_botInterfaceCoopMapEntities;
}



static void BotInterface_BeginFrame(float time)
{
    g_botInterfaceFrameTime = time;
	BotLib_LogSetTime(time);
	BotGoal_SetCurrentTime(time);
    g_botInterfaceFrameNumber += 1U;
    Bridge_SetFrameTime(time);
    AAS_SoundSubsystem_SetFrameTime(time);
    BotInterface_ResetFrameQueues();
}




static void BotInterface_PrintBanner(int priority, const char *message)
{
    if (message == NULL)
    {
        return;
    }

    if (BotInterface_GetEngineImport() != NULL &&
        BotInterface_GetEngineImport()->Print != NULL)
    {
        BotInterface_GetEngineImport()->Print(priority, "%s", message);
    }
    else
    {
        BotLib_Print(priority, "%s", message);
    }
}

void BotInterface_Printf(int priority, const char *fmt, ...)
{
    if (BotInterface_GetEngineImport() == NULL ||
        BotInterface_GetEngineImport()->Print == NULL || fmt == NULL)
    {
        return;
    }

    va_list args;
    va_start(args, fmt);

    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), fmt, args);

    va_end(args);

    BotInterface_GetEngineImport()->Print(priority, "%s", buffer);
}

/*
=============
BotInterface_EnsureLibraryReady

Applies the retail shared bot-library setup guard and diagnostic.
=============
*/
bool BotInterface_EnsureLibraryReady(const char *function_name)
{
	if (BotInterface_GetEngineImport() == NULL)
	{
		return false;
	}

	if (!BotLibraryInitialized())
	{
		if (function_name != NULL)
		{
			BotInterface_Printf(PRT_ERROR,
				"%s: bot library used before being setup\n",
				function_name);
		}
		return false;
	}

	return true;
}

/*
=============
BotInterface_ValidateClientNumber

Applies the retail inclusive maxclients guard and exact diagnostic contract.
=============
*/
static qboolean BotInterface_ValidateClientNumber(int client, const char *function_name)
{
	int max_client_number = BotState_ClientCapacity();
	if (client >= 0 && client <= max_client_number)
	{
		return qtrue;
	}

	BotInterface_Printf(PRT_ERROR,
		"%s: invalid client number %d, [0, %d]\n",
		function_name,
		client,
		max_client_number);
	return qfalse;
}

static bot_chatstate_t *BotInterface_EnsureConsoleChatState(void)
{
    if (g_botInterfaceConsoleChat == NULL) {
        g_botInterfaceConsoleChat = BotAllocChatState();
        if (g_botInterfaceConsoleChat == NULL) {
            if (BotInterface_GetEngineImport() != NULL &&
                BotInterface_GetEngineImport()->Print != NULL) {
                BotInterface_GetEngineImport()->Print(PRT_ERROR,
                    "[bot_interface] failed to allocate console chat state\n");
            }
        }
    }

    return g_botInterfaceConsoleChat;
}

static void BotInterface_Log(int priority, const char *functionName)
{
    if (functionName == NULL)
    {
        return;
    }

    BotLib_Print(priority, "[bot_interface] %s\n", functionName);
}

static char *BotVersion(void)
{
    static char version[] = "BotLib v0.96";

    return version;
}

/*
 * Version marker for the reconstruction itself.
 *
 * BotVersion() above reports the legacy botlib version and must keep returning
 * exactly "BotLib v0.96": the Gladiator mod links against that ABI and the
 * parity contract pins the string by address (0x10037bef, see
 * tests/reference/botlib_contract.json).  The reconstruction carries its own
 * version so a shipped module can be identified, and it is kept strictly out
 * of the retail code paths -- nothing below is reachable from BotSetupLibrary,
 * so the startup banner still emits retail's four lines verbatim.
 *
 * The marker is an ordinary non-static const object rather than a string
 * literal inside the function so that it survives into .rodata and can be
 * recovered from a shipped binary with strings(1) / `what`.  It is not
 * exported: the module marks only GetBotAPI with GLADIATOR_API, builds with
 * WINDOWS_EXPORT_ALL_SYMBOLS off, and hides everything else on ELF/Mach-O.
 */
const char g_gladiatorReconstructionVersion[] =
	"@(#) " GLADIATOR_RECON_PRODUCT_NAME " " GLADIATOR_RECON_VERSION_FULL
	" [botlib ABI " GLADIATOR_LEGACY_BOTLIB_VERSION "]";

/*
=============
BotReconstructionVersion

Diagnostic accessor for the reconstruction version.  Deliberately absent from
the retail export table; see the marker comment above.
=============
*/
const char *BotReconstructionVersion(void)
{
	return g_gladiatorReconstructionVersion;
}

/*
=============
BotSetupLibraryWrapper

Initialises the botlib bridge and emits the historical startup banners.
=============
*/
static int BotSetupLibraryWrapper(void)
{
	if (BotLibraryInitialized())
	{
		BotInterface_Printf(PRT_ERROR, "bot library already setup\n");
		return BLERR_LIBRARYALREADYSETUP;
	}

	BotLib_LogOpen("botlib.log");
	BotInterface_PrintBanner(PRT_MESSAGE, "------- BotLib Initialization -------\n");
	BotInterface_PrintBanner(PRT_MESSAGE, "BotLib v0.96\n");
	BotInterface_SetImportTable(BotInterface_GetBotlibImportTable());

	int result = BotSetupLibrary();
	if (result != BLERR_NOERROR)
	{
		return result;
	}

	BotInterface_PrintBanner(PRT_MESSAGE, "-------------------------------------\n");
	return result;
}

/*
=============
BotShutdownLibraryWrapper

Tears down botlib state, then clears the retail state, import, and export
blocks after their callbacks are no longer needed.
=============
*/
static int BotShutdownLibraryWrapper(void)
{
	if (!BotLibraryInitialized())
	{
		BotInterface_Printf(PRT_ERROR, "bot library already shutdown\n");
		return BLERR_LIBRARYNOTSETUP;
	}

	if (g_botInterfaceConsoleChat != NULL)
	{
		BotDestroyChatState(g_botInterfaceConsoleChat);
		g_botInterfaceConsoleChat = NULL;
	}

	int result = BotShutdownLibrary();

	BotInterface_ResetMapCache();
	BotInterface_ResetEntityCache();
	BotInterface_ResetFrameQueues();
	Bridge_ResetCachedUpdates();
	g_botInterfaceDebugDrawEnabled = false;
	Q2Bridge_SetDebugLinesEnabled(false);
	BotInterface_FreeImportCache();
	BotMemory_SetAllocatorCallbacks(NULL, NULL);
	BotInterface_SetImportTable(NULL);
	BotInterface_SetImportCapture(NULL);
	Q2Bridge_SetImportTable(NULL);
	BotInterface_ClearEngineImportState();

	if (g_botRetailExportTable != NULL)
	{
		memset(g_botRetailExportTable, 0, sizeof(*g_botRetailExportTable));
	}
	if (g_botExtendedExportTable != NULL)
	{
		memset(g_botExtendedExportTable, 0, sizeof(*g_botExtendedExportTable));
	}
	g_botRetailExportTable = NULL;
	g_botExtendedExportTable = NULL;

	return result;
}

/*
=============
BotLibraryInitializedWrapper

Returns the retail AAS continuation-initialization state.
=============
*/
static int BotLibraryInitializedWrapper(void)
{
	return AAS_Initialized();
}

/*
=============
BotLibVarSetWrapper

Applies the retail local libvar update and unconditional zero return contract.
=============
*/
static int BotLibVarSetWrapper(char *var_name, char *value)
{
	(void)BotInterface_UpdateImportCache(var_name, value);
	LibVarSet(var_name, value);
	return BLERR_NOERROR;
}

static int BotInterface_BotLibraryInitialized(void)
{
    assert(BotInterface_GetEngineImport() != NULL);
    return BotLibraryInitialized() ? 1 : 0;
}

static int BotInterface_BotLibVarSet(char *var_name, char *value)
{
    assert(BotInterface_GetEngineImport() != NULL);
    BotInterface_Log(PRT_WARNING, __func__);
    (void)var_name;
    (void)value;
    return BLERR_NOERROR;
}

/*
=============
BotDefineWrapper

Adds a global precompiler define while preserving the retail return contract.
=============
*/
static int BotDefineWrapper(char *string)
{
	if (!PC_AddGlobalDefine(string))
	{
		BotInterface_Printf(PRT_ERROR,
			"couldn't add define %s\n",
			string);
	}

	return BLERR_NOERROR;
}

/*
 * Retail sub_10028c30 caches the twelve deathmatch libvars and, under ctf, the
 * two static flag goals plus six model indices in the globals listed beside
 * each field.  The reconstruction resolves flag goals and tech models by name
 * on demand, so these only mirror retail's map-load side effects.
 */
static libvar_t *g_botDeathmatchDmflags;      /* data_10064470 */
static libvar_t *g_botDeathmatchCtf;          /* data_100643ac */
static libvar_t *g_botDeathmatchCh;           /* data_1006445c */
static libvar_t *g_botDeathmatchRa;           /* data_10064464 */
static libvar_t *g_botDeathmatchFastchat;     /* data_1006447c */
static libvar_t *g_botDeathmatchNochat;       /* data_10064474 */
static libvar_t *g_botDeathmatchTeamplay;     /* data_10064460 */
static libvar_t *g_botDeathmatchUsehook;      /* data_10064458 */
static libvar_t *g_botDeathmatchRocketjump;   /* data_10064478 */
static libvar_t *g_botDeathmatchRunes;        /* data_10064468 */
static libvar_t *g_botDeathmatchTeamplayShell; /* data_10064488 */
static libvar_t *g_botDeathmatchAssimilation; /* data_10064480 */
static bot_goal_t g_botDeathmatchRedFlagGoal;  /* data_10064420 */
static bot_goal_t g_botDeathmatchBlueFlagGoal; /* data_100643e0 */
static int g_botDeathmatchFlagModel1;          /* data_10064484 */
static int g_botDeathmatchFlagModel2;          /* data_1006448c */
static int g_botDeathmatchResistanceModel;     /* data_1006449c */
static int g_botDeathmatchStrengthModel;       /* data_10064498 */
static int g_botDeathmatchHasteModel;          /* data_10064494 */
static int g_botDeathmatchRegenerationModel;   /* data_10064490 */

/*
=============
BotSetupDeathmatchAI

Reproduces the retail map-load deathmatch pass at 0x10028c30.
=============
*/
static void BotSetupDeathmatchAI(void)
{
	g_botDeathmatchDmflags = LibVar("dmflags", "0");
	g_botDeathmatchCtf = LibVar("ctf", "0");
	g_botDeathmatchCh = LibVar("ch", "0");
	g_botDeathmatchRa = LibVar("ra", "0");
	g_botDeathmatchFastchat = LibVar("fastchat", "0");
	g_botDeathmatchNochat = LibVar("nochat", "0");
	g_botDeathmatchTeamplay = LibVar("teamplay", "0");
	g_botDeathmatchUsehook = LibVar("usehook", "0");
	g_botDeathmatchRocketjump = LibVar("rocketjump", "1");
	g_botDeathmatchRunes = LibVar("runes", "0");
	g_botDeathmatchTeamplayShell = LibVar("teamplay_shell", "0");
	g_botDeathmatchAssimilation = LibVar("assimilation", "0");

	/*
	 * 0x10028d39 gates the rest on the ctf libvar, resolves the two static
	 * flag goals and warns through the Print import when either is missing
	 * (0x10028d5e / 0x10028d86), then caches the flag and rune model indices
	 * from 0x10028d9e onwards.
	 */
	if (LibVarGetValue("ctf") == 0.0f)
	{
		return;
	}

	char red_name[] = "Red Flag";
	char blue_name[] = "Blue Flag";
	if (BotGetLevelItemGoal(-1, red_name, &g_botDeathmatchRedFlagGoal) < 0)
	{
		BotInterface_Printf(PRT_WARNING, "CTF without Red Flag\n");
	}
	if (BotGetLevelItemGoal(-1, blue_name, &g_botDeathmatchBlueFlagGoal) < 0)
	{
		BotInterface_Printf(PRT_WARNING, "CTF without Blue Flag\n");
	}

	g_botDeathmatchFlagModel1 = IndexFromModel("players/male/flag1.md2");
	g_botDeathmatchFlagModel2 = IndexFromModel("players/male/flag2.md2");
	g_botDeathmatchResistanceModel =
		IndexFromModel("models/ctf/resistance/tris.md2");
	g_botDeathmatchStrengthModel =
		IndexFromModel("models/ctf/strength/tris.md2");
	g_botDeathmatchHasteModel = IndexFromModel("models/ctf/haste/tris.md2");
	g_botDeathmatchRegenerationModel =
		IndexFromModel("models/ctf/regeneration/tris.md2");
}

/*
=============
BotLoadMap

Loads a named map or refreshes retail asset-index tables when the name is NULL.
=============
*/
static int BotLoadMap(char *mapname,
	int modelindexes,
	char *modelindex[],
	int soundindexes,
	char *soundindex[],
	int imageindexes,
	char *imageindex[])
{
	if (BotInterface_GetEngineImport() == NULL)
	{
		return BLERR_LIBRARYNOTSETUP;
	}

	if (!BotLibraryEnsureSetup("BotLoadMap"))
	{
		return BLERR_LIBRARYNOTSETUP;
	}

	if (mapname == NULL)
	{
		return AAS_LoadMap(NULL,
			modelindexes,
			modelindex,
			soundindexes,
			soundindex,
			imageindexes,
			imageindex);
	}

	BotInterface_PrintBanner(PRT_MESSAGE,
		"------------ Map Loading ------------\n");

	int status = AAS_LoadMap(mapname,
							 modelindexes,
							 modelindex,
							 soundindexes,
							 soundindex,
							 imageindexes,
							 imageindex);
	if (status != BLERR_NOERROR)
	{
		return status;
	}

	/*
	 * Retail's map-load reset driver 0x10029c10 loops sub_10029a40 over every
	 * client record, and that routine reads the record head before the
	 * 0x10029afc memset (`10029a97  int32_t eax = *arg1`) and writes it back at
	 * 0x10029b42.  That head word is the only "active" test BotUpdateClient
	 * makes (0x1002989d), so a client set up before BotLoadMap keeps working
	 * across the load.  Clear the cached frames but keep the bridge's mirror of
	 * that flag.
	 */
	Bridge_ResetCachedFrames();
	BotInterface_ResetFrameQueues();
	BotInterface_ResetEntityCache();
	BotInterface_ResetMapCache();
	TranslateEntity_SetWorldLoaded(qfalse);
	bool recorded_assets = BotInterface_RecordMapAssets(mapname,
		modelindexes,
		modelindex,
		soundindexes,
		soundindex,
		imageindexes,
		imageindex);
	if (!recorded_assets)
	{
		BotInterface_Printf(PRT_WARNING,
			"[bot_interface] BotLoadMap: failed to record asset lists for %s\n",
			mapname);
	}
	BotGoal_SetMapModelIndexes(modelindexes, modelindex);

	if (recorded_assets && !BotMove_MoverCatalogueFinalize(
		BotInterface_MapModelEntries(),
		BotInterface_MapModelCount()))
	{
		BotInterface_Printf(PRT_WARNING,
			"[bot_interface] BotLoadMap: failed to finalize mover catalogue for %s\n",
			mapname);
	}

	BotState_ResetAllForNewMap();
	BotInitLevelItems();
	/* Retail 0x10029c63 runs sub_10028c30 as the last map-load reset step. */
	BotSetupDeathmatchAI();
	BotInterface_PrintBanner(PRT_MESSAGE,
		"-------------------------------------\n");

	TranslateEntity_SetWorldLoaded(qtrue);
	TranslateEntity_SetCurrentTime(0.0f);

	return BLERR_NOERROR;
}

/*
=============
BotSetupClient

Initializes a bot client while preserving the retail boolean export ABI.
=============
*/
static int BotSetupClient(int client, bot_settings_t *settings)
{
	if (!BotInterface_EnsureLibraryReady("BotSetupClient"))
	{
		return qfalse;
	}

	if (!BotInterface_ValidateClientNumber(client, "BotSetupClient"))
	{
		return qfalse;
	}

	/*
	 * Retail 0x10037f2c calls sub_100085f0 (0x100085f0 ->
	 * sub_10037850(aasworld.mapname, dentdata, entdatasize)) after the two
	 * guards and BEFORE the 0x100294a0 "client %d already setup" early-out, so
	 * every call that clears the guards registers the map's entity-lump source
	 * checksum.  The list insert at 0x100376b0 dedupes by source name.
	 */
	CRC_RegisterSourceData(aasworld.mapName,
		aasworld.bspEntityData,
		aasworld.bspEntityDataSize);

	bot_client_state_t *state = BotState_Get(client);
	bot_chatstate_t *retained_chat_state =
		state != NULL ? state->chat_state : NULL;
	if (state != NULL && state->active)
	{
		BotInterface_Printf(PRT_FATAL, "client %d already setup\n", client);
		return qfalse;
	}

	if (settings == NULL)
	{
		BotInterface_Printf(PRT_ERROR, "[bot_interface] BotSetupClient: NULL settings pointer for client %d\n", client);
		return qfalse;
	}

	state = BotState_Create(client);
	if (state == NULL)
	{
		BotInterface_Printf(PRT_ERROR, "[bot_interface] BotSetupClient: failed to allocate state for client %d\n", client);
		return qfalse;
	}

	/*
	 * The record slab is zero-cleared like retail's, so the combat block's
	 * reconstruction timestamps still read as an enemy sighted, killed, and
	 * damage taken at time zero.  Seed their "never happened" values before
	 * any setup step can record a real combat event.
	 */
	BotState_InitCombatSentinels(state);

	int status = BLERR_NOERROR;

	bot_character_t *character = BotLoadCharacter(settings->characterfile,
		settings->charactername);
	/* Retail stores the loader result before checking it and retains partial state. */
	state->character = character;
	if (character == NULL)
	{
		BotInterface_Printf(PRT_FATAL,
							"couldn't load bot character %s from %s\n",
							settings->charactername,
							settings->characterfile);
		return qfalse;
	}

	status = BotState_AttachCharacter(state, character);
	if (status != BLERR_NOERROR)
	{
		BotFreeCharacter(character);
		BotInterface_Printf(PRT_ERROR,
							"[bot_interface] BotSetupClient: failed to attach character resources for client %d\n",
							client);
		BotState_Destroy(client);
		return qfalse;
	}
	if (retained_chat_state != NULL)
	{
		state->chat_state = retained_chat_state;
	}

	memcpy(&state->settings, settings, sizeof(*settings));

	if (state->goal_handle <= 0)
	{
		state->goal_handle = AI_GoalBotlib_AllocState(client);
	}
	if (state->goal_handle <= 0)
	{
		BotInterface_Printf(PRT_ERROR,
							"[bot_interface] BotSetupClient: failed to allocate goal handle for client %d\n",
							client);
		BotState_Destroy(client);
		return qfalse;
	}
	const char *item_weights_file =
		Characteristic_String(state->character, BOT_CHARACTERISTIC_ITEMWEIGHTS);
	status = AI_GoalBotlib_LoadItemWeights(state->goal_handle,
		item_weights_file);
	if (status != BLERR_NOERROR)
	{
		return qfalse;
	}

	const bot_goalstate_t *goal_owner = AI_GoalBotlib_DebugPeek(state->goal_handle);
	if (goal_owner == NULL || goal_owner->itemweightconfig == NULL)
	{
		BotInterface_Printf(PRT_ERROR,
			"[bot_interface] BotSetupClient: item weights have no owning goal state for client %d\n",
			client);
		BotState_Destroy(client);
		return qfalse;
	}
	state->item_weights = goal_owner->itemweightconfig;

	if (state->weapon_state <= 0)
	{
		state->weapon_state = BotAllocWeaponState();
	}
	if (state->weapon_state <= 0)
	{
		BotInterface_Printf(PRT_ERROR,
			"[bot_interface] BotSetupClient: failed to allocate weapon state for client %d\n",
			client);
		BotState_Destroy(client);
		return qfalse;
	}

	const char *weapon_weights_file =
		Characteristic_String(state->character, BOT_CHARACTERISTIC_WEAPONWEIGHTS);
	status = BotLoadWeaponWeightsFresh(state->weapon_state, weapon_weights_file);
	if (status != BLERR_NOERROR)
	{
		AI_GoalBotlib_FreeItemWeights(state->goal_handle);
		return qfalse;
	}

	const bot_weaponstate_t *weapon_owner = BotWeaponStatePeek(state->weapon_state);
	if (weapon_owner == NULL || weapon_owner->weights == NULL)
	{
		BotInterface_Printf(PRT_ERROR,
			"[bot_interface] BotSetupClient: weapon weights have no owning weapon state for client %d\n",
			client);
		BotState_Destroy(client);
		return qfalse;
	}
	state->weapon_weights = weapon_owner->weights;

	const char *chat_file =
		Characteristic_String(state->character, BOT_CHARACTERISTIC_CHAT_FILE);
	const char *chat_name =
		Characteristic_String(state->character, BOT_CHARACTERISTIC_CHAT_NAME);
	if (state->chat_state == NULL)
	{
		state->chat_state = BotAllocChatState();
	}
	if (state->chat_state == NULL)
	{
		BotInterface_Printf(PRT_ERROR,
			"[bot_interface] BotSetupClient: failed to allocate chat state for client %d\n",
			client);
		BotState_Destroy(client);
		return qfalse;
	}
	status = BotLoadChatFile(state->chat_state, chat_file, chat_name);
	if (status != BLERR_NOERROR)
	{
		AI_GoalBotlib_FreeItemWeights(state->goal_handle);
		BotFreeWeaponWeights(state->weapon_state);
		return qfalse;
	}

	const char *gender =
		Characteristic_String(state->character, BOT_CHARACTERISTIC_GENDER);
	if (gender != NULL && (gender[0] == 'f' || gender[0] == 'F'))
	{
		BotSetChatGender(state->chat_state, CHAT_GENDERFEMALE);
	}
	else if (gender != NULL && (gender[0] == 'm' || gender[0] == 'M'))
	{
		BotSetChatGender(state->chat_state, CHAT_GENDERMALE);
	}
	else
	{
		BotSetChatGender(state->chat_state, CHAT_GENDERLESS);
	}

	if (state->goal_state == NULL)
	{
		state->goal_state = AI_GoalState_Create();
	}
	else
	{
		AI_GoalState_Reset(state->goal_state);
	}
	if (state->goal_state == NULL)
	{
		BotInterface_Printf(PRT_ERROR,
			"[bot_interface] BotSetupClient: failed to allocate goal state for client %d\n",
			client);
		BotState_Destroy(client);
		return qfalse;
	}

	if (state->move_handle <= 0)
	{
		state->move_handle = BotAllocMoveStateHandle();
	}
	else
	{
		BotResetMoveStateHandle(state->move_handle);
	}
	if (state->move_handle <= 0)
	{
		BotInterface_Printf(PRT_ERROR,
							"[bot_interface] BotSetupClient: failed to allocate move handle for client %d\n",
							client);
		BotState_Destroy(client);
		return qfalse;
	}

	ai_goal_services_t goal_services = {
		.weight_fn = BotInterface_GoalWeight,
		.travel_time_fn = BotInterface_GoalTravelTime,
		.notify_fn = BotInterface_GoalNotify,
		.area_fn = NULL,
		.userdata = state,
		.avoid_duration = 5.0f,
	};
	AI_GoalState_SetServices(state->goal_state, &goal_services);
	state->goal_avoid_duration = goal_services.avoid_duration;

	if (state->dm_state == NULL)
	{
		state->dm_state = AI_DMState_Create(client);
	}
	else
	{
		AI_DMState_Reset(state->dm_state);
	}
	if (state->dm_state == NULL)
	{
		BotInterface_Printf(PRT_ERROR,
							"[bot_interface] BotSetupClient: failed to allocate DM state for client %d\n",
							client);
		BotState_Destroy(client);
		return qfalse;
	}

	state->active = true;
	state->client_number = client;
	state->entity_number = client + 1;
	state->client_commands_pending = true;
	state->enter_game_time = AAS_Time();
	/*
	 * Retail keeps presentation settings in a table separate from the client
	 * record, so they outlive a client shutdown that clears the record and are
	 * only replaced by an explicit BotClientSettings call.  Re-seed the record's
	 * mirror from that table so a client set up again reports the settings the
	 * game last supplied rather than a cleared name and skin.
	 */
	const bot_clientsettings_t *retained_settings =
		BotState_ClientSettings(client);
	if (retained_settings != NULL)
	{
		state->client_settings = *retained_settings;
	}
	BotState_SetActive(state, true);
	Bridge_ClearClientSlot(client);
	Bridge_SetClientActive(client, qtrue);
	return qtrue;
}

/*
=============
BotShutdownClient

Shuts down a bot client slot and clears the bridge cache.
=============
*/
static int BotShutdownClient(int client)
{
	if (!BotInterface_EnsureLibraryReady("BotShutdownClient"))
	{
		return BLERR_LIBRARYNOTSETUP;
	}

	if (!BotInterface_ValidateClientNumber(client, "BotShutdownClient"))
	{
		return BLERR_INVALIDCLIENTNUMBER;
	}

	bot_client_state_t *state = BotState_Get(client);
	if (state == NULL || !state->active)
	{
		BotInterface_Printf(PRT_ERROR, "client %d already shutdown\n", client);
		Bridge_ClearClientSlot(client);
		return BLERR_AICLIENTALREADYSHUTDOWN;
	}

	if (BotAI_ConstructLifecycleChat(state,
		"exit_game",
		CHARACTERISTIC_CHAT_ENTEREXITGAME,
		false))
	{
		BotEnterChat(state->chat_state, state->client_number, 0);
	}

	BotState_Destroy(client);
	Bridge_SetClientActive(client, qfalse);
	Bridge_ClearClientSlot(client);
	return BLERR_NOERROR;
}

/*
=============
BotMoveClient

Migrates an active bot client to a new slot and preserves bridge state.
=============
*/
static int BotMoveClient(int oldclnum, int newclnum)
{
	if (!BotInterface_EnsureLibraryReady("BotMoveClient"))
	{
		return BLERR_LIBRARYNOTSETUP;
	}

	if (!BotInterface_ValidateClientNumber(oldclnum, "BotMoveClient, parm0"))
	{
		return BLERR_INVALIDCLIENTNUMBER;
	}

	if (!BotInterface_ValidateClientNumber(newclnum, "BotMoveClient, parm1"))
	{
		return BLERR_INVALIDCLIENTNUMBER;
	}

	bot_client_state_t *state = BotState_Get(oldclnum);
	if (state == NULL || !state->active)
	{
		BotInterface_Printf(PRT_FATAL,
			"tried to move inactive bot client\n");
		return BLERR_AIMOVEINACTIVECLIENT;
	}

	bot_client_state_t *destination = BotState_Get(newclnum);
	if (destination != NULL && destination->active)
	{
		BotInterface_Printf(PRT_FATAL,
			"tried to move client to active client\n");
		return BLERR_AIMOVETOACTIVECLIENT;
	}

	BotState_Move(oldclnum, newclnum);
	Bridge_ClearClientSlot(newclnum);
	Bridge_SetClientActive(newclnum, qtrue);
	int status = Bridge_MoveClientSlot(oldclnum, newclnum);
	if (status != BLERR_NOERROR)
	{
		BotInterface_Printf(PRT_ERROR,
		                    "[bot_interface] BotMoveClient: bridge move failed for %d -> %d\n",
		                    oldclnum,
		                    newclnum);
		Bridge_ClearClientSlot(newclnum);
		Bridge_SetClientActive(newclnum, qfalse);
		Bridge_SetClientActive(oldclnum, qtrue);
		BotState_Move(newclnum, oldclnum);
		return status;
	}
	Bridge_SetClientActive(oldclnum, qfalse);
	Bridge_SetClientActive(newclnum, qtrue);

	return BLERR_NOERROR;
}

/*
=============
BotClientSettings

Stores game-provided presentation settings for a client slot.
=============
*/
static int BotClientSettings(int client, bot_clientsettings_t *settings)
{
	if (!BotInterface_EnsureLibraryReady("BotClientSettings"))
	{
		return BLERR_LIBRARYNOTSETUP;
	}

	if (!BotInterface_ValidateClientNumber(client, "BotClientSettings"))
	{
		return BLERR_INVALIDCLIENTNUMBER;
	}

	if (settings == NULL)
	{
		BotInterface_Printf(PRT_ERROR, "[bot_interface] BotClientSettings: NULL output buffer\n");
		return BLERR_INVALIDIMPORT;
	}

	return BotState_SetClientSettings(client, settings);
}

/*
=============
BotSettings

Updates the stored bot setup configuration for an active client.
=============
*/
static int BotSettings(int client, bot_settings_t *settings)
{
	if (!BotInterface_EnsureLibraryReady("BotSettings"))
	{
		return BLERR_LIBRARYNOTSETUP;
	}

	if (!BotInterface_ValidateClientNumber(client, "BotSettings"))
	{
		return BLERR_INVALIDCLIENTNUMBER;
	}

	bot_client_state_t *state = BotState_Get(client);
	if (state == NULL || !state->active)
	{
		BotInterface_Printf(PRT_FATAL,
			"tried to update settings of inactive client\n");
		return BLERR_SETTINGSINACTIVECLIENT;
	}

	if (settings == NULL)
	{
		BotInterface_Printf(PRT_ERROR, "[bot_interface] BotSettings: NULL output buffer\n");
		return BLERR_INVALIDIMPORT;
	}

	memcpy(&state->settings, settings, sizeof(state->settings));
	return BLERR_NOERROR;
}

static int BotAI_CountCoopEnemiesInArea(int area)
{
	int enemy_count = 0;

	if (area <= 0 || !AAS_Initialized() || aasworld.entities == NULL)
	{
		return 0;
	}

	for (int entity = aasworld.maxClients + 1;
		entity < aasworld.maxEntities;
		entity += 1)
	{
		aas_entityinfo_t entity_info;
		memset(&entity_info, 0, sizeof(entity_info));
		AAS_EntityInfo(entity, &entity_info);
		if (BotAI_EntityIsDead(&entity_info) ||
			AAS_PointAreaNum(entity_info.origin) != area)
		{
			continue;
		}
		enemy_count += 1;
	}
	return enemy_count;
}

/*
=============
BotAI_UpdateCoopBotArea

Observe the companion's current AAS area once per AI frame.  A current area
becomes CLEARED only after combat was observed there and no live coop enemy
remains.  This is deliberately conservative; the complete adjacency remains
in the opt-in map dump.
=============
*/
static void BotAI_UpdateCoopBotArea(bot_client_state_t *state)
{
	bot_coop_area_memory_t *memory;
	int bot_area;
	int previous_area;
	int previous_enemy_count;
	int enemy_count;
	bot_coop_area_state_t previous_state;
	const char *area_state;
	bool dangerous;

	if (state == NULL || !BotAI_CoopMode() ||
		(LibVarGetValue("coopbot_map_model") == 0.0f &&
			LibVarGetValue("coopbot_objective_htn") == 0.0f &&
			LibVarGetValue("coopbot_safe_area_retreat") == 0.0f) ||
		!AAS_Initialized())
	{
		return;
	}
	bot_area = AAS_PointAreaNum(state->last_client_update.origin);
	if (bot_area <= 0)
	{
		return;
	}
	memory = BotInterface_FindCoopAreaMemory(state, bot_area, true);
	if (memory == NULL)
	{
		return;
	}
	enemy_count = BotAI_CountCoopEnemiesInArea(bot_area);
	dangerous = BotInterface_CoopAreaIsDangerous(state, enemy_count);
	previous_area = state->coop_bot_area_valid
		? state->coop_bot_last_area : 0;
	previous_state = memory->state;
	previous_enemy_count = memory->enemy_count;
	if (enemy_count > 0)
	{
		memory->combat_seen = true;
		if (dangerous)
		{
			memory->state = BOT_COOP_AREA_DANGEROUS;
		}
		else if (previous_enemy_count > enemy_count)
		{
			memory->state = BOT_COOP_AREA_PARTIALLY_CLEARED;
		}
		else
		{
			memory->state = BOT_COOP_AREA_ACTIVE_COMBAT;
		}
	}
	else if (memory->combat_seen)
	{
		memory->state = BOT_COOP_AREA_CLEARED;
	}
	else
	{
		memory->state = BOT_COOP_AREA_VISITED;
	}
	memory->enemy_count = enemy_count;
	memory->last_observed = AAS_Time();
	state->coop_current_area = bot_area;
	state->coop_area_enemy_count = enemy_count;
	state->coop_area_combat_seen = memory->combat_seen;
	state->coop_area_state = memory->state;
	area_state = state->coop_role == BOT_COOP_ROLE_REGROUP
		? "REGROUP" : BotInterface_CoopAreaStateName(memory->state);
	if (!state->coop_bot_area_valid || bot_area != previous_area ||
		memory->state != previous_state ||
		previous_enemy_count != enemy_count)
	{
		BotInterface_LogCoopAreaTransition(state, "bot", previous_area,
			bot_area, area_state);
	}
	state->coop_bot_last_area = bot_area;
	state->coop_bot_area_valid = true;
	BotInterface_RecordCoopSafeArea(state, bot_area, enemy_count);
}

/*
=============
BotStartFrame

Advance the shared retail botlib/AAS frame state.
=============
*/
static int BotStartFrame(float time)
{
	if (!BotInterface_EnsureLibraryReady("BotStartFrame"))
	{
		return BLERR_LIBRARYNOTSETUP;
	}
	if (!g_botInterfaceStartFrameLogged)
	{
		g_botInterfaceStartFrameLogged = true;
		BotLib_LogWrite(
			"coopbot_start_frame aas_initialized=%d map_model=%.1f",
			AAS_Initialized(), LibVarGetValue("coopbot_map_model"));
	}

	AAS_FrameSynchronise(time);
	AAS_InvalidateEntities();
	BotInterface_BeginFrame(time);
	AAS_ContinueInit(time);
	BotInterface_DumpCoopMapModel(BotInterface_CoopMapEntities());
	AAS_BeginFrameRouting();
	AAS_RunFrameDiagnostics();

	return BLERR_NOERROR;
}

static int BotUpdateClient(int client, bot_updateclient_t *buc)
{
	if (!BotInterface_EnsureLibraryReady("BotUpdateClient"))
	{
		return BLERR_LIBRARYNOTSETUP;
	}

	if (!BotInterface_ValidateClientNumber(client, "BotUpdateClient"))
	{
		return BLERR_INVALIDCLIENTNUMBER;
	}

    bot_client_state_t *state = BotState_Get(client);
    if (state == NULL || !state->active)
    {
		BotInterface_Printf(PRT_FATAL,
			"tried to updated inactive bot client\n");
        return BLERR_AIUPDATEINACTIVECLIENT;
    }

    int status = Bridge_UpdateClient(client, buc);
    if (status != BLERR_NOERROR)
    {
        return status;
    }

    AASClientFrame translated;
    if (!Bridge_ReadClientFrame(client, &translated))
    {
        return BLERR_AIUPDATEINACTIVECLIENT;
    }

    bot_updateclient_t quantised = {0};
    quantised.pm_type = translated.pm_type;
    VectorCopy(translated.origin, quantised.origin);
    VectorCopy(translated.velocity, quantised.velocity);
    VectorCopy(translated.delta_angles, quantised.delta_angles);
    quantised.pm_flags = translated.pm_flags;
    quantised.pm_time = translated.pm_time;
    quantised.gravity = translated.gravity;
    VectorCopy(translated.viewangles, quantised.viewangles);
    VectorCopy(translated.viewoffset, quantised.viewoffset);
    VectorCopy(translated.kick_angles, quantised.kick_angles);
    VectorCopy(translated.gunangles, quantised.gunangles);
    VectorCopy(translated.gunoffset, quantised.gunoffset);
    quantised.gunindex = translated.gunindex;
    quantised.gunframe = translated.gunframe;
    memcpy(quantised.blend, translated.blend, sizeof(quantised.blend));
    quantised.fov = translated.fov;
    quantised.rdflags = translated.rdflags;
    memcpy(quantised.stats, translated.stats, sizeof(quantised.stats));
    memcpy(quantised.inventory, translated.inventory, sizeof(quantised.inventory));

    if (buc != NULL)
    {
        *buc = quantised;
    }

    state->last_client_update = quantised;
    state->client_update_valid = true;
    state->last_update_time = translated.last_update_time;
	AI_DMState_ApplyDeltaAngles(state->dm_state, quantised.delta_angles);

    if (state->goal_state != NULL)
    {
        status = AI_GoalState_RecordClientUpdate(state->goal_state, &quantised);
        if (status != BLERR_NOERROR)
        {
            return status;
        }
    }

    return BLERR_NOERROR;
}






typedef enum bot_ai_frame_work_e
{
	BOT_AI_FRAME_WORK_NONE = 0,
	BOT_AI_FRAME_WORK_STAND,
	BOT_AI_FRAME_WORK_GOAL,
	BOT_AI_FRAME_WORK_FIGHT,
	BOT_AI_FRAME_WORK_CHASE,
	BOT_AI_FRAME_WORK_BATTLE_NBG,
	BOT_AI_FRAME_WORK_BATTLE_RETREAT,
	BOT_AI_FRAME_WORK_BATTLE_RETREAT_IDLE,
} bot_ai_frame_work_t;

typedef struct bot_ai_node_frame_s
{
	bot_ai_frame_work_t work;
	bool post_acquire_enemy;
	float thinktime;
	ai_dm_enemy_info_t enemy;
	bot_goal_t movement_goal;
	bool has_movement_goal;
} bot_ai_node_frame_t;



typedef enum bot_team_goal_result_e
{
	BOT_TEAM_GOAL_NONE = 0,
	BOT_TEAM_GOAL_READY,
	BOT_TEAM_GOAL_HANDLED,
} bot_team_goal_result_t;

static bot_team_goal_result_t BotAI_ResolveTeamLongTermGoal(bot_client_state_t *state,
	float thinktime,
	bot_goal_t *goal,
	vec3_t held_viewangles,
	bool *held_view_set,
	int *held_actionflags,
	int retreat);

/*
=============
BotAI_NodeStep

Executes one retail AI-node decision and reports whether the frame is done.
=============
*/
static int BotAI_NodeStep(bot_client_state_t *state, void *context)
{
	bot_ai_node_frame_t *frame = (bot_ai_node_frame_t *)context;
	if (state == NULL || frame == NULL)
	{
		return qtrue;
	}

	frame->work = BOT_AI_FRAME_WORK_NONE;
	frame->post_acquire_enemy = false;
	BotAI_InitEnemyInfo(&frame->enemy);

	switch (state->ai_node)
	{
	case BOT_AI_NODE_OBSERVER:
	case BOT_AI_NODE_INTERMISSION:
		frame->work = BOT_AI_FRAME_WORK_STAND;
		return qtrue;

	case BOT_AI_NODE_STAND:
		if (BotAI_FindEnemy(state, &frame->enemy))
		{
			BotAI_EnterNode(state, BOT_AI_NODE_BATTLE_FIGHT);
			return qfalse;
		}
		if (state->chat_standing)
		{
			if (!BotAI_ReplyStandActive(state, frame->thinktime))
			{
				BotAI_EnterNode(state, BOT_AI_NODE_SEEK_LTG);
				return qfalse;
			}
			frame->work = BOT_AI_FRAME_WORK_STAND;
			return qtrue;
		}
		frame->work = BOT_AI_FRAME_WORK_STAND;
		return qtrue;

	case BOT_AI_NODE_ACTIVATE_ENTITY:
		/* Activation commits movement before its delayed enemy-acquisition pass. */
		state->combat.current_enemy = 0;
		if (BotTouchingGoal(state->last_client_update.origin,
			&state->activation_goal))
		{
			state->activation_goal_time = 0.0f;
			if (state->coop_control_phase == BOT_COOP_CONTROL_NAVIGATE)
			{
				if (LibVarGetValue("coopbot_log") >= 1.0f)
				{
					BotLib_LogWriteTimeStamped(
						"coopbot_objective client=%d objective=OPEN_PATH "
						"phase=ACTIVATE control_entity=%d goal_area=%d",
						state->client_number,
						state->coop_control_entity,
						state->coop_control_goal_area);
				}
				/* Contact is the Activate step.  Keep the bot at the
				 * control until the human catches up instead of letting the
				 * next LTG immediately pull it through the opened path. */
				BotAI_CoopSetControlObjectivePhase(state,
					BOT_COOP_CONTROL_WAIT_PLAYER,
					state->coop_control_entity,
					state->coop_control_goal_area);
			}
		}
		if (AAS_Time() > state->activation_goal_time)
		{
			if (state->coop_control_phase == BOT_COOP_CONTROL_NAVIGATE)
			{
				BotAI_CoopSetControlObjectivePhase(state,
					BOT_COOP_CONTROL_RETRY,
					state->coop_control_entity,
					state->coop_control_goal_area);
			}
			BotAI_EnterNode(state, BOT_AI_NODE_SEEK_NBG);
			return qfalse;
		}
		frame->post_acquire_enemy = true;
		frame->work = BOT_AI_FRAME_WORK_GOAL;
		return qtrue;

	case BOT_AI_NODE_SEEK_NBG:
		state->combat.current_enemy = 0;
	{
		bot_goal_t nearby_goal;
		bool has_nearby_goal = state->goal_handle > 0 &&
			AI_GoalBotlib_GetTopGoal(state->goal_handle, &nearby_goal) != 0;
		if (!has_nearby_goal ||
			BotAI_NearbyGoalReached(state, &nearby_goal))
		{
			state->nearby_goal_time = 0.0f;
		}
		if (!has_nearby_goal || AAS_Time() > state->nearby_goal_time)
		{
			if (state->goal_handle > 0)
			{
				AI_GoalBotlib_PopGoal(state->goal_handle);
			}
			BotAI_EnterNode(state, BOT_AI_NODE_SEEK_LTG);
			return qfalse;
		}
	}
		frame->post_acquire_enemy = true;
		frame->work = BOT_AI_FRAME_WORK_GOAL;
		return qtrue;

	case BOT_AI_NODE_SEEK_LTG:
	{
		if (BotAI_ConstructRandomChat(state, frame->thinktime))
		{
			state->stand_time = AAS_Time() + BotAI_ChatTime(state);
			state->chat_standing = true;
			BotAI_EnterNode(state, BOT_AI_NODE_STAND);
			return qfalse;
		}
		state->combat.current_enemy = 0;
		BotAI_TryRecentEnemyDeathWave(state, frame->thinktime);
		if (BotAI_FindEnemy(state, &frame->enemy))
		{
			BotAI_EnterFoundEnemy(state, false);
			return qfalse;
		}
		frame->work = BOT_AI_FRAME_WORK_GOAL;
		return qtrue;
	}

	case BOT_AI_NODE_BATTLE_FIGHT:
		if (state->combat.current_enemy > 0 &&
			state->combat.current_enemy < aasworld.maxEntities &&
			aasworld.entities != NULL &&
			aasworld.entities[state->combat.current_enemy].inuse)
		{
			aas_entityinfo_t entity_info;
			AAS_EntityInfo(state->combat.current_enemy, &entity_info);
			if (BotAI_EntityIsDead(&entity_info))
			{
				if (BotAI_ConstructKillChat(state))
				{
					state->stand_time = AAS_Time() + BotAI_ChatTime(state);
					state->chat_standing = true;
					BotAI_EnterNode(state, BOT_AI_NODE_STAND);
					return qfalse;
				}
			}
		}
		if (!BotAI_ResolveCurrentEnemy(state, &frame->enemy))
		{
			BotAI_EnterNode(state, BOT_AI_NODE_SEEK_LTG);
			return qfalse;
		}
		/*
		 * Retail commits the reachable enemy area and origin before it
		 * projects the enemy into the battle inventory, so Battle Chase sees
		 * the newest sample even when the visibility test diverts this frame.
		 */
		BotAI_RecordLastEnemyLocation(state, &frame->enemy);
		BotAI_UpdateEnemyBattleInventory(state, frame->enemy.entity);
		if (!frame->enemy.visible)
		{
			if (BotAI_WantsToChase(state))
			{
				BotAI_EnterBattleChase(state);
			}
			else
			{
				BotAI_EnterNode(state, BOT_AI_NODE_SEEK_LTG);
			}
			return qfalse;
		}
		frame->work = BOT_AI_FRAME_WORK_FIGHT;
		return qtrue;

	case BOT_AI_NODE_BATTLE_CHASE:
		if (state->combat.current_enemy == 0)
		{
			BotAI_EnterNode(state, BOT_AI_NODE_SEEK_LTG);
			return qfalse;
		}
		if (state->coop_joint_retreat_active ||
			BotAI_CoopIntentIsConfident(state,
			BOT_COOP_INTENT_RETREAT) ||
			BotAI_CoopRoleBreaksChase(state))
		{
			/* Do not keep chasing after the human has clearly fallen back. */
			BotAI_EnterNode(state, BOT_AI_NODE_BATTLE_RETREAT);
			return qfalse;
		}
		if (BotAI_CurrentEnemyVisible(state))
		{
			BotAI_ResetFightNavigation(state, false);
			BotAI_EnterNode(state, BOT_AI_NODE_BATTLE_FIGHT);
			return qfalse;
		}
		if (BotAI_FindEnemy(state, &frame->enemy))
		{
			BotAI_EnterNode(state, BOT_AI_NODE_BATTLE_FIGHT);
			return qfalse;
		}
		if (state->combat.last_enemy_area == 0)
		{
			BotAI_EnterNode(state, BOT_AI_NODE_SEEK_LTG);
			return qfalse;
		}
		bot_goal_t chase_goal;
		BotAI_BuildBattleChaseGoal(state, &chase_goal);
		if (BotTouchingGoal(state->last_client_update.origin, &chase_goal))
		{
			state->combat.chase_time = 0.0f;
		}
		if (AAS_Time() > state->combat.chase_time)
		{
			BotAI_EnterNode(state, BOT_AI_NODE_SEEK_LTG);
			return qfalse;
		}
		if (BotAI_TryBattleChaseNearbyGoal(state,
			&chase_goal,
			BotAI_BattleChaseTravelFlags(state)))
		{
			return qfalse;
		}
		frame->work = BOT_AI_FRAME_WORK_CHASE;
		return qtrue;

	case BOT_AI_NODE_BATTLE_RETREAT:
	{
		if (state->combat.current_enemy == 0 ||
			!BotAI_ResolveCurrentEnemy(state, &frame->enemy))
		{
			BotAI_EnterNode(state, BOT_AI_NODE_SEEK_LTG);
			return qfalse;
		}
		BotAI_UpdateEnemyBattleInventory(state,
			state->combat.current_enemy);
		if (BotAI_WantsToChase(state) &&
			!state->coop_joint_retreat_active &&
			!BotAI_CoopRoleBreaksChase(state))
		{
			if (state->goal_handle > 0)
			{
				AI_GoalBotlib_EmptyGoalStack(state->goal_handle);
			}
			BotAI_EnterBattleChase(state);
			return qfalse;
		}
		if (!frame->enemy.visible)
		{
			BotAI_EnterNode(state, BOT_AI_NODE_SEEK_LTG);
			return qfalse;
		}

		/*
		 * 0x10020792: BotCTFRetreatGoals runs first and only promotes the LTG,
		 * then 0x100207a6 resolves the goal with BotLongTermGoal(bs, tfl, 1).
		 * Battle Retreat passes its own travel mask, so the item tail must use
		 * that rather than Seek LTG's rocket-jump-capable one.
		 */
		if (LibVarGetValue("ctf") != 0.0f)
		{
			BotAI_CTFRetreatGoals(state);
		}

		bot_goal_t retreat_goal;
		int travel_flags = BotAI_BattleRetreatTravelFlags();
		vec3_t retreat_viewangles;
		bool retreat_view_set = false;
		int retreat_actionflags = 0;
		bool has_retreat_goal = false;
		bot_team_goal_result_t retreat_result =
			BotAI_ResolveTeamLongTermGoal(state,
				frame->thinktime,
				&retreat_goal,
				retreat_viewangles,
				&retreat_view_set,
				&retreat_actionflags,
				1);
		if (retreat_result == BOT_TEAM_GOAL_READY)
		{
			has_retreat_goal = true;
		}
		else if (retreat_result == BOT_TEAM_GOAL_NONE)
		{
			has_retreat_goal = BotAI_GetItemLongTermGoal(state,
				&retreat_goal,
				travel_flags);
		}
		if (!has_retreat_goal)
		{
			/* Team and item retreat semantics remain higher priority. */
			has_retreat_goal = BotAI_BuildCoopSafeAreaGoal(state,
				&retreat_goal);
		}
		if (!has_retreat_goal)
		{
			/*
			 * 0x100207b1: retail keeps Battle Retreat active and only advances
			 * its view turn when BotLongTermGoal yields nothing.
			 */
			frame->work = BOT_AI_FRAME_WORK_BATTLE_RETREAT_IDLE;
			return qtrue;
		}
		if (BotAI_TryBattleChaseNearbyGoal(state,
			&retreat_goal,
			travel_flags))
		{
			return qfalse;
		}
		frame->movement_goal = retreat_goal;
		frame->has_movement_goal = true;
		frame->work = BOT_AI_FRAME_WORK_BATTLE_RETREAT;
		return qtrue;
	}

	case BOT_AI_NODE_BATTLE_NBG:
	{
		if (state->combat.current_enemy == 0 ||
			!BotAI_ResolveCurrentEnemy(state, &frame->enemy))
		{
			BotAI_EnterNode(state, BOT_AI_NODE_SEEK_NBG);
			return qfalse;
		}
		BotAI_RecordLastEnemyLocation(state, &frame->enemy);
		bot_goal_t nearby_goal;
		bool has_nearby_goal = state->goal_handle > 0 &&
			AI_GoalBotlib_GetTopGoal(state->goal_handle, &nearby_goal) != 0;
		if (!has_nearby_goal ||
			BotAI_TouchingNearbyGoal(state, &nearby_goal))
		{
			state->nearby_goal_time = 0.0f;
		}
		if (AAS_Time() > state->nearby_goal_time)
		{
			if (state->goal_handle > 0)
			{
				AI_GoalBotlib_PopGoal(state->goal_handle);
			}
			if (state->goal_handle <= 0 ||
				!AI_GoalBotlib_GetTopGoal(state->goal_handle, &nearby_goal))
			{
				BotAI_EnterNode(state, BOT_AI_NODE_BATTLE_FIGHT);
			}
			else
			{
				BotAI_EnterNode(state, BOT_AI_NODE_BATTLE_RETREAT);
			}
			return qfalse;
		}
		frame->work = BOT_AI_FRAME_WORK_BATTLE_NBG;
		return qtrue;
	}
	}

	BotAI_EnterNode(state, BOT_AI_NODE_SEEK_LTG);
	return qfalse;
}

/*
=============
BotAI_RunNodeSwitchLoop

Runs immediate retail node transitions until work completes or 50 switches.
=============
*/
/*
 * Retail keeps numnodeswitches at 0x100644a0 and a nodeswitch[51][144] record
 * array at 0x10064a80 (ref be_ai2_dmnet.c:58-59).  Nothing downstream reads
 * them; they exist only for the overflow dump.
 */
#define BOT_AI_NODE_SWITCH_RECORD_CHARS 144

static char g_bot_node_switches[BOT_AI_MAX_NODE_SWITCHES + 1]
	[BOT_AI_NODE_SWITCH_RECORD_CHARS];
static int g_bot_node_switch_count;

/*
=============
BotAI_NodeSwitchName

Return the retail node name recorded on entry to each AI node.

The strings are the literals the AIEnter_* routines pass to BotRecordNodeSwitch
(sub_1001d3a0).  Retail has a "respawn" entry too; this reconstruction has no
separate respawn node.
=============
*/
static const char *BotAI_NodeSwitchName(int node)
{
	switch (node)
	{
	case BOT_AI_NODE_OBSERVER: return "observer";
	case BOT_AI_NODE_INTERMISSION: return "intermission";
	case BOT_AI_NODE_STAND: return "stand";
	case BOT_AI_NODE_ACTIVATE_ENTITY: return "activate entity";
	case BOT_AI_NODE_BATTLE_FIGHT: return "battle fight";
	case BOT_AI_NODE_BATTLE_CHASE: return "battle chase";
	case BOT_AI_NODE_BATTLE_RETREAT: return "battle retreat";
	case BOT_AI_NODE_BATTLE_NBG: return "battle NBG";
	case BOT_AI_NODE_SEEK_NBG: return "seek NBG";
	case BOT_AI_NODE_SEEK_LTG: return "seek LTG";
	default: break;
	}
	return "";
}

/*
=============
BotAI_ResetNodeSwitches

Retail sub_1001d2b0: clear the per-frame node-switch record count.  Called from
0x10028b7e, immediately before the 50-iteration node loop.
=============
*/
static void BotAI_ResetNodeSwitches(void)
{
	g_bot_node_switch_count = 0;
}

/*
=============
BotAI_EnterNode

Switch the bot to an AI node and record it, reproducing retail's
BotRecordNodeSwitch (sub_1001d3a0).

0x1001d3a0 formats "%s at %2.1f entered %s: %s\n" from ClientName, AAS_Time(),
the node name and a per-node string, then increments numnodeswitches.  The
seek nodes pass BotGoalName(goal->number) for the last field and the literal
"no goal" when none was selected; every other node passes "".

Retail records from each AIEnter_* with the goal it just chose.  This
reconstruction has no separate AIEnter_* layer, so the seek nodes report the
goal currently on top of the stack - the same goal in the ordinary case, and an
approximation only when the record is taken before the stack is updated.
=============
*/
void BotAI_EnterNode(bot_client_state_t *state, int node)
{
	state->ai_node = node;

	if (g_bot_node_switch_count < 0 ||
		g_bot_node_switch_count >= BOT_AI_MAX_NODE_SWITCHES + 1)
	{
		return;
	}

	char detail[BOT_AI_NODE_SWITCH_RECORD_CHARS];
	detail[0] = '\0';
	if (node == BOT_AI_NODE_SEEK_NBG || node == BOT_AI_NODE_SEEK_LTG)
	{
		bot_goal_t goal;
		if (state->goal_handle > 0 &&
			AI_GoalBotlib_GetTopGoal(state->goal_handle, &goal) != 0)
		{
			BotGoalName(goal.number, detail, (int)sizeof(detail));
		}
		else
		{
			snprintf(detail, sizeof(detail), "no goal");
		}
	}

	if (LibVarGetValue("coopbot_log") >= 2.0f)
	{
		BotLib_LogWriteTimeStamped(
			"event=node client=%d name=\"%s\" node=\"%s\" goal=\"%s\" enemy=%d",
			state->client_number,
			BotState_ClientName(state->client_number),
			BotAI_NodeSwitchName(node), detail,
			state->combat.current_enemy);
	}

	snprintf(g_bot_node_switches[g_bot_node_switch_count],
		sizeof(g_bot_node_switches[0]),
		"%s at %2.1f entered %s: %s\n",
		BotState_ClientName(state->client_number),
		AAS_Time(),
		BotAI_NodeSwitchName(node),
		detail);
	g_bot_node_switch_count += 1;
}

/*
=============
BotAI_DumpNodeSwitches

Reproduce retail BotDumpNodeSwitches (sub_1001d2d0): build the header plus
every recorded switch into one buffer and emit it as a single PRT_FATAL
message.  Retail passes that buffer to Print as the format string and sizes it
at only 1400 bytes against up to 50 records, so it can smash its own stack; we
pass "%s" and size the buffer for the whole set.
=============
*/
static void BotAI_DumpNodeSwitches(bot_client_state_t *state)
{
	char message[(BOT_AI_MAX_NODE_SWITCHES + 2) *
		BOT_AI_NODE_SWITCH_RECORD_CHARS];
	int written = snprintf(message,
		sizeof(message),
		"%s at %1.1f switched more than %d AI nodes\n",
		BotState_ClientName(state->client_number),
		AAS_Time(),
		BOT_AI_MAX_NODE_SWITCHES);
	size_t length = (written > 0) ? (size_t)written : 0U;
	if (length >= sizeof(message))
	{
		length = sizeof(message) - 1U;
	}

	for (int index = 0; index < g_bot_node_switch_count; ++index)
	{
		const char *record = g_bot_node_switches[index];
		size_t record_length = strlen(record);
		if (record_length >= sizeof(message) - length)
		{
			break;
		}
		memcpy(message + length, record, record_length + 1U);
		length += record_length;
	}

	BotInterface_Printf(PRT_FATAL, "%s", message);
}

int BotAI_RunNodeSwitchLoop(bot_client_state_t *state,
	bot_ai_node_step_fn step,
	void *context)
{
	if (state == NULL || step == NULL)
	{
		return qfalse;
	}

	state->ai_node_switches = 0;
	state->ai_node_overflow = false;
	/* 0x10028b7e resets the record count immediately before the node loop.
	   Retail's AIEnter_Stand call at 0x10028b75 happens BEFORE this, so its
	   record is intentionally discarded. */
	BotAI_ResetNodeSwitches();
	while (state->ai_node_switches < BOT_AI_MAX_NODE_SWITCHES)
	{
		if (step(state, context) != 0)
		{
			return qtrue;
		}
		state->ai_node_switches++;
	}

	state->ai_node_overflow = true;
	return qfalse;
}

/*
=============
BotAI_TeamGoalDistance

Returns the direct Euclidean goal distance used by the retail defend and camp
long-term-goal branches.
=============
*/
static float BotAI_TeamGoalDistance(const bot_client_state_t *state,
	const bot_goal_t *goal)
{
	if (state == NULL || goal == NULL)
	{
		return 0.0f;
	}

	vec3_t direction;
	VectorSubtract(goal->origin, state->last_client_update.origin, direction);
	return sqrtf(DotProduct(direction, direction));
}

/*
=============
BotAI_TeamPatrolName

Builds the ordered waypoint list supplied to the delayed patrol-start chat.
=============
*/
static void BotAI_TeamPatrolName(const bot_client_state_t *state,
	char *name,
	size_t name_size)
{
	if (name == NULL || name_size == 0U)
	{
		return;
	}

	name[0] = '\0';
	if (state == NULL)
	{
		return;
	}

	for (const bot_console_waypoint_t *waypoint = state->patrol_points;
		waypoint != NULL;
		waypoint = waypoint->next)
	{
		size_t length = strlen(name);
		if (length + 1U >= name_size)
		{
			return;
		}

		int written = snprintf(name + length,
			name_size - length,
			"%s%s",
			length != 0U ? " to " : "",
			waypoint->name);
		if (written < 0 || (size_t)written >= name_size - length)
		{
			name[name_size - 1U] = '\0';
			return;
		}
	}
}

/*
=============
BotAI_ResolveTeamLongTermGoal

Reconstructs BotLongTermGoal's direct help, accompany, defend, CTF, camp, and
patrol branches before the ordinary item-goal stack is consulted. Remaining
LTG variants fall through to the ordinary item-goal stack.

retreat is BotLongTermGoal's third argument (sub_1001d760).  It suppresses
exactly three branches - ltgtype 1 at 0x1001d78b `if (eax == 1 && arg3 == 0)`,
ltgtype 2 at 0x1001d96d, and ltgtype 3 at 0x1001de8c `if (... || arg3 != 0)`.
Captureflag, rush base, camp and patrol all run unchanged under retreat.
=============
*/
static bot_team_goal_result_t BotAI_ResolveTeamLongTermGoal(bot_client_state_t *state,
	float thinktime,
	bot_goal_t *goal,
	vec3_t held_viewangles,
	bool *held_view_set,
	int *held_actionflags,
	int retreat)
{
	if (held_view_set != NULL)
	{
		*held_view_set = false;
	}
	if (held_actionflags != NULL)
	{
		*held_actionflags = 0;
	}
	if (state == NULL || goal == NULL)
	{
		return BOT_TEAM_GOAL_NONE;
	}

	float now = AAS_Time();
	char name[BOT_CONSOLE_MESSAGE_STORAGE_CHARS];
	switch (state->ltg_type)
	{
	case 1:
	{
		/* 0x1001d78b: `eax == 1 && arg3 == 0`. */
		if (retreat)
		{
			return BOT_TEAM_GOAL_NONE;
		}
		int teammate_entity = state->ltg_teammate + 1;
		if (state->team_message_time != 0.0f &&
			now > state->team_message_time)
		{
			BotAI_ConsoleEasyClientName(state->ltg_teammate,
				name,
				sizeof(name));
			BotAI_ConsoleEnterInitialTeamChat(state, "help_start", name);
			state->team_message_time = 0.0f;
		}
		if (now > state->team_goal_time)
		{
			state->ltg_type = 0;
		}
		if (now - 10.0f > state->teammate_visible_time)
		{
			state->ltg_type = 0;
		}

		aas_entityinfo_t teammate_info;
		memset(&teammate_info, 0, sizeof(teammate_info));
		if (teammate_entity > 0 && teammate_entity <= aasworld.maxClients &&
			aasworld.entities != NULL)
		{
			AAS_EntityInfo(teammate_entity, &teammate_info);
		}
		if (BotAI_EntityVisible(state, teammate_entity))
		{
			vec3_t direction;
			VectorSubtract(teammate_info.origin,
				state->last_client_update.origin,
				direction);
			if (sqrtf(DotProduct(direction, direction)) < 100.0f)
			{
				BotResetAvoidReachHandle(state->move_handle);
				return BOT_TEAM_GOAL_HANDLED;
			}
		}
		else
		{
			state->teammate_visible_time = now;
		}

		if (teammate_info.valid)
		{
			int area = AAS_PointAreaNum(teammate_info.origin);
			if (area != 0 && AAS_AreaReachability(area) != 0)
			{
				BotAI_ConsoleSetPointGoal(&state->team_goal,
					teammate_info.origin,
					area,
					teammate_entity);
			}
		}

		*goal = state->team_goal;
		return BOT_TEAM_GOAL_READY;
	}

	case 2:
	{
		/* 0x1001d96d: `eax == 2 && arg3 == 0`. */
		if (retreat)
		{
			return BOT_TEAM_GOAL_NONE;
		}
		int teammate_entity = state->ltg_teammate + 1;
		if (state->team_message_time != 0.0f &&
			now > state->team_message_time)
		{
			BotAI_ConsoleEasyClientName(state->ltg_teammate,
				name,
				sizeof(name));
			BotAI_ConsoleEnterInitialTeamChat(state, "accompany_start", name);
			state->team_message_time = 0.0f;
		}
		if (now > state->team_goal_time)
		{
			BotAI_ConsoleEasyClientName(state->ltg_teammate,
				name,
				sizeof(name));
			BotAI_ConsoleEnterInitialTeamChat(state, "accompany_stop", name);
			state->ltg_type = 0;
		}

		aas_entityinfo_t teammate_info;
		memset(&teammate_info, 0, sizeof(teammate_info));
		if (teammate_entity > 0 && teammate_entity <= aasworld.maxClients &&
			aasworld.entities != NULL)
		{
			AAS_EntityInfo(teammate_entity, &teammate_info);
		}
		if (BotAI_EntityVisible(state, teammate_entity))
		{
			state->teammate_visible_time = now;
			vec3_t direction;
			VectorSubtract(teammate_info.origin,
				state->last_client_update.origin,
				direction);
			if (sqrtf(DotProduct(direction, direction)) < state->formation_dist)
			{
				float crouch_time = AI_DMState_GetAttackCrouchTime(state->dm_state);
				if (crouch_time < now - 5.0f && state->character != NULL)
				{
					float croucher = Characteristic_BFloat(state->character,
						CHARACTERISTIC_CROUCHER,
						0.0f,
						1.0f);
					if (BotAI_LongTermGoalRandom() < thinktime * croucher)
					{
						crouch_time = now + 5.0f + croucher * 15.0f;
						AI_DMState_SetAttackCrouchTime(state->dm_state, crouch_time);
					}
				}
				if (AAS_Swimming(state->last_client_update.origin))
				{
					crouch_time = now - 1.0f;
					AI_DMState_SetAttackCrouchTime(state->dm_state, crouch_time);
				}

				if (state->arrive_time < now - 2.0f &&
					state->arrive_time == 0.0f)
				{
					BotAI_ConsoleEasyClientName(state->ltg_teammate,
						name,
						sizeof(name));
					EA_Gesture(state->client_number, 1);
					BotAI_ConsoleEnterInitialTeamChat(state,
						"accompany_arrive",
						name);
					state->arrive_time = now;
				}
				else if (state->arrive_time < now - 2.0f && crouch_time > now)
				{
					if (held_actionflags != NULL)
					{
						*held_actionflags |= ACTION_CROUCH;
					}
				}
				else if (state->arrive_time < now - 2.0f &&
					BotAI_LongTermGoalRandom() < thinktime * 0.3f)
				{
					int gesture = (int)(BotAI_LongTermGoalRandom() * 2.9f);
					EA_Gesture(state->client_number,
						gesture == 0 ? 0 : gesture == 1 ? 2 : 3);
				}
				if (state->arrive_time > now - 2.0f &&
					held_viewangles != NULL && held_view_set != NULL)
				{
					Vector2Angles(direction, held_viewangles);
					held_viewangles[ROLL] *= 0.5f;
					*held_view_set = true;
				}
				else if (held_viewangles != NULL && held_view_set != NULL &&
					BotAI_LongTermGoalRandom() < thinktime * 0.8f)
				{
					vec3_t roam_goal;
					BotAI_RoamGoal(state, roam_goal);
					VectorSubtract(roam_goal,
						state->last_client_update.origin,
						direction);
					Vector2Angles(direction, held_viewangles);
					held_viewangles[ROLL] *= 0.5f;
					*held_view_set = true;
				}
				BotResetAvoidReachHandle(state->move_handle);
				return BOT_TEAM_GOAL_HANDLED;
			}
		}

		if (teammate_info.valid)
		{
			int area = AAS_PointAreaNum(teammate_info.origin);
			if (area != 0 && AAS_AreaReachability(area) != 0)
			{
				BotAI_ConsoleSetPointGoal(&state->team_goal,
					teammate_info.origin,
					area,
					teammate_entity);
			}
		}

		*goal = state->team_goal;
		if (now - 60.0f > state->teammate_visible_time)
		{
			BotAI_ConsoleEasyClientName(state->ltg_teammate,
				name,
				sizeof(name));
			BotAI_ConsoleEnterInitialTeamChat(state,
				"accompany_cannotfind",
				name);
			state->ltg_type = 0;
			state->teammate_visible_time = now;
		}
		return BOT_TEAM_GOAL_READY;
	}

	case 3:
		/* 0x1001de8c: `if ((eax:1.b & 0x41) != 0 || arg3 != 0)` falls straight
		   through to the item tail. */
		if (retreat || now <= state->defend_away_time)
		{
			return BOT_TEAM_GOAL_NONE;
		}
		if (state->team_message_time != 0.0f &&
			now > state->team_message_time)
		{
			BotGoalName(state->team_goal_number, name, (int)sizeof(name));
			BotAI_ConsoleEnterInitialTeamChat(state, "defend_start", name);
			state->team_message_time = 0.0f;
		}

		*goal = state->team_goal;
		if (now > state->team_goal_time)
		{
			BotGoalName(state->team_goal_number, name, (int)sizeof(name));
			BotAI_ConsoleEnterInitialTeamChat(state, "defend_stop", name);
			state->ltg_type = 0;
		}
		if (BotAI_TeamGoalDistance(state, goal) < 70.0f)
		{
			/* Retail 0x1001df88 calls BotResetAvoidReach, not the single-slot
			   BotResetLastAvoidReach at sub_10034b20. */
			BotResetAvoidReachHandle(state->move_handle);
			state->defend_away_time = now + 5.0f +
				10.0f * BotAI_ConsoleRandom();
		}
		return BOT_TEAM_GOAL_READY;

	case 4:
	{
		bot_goal_t red_flag;
		bot_goal_t blue_flag;
		if (state->team_message_time != 0.0f &&
			now > state->team_message_time)
		{
			BotAI_ConsoleEnterInitialTeamChat(state, "captureflag_start", NULL);
			state->team_message_time = 0.0f;
		}
		if (!BotAI_ConsoleCTFFlagGoals(&red_flag, &blue_flag))
		{
			state->ltg_type = 0;
			return BOT_TEAM_GOAL_HANDLED;
		}

		*goal = BotAI_CTFTeam(state) == 1 ? blue_flag : red_flag;
		if (BotTouchingGoal(state->last_client_update.origin, goal) ||
			now > state->team_goal_time)
		{
			state->ltg_type = 0;
		}
		return BOT_TEAM_GOAL_READY;
	}

	case 5:
	{
		bot_goal_t red_flag;
		bot_goal_t blue_flag;
		if (now <= state->rush_base_away_time)
		{
			return BOT_TEAM_GOAL_NONE;
		}
		if (!BotAI_ConsoleCTFFlagGoals(&red_flag, &blue_flag))
		{
			state->ltg_type = 0;
			return BOT_TEAM_GOAL_HANDLED;
		}

		*goal = BotAI_CTFTeam(state) == 1 ? red_flag : blue_flag;
		/* Retail's only unconditional clear here is the deadline
		   (0x1001e57e -> 0x1001e580). The flag-carrying test lives inside the
		   contact branch at 0x1001e5a9, reproduced below. */
		if (now > state->team_goal_time)
		{
			state->ltg_type = 0;
		}
		if (BotTouchingGoal(state->last_client_update.origin, goal))
		{
			if (BotAI_CarryingFlag(state) == 0)
			{
				state->ltg_type = 0;
			}
			else
			{
				BotResetAvoidReachHandle(state->move_handle);
				state->rush_base_away_time = now + 5.0f +
					10.0f * BotAI_ConsoleRandom();
			}
		}
		return BOT_TEAM_GOAL_READY;
	}

	case 6:
		if (state->team_message_time != 0.0f &&
			now > state->team_message_time)
		{
			BotAI_ConsoleEasyClientName(state->ltg_teammate,
				name,
				sizeof(name));
			BotAI_ConsoleEnterInitialTeamChat(state, "camp_start", name);
			state->team_message_time = 0.0f;
		}

		*goal = state->team_goal;
		if (now > state->team_goal_time)
		{
			BotAI_ConsoleEnterInitialTeamChat(state, "camp_stop", NULL);
			state->ltg_type = 0;
		}
		if (BotAI_TeamGoalDistance(state, goal) < 40.0f)
		{
			if (state->arrive_time == 0.0f)
			{
				BotAI_ConsoleEasyClientName(state->ltg_teammate,
					name,
					sizeof(name));
				BotAI_ConsoleEnterInitialTeamChat(state, "camp_arrive", name);
				state->arrive_time = now;
			}
			if (held_viewangles != NULL && held_view_set != NULL &&
				BotAI_LongTermGoalRandom() < thinktime * 0.8f)
			{
				vec3_t roam_goal;
				vec3_t direction;
				BotAI_RoamGoal(state, roam_goal);
				VectorSubtract(roam_goal,
					state->last_client_update.origin,
					direction);
				Vector2Angles(direction, held_viewangles);
				held_viewangles[ROLL] *= 0.5f;
				*held_view_set = true;
			}

			float crouch_time = AI_DMState_GetAttackCrouchTime(state->dm_state);
			if (crouch_time < now - 5.0f && state->character != NULL)
			{
				float croucher = Characteristic_BFloat(state->character,
					CHARACTERISTIC_CROUCHER,
					0.0f,
					1.0f);
				if (BotAI_LongTermGoalRandom() < thinktime * croucher)
				{
					crouch_time = now + 5.0f + croucher * 15.0f;
					AI_DMState_SetAttackCrouchTime(state->dm_state, crouch_time);
				}
			}
			if (crouch_time > now && held_actionflags != NULL)
			{
				*held_actionflags |= ACTION_CROUCH;
			}
			if (AAS_Swimming(state->last_client_update.origin))
			{
				AI_DMState_SetAttackCrouchTime(state->dm_state, now - 1.0f);
			}
			/* Retail 0x1001e275 feeds the PointContents wrapper bs->eye
			   (arg1 + 0x6b0, origin + view_offset at 0x100289ec), not the
			   origin the AAS_Swimming test above uses. */
			vec3_t camp_eye;
			BotInterface_ClientEyePosition(state, camp_eye);
			if ((AAS_PointContents(camp_eye) &
				(CONTENTS_WATER | CONTENTS_LAVA | CONTENTS_SLIME)) != 0)
			{
				BotAI_ConsoleEnterInitialTeamChat(state, "camp_stop", NULL);
				state->ltg_type = 0;
			}
			/* Retail 0x1001e2a5 calls BotResetAvoidReach (sub_10034af0). */
			BotResetAvoidReachHandle(state->move_handle);
			return BOT_TEAM_GOAL_HANDLED;
		}
		return BOT_TEAM_GOAL_READY;

	case 7:
		if (state->team_message_time != 0.0f &&
			now > state->team_message_time)
		{
			BotAI_TeamPatrolName(state, name, sizeof(name));
			BotAI_ConsoleEnterInitialTeamChat(state, "patrol_start", name);
			state->team_message_time = 0.0f;
		}
		if (state->current_patrol_point == NULL)
		{
			state->ltg_type = 0;
			return BOT_TEAM_GOAL_HANDLED;
		}

		if (BotTouchingGoal(state->last_client_update.origin,
			&state->current_patrol_point->goal))
		{
			bot_console_waypoint_t *current = state->current_patrol_point;
			if ((state->patrol_flags & BOT_CONSOLE_PATROL_FORWARD) == 0)
			{
				if (current->prev == NULL)
				{
					state->patrol_flags |= BOT_CONSOLE_PATROL_FORWARD;
					state->current_patrol_point = current->next;
				}
				else
				{
					state->current_patrol_point = current->prev;
				}
			}
			else if (current->next == NULL)
			{
				state->patrol_flags &= ~BOT_CONSOLE_PATROL_FORWARD;
				state->current_patrol_point = current->prev;
			}
			else
			{
				state->current_patrol_point = current->next;
			}
		}

		if (now > state->team_goal_time)
		{
			BotAI_ConsoleEnterInitialTeamChat(state, "patrol_stop", NULL);
			state->ltg_type = 0;
		}
		if (state->current_patrol_point == NULL)
		{
			state->ltg_type = 0;
			return BOT_TEAM_GOAL_HANDLED;
		}

		*goal = state->current_patrol_point->goal;
		return BOT_TEAM_GOAL_READY;

	default:
		return BOT_TEAM_GOAL_NONE;
	}
}

/*
=============
BotAI_ApplyLongTermMoveResultView

Reconstructs Activate, Seek LTG, and Seek NBG's post-move ideal-view policy.
Only the Seek nodes carry the waiting-roam arm (0x1001f47d) and the mover-set
guard on the private turn (0x1001f5d7); the activate node's block has two arms
and an unconditional turn (0x1001f083, 0x1001f169).
=============
*/
static bool BotAI_ApplyLongTermMoveResultView(bot_client_state_t *state,
	const bot_goal_t *goal,
	int travel_flags,
	float thinktime,
	const bot_moveresult_t *result,
	bot_input_t *input)
{
	if (state == NULL || goal == NULL || result == NULL || input == NULL)
	{
		return false;
	}

	bool activate_node = state->ai_node == BOT_AI_NODE_ACTIVATE_ENTITY;

	/*
	 * MOVERESULT_MOVEMENTVIEWSET suppresses only this frame's
	 * BotChangeViewAngles (0x1001f5d7, 0x1001fba4, 0x10020471); the
	 * ideal-viewangles block ahead of it runs unconditionally, and the
	 * retained value is read later by Seek LTG's no-goal branch and by
	 * AINode_Stand.  sub_1001ef40 turns unconditionally at 0x1001f169.
	 */
	const bool turn = activate_node ||
		(result->flags & MOVERESULT_MOVEMENTVIEWSET) == 0;

	if ((result->flags & (MOVERESULT_MOVEMENTVIEW |
		MOVERESULT_SWIMVIEW)) != 0)
	{
		AI_DMState_SetIdealViewAngles(state->dm_state,
			result->ideal_viewangles);
		return turn;
	}

	vec3_t viewangles;
	if (!activate_node && (result->flags & MOVERESULT_WAITING) != 0)
	{
		/* 0x1001f4aa / 0x1001fad2: the random gate skips the ideal write
		   only, not the turn decision. */
		if (BotAI_LongTermGoalRandom() < thinktime * 0.8f)
		{
			vec3_t roam_goal;
			vec3_t direction;
			BotAI_RoamGoal(state, roam_goal);
			VectorSubtract(roam_goal,
				state->last_client_update.origin,
				direction);
			Vector2Angles(direction, viewangles);
			viewangles[ROLL] *= 0.5f;
			AI_DMState_SetIdealViewAngles(state->dm_state, viewangles);
		}
		return turn;
	}

	/*
	 * 0x1001f504 loads the Seek NBG view goal with BotGetSecondGoal; when that
	 * returns nothing the following BotGetTopGoal result is never moved into
	 * EDI, so a single-entry stack passes a NULL goal and
	 * BotMovementViewTarget's NULL guard forces the movedir fallback.  Seek
	 * LTG (0x1001fb3d) and the activate node (0x1001f0c1) pass their own goal.
	 */
	const bot_goal_t *view_goal = goal;
	bot_goal_t second_goal;
	if (state->ai_node == BOT_AI_NODE_SEEK_NBG)
	{
		view_goal = (state->goal_handle > 0 &&
			AI_GoalBotlib_GetSecondGoal(state->goal_handle, &second_goal) != 0)
			? &second_goal
			: NULL;
	}

	vec3_t target;
	vec3_t direction;
	if (BotMovementViewTargetHandle(state->move_handle,
		view_goal,
		travel_flags,
		300.0f,
		target) != 0)
	{
		VectorSubtract(target, state->last_client_update.origin, direction);
	}
	else
	{
		VectorCopy(result->movedir, direction);
	}
	Vector2Angles(direction, viewangles);
	viewangles[ROLL] *= 0.5f;
	AI_DMState_SetIdealViewAngles(state->dm_state, viewangles);
	return turn;
}

/*
=============
BotAI_BlockedBspModelBounds

Maps the game-facing inline model number on a blocked BSP entity back to the
zero-based AAS model bounds used by Gladiator's static-entity logic.
=============
*/
static bool BotAI_BlockedBspModelBounds(const char *model,
	bool allow_inline_fallback,
	vec3_t mins,
	vec3_t maxs)
{
	if (model == NULL || mins == NULL || maxs == NULL)
	{
		return false;
	}

	int modelindex = IndexFromModel(model);
	if (modelindex == 0 && allow_inline_fallback && model[0] == '*')
	{
		modelindex = (int)strtol(model + 1, NULL, 10);
	}
	if (modelindex <= 0 || aasworld.bspModels == NULL ||
		modelindex > aasworld.numBspModels)
	{
		return false;
	}

	vec3_t zero_angles = {0.0f, 0.0f, 0.0f};
	AAS_BSPModelMinsMaxsOrigin(modelindex - 1,
		zero_angles,
		mins,
		maxs,
		NULL);
	return true;
}

/*
=============
BotAI_SetBlockedAttack

Issues the retail blocked-door and shoot-button Blaster action while storing
the ideal movement view in the result.
=============
*/
static void BotAI_SetBlockedAttack(bot_client_state_t *state,
	bot_moveresult_t *result,
	const vec3_t target)
{
	if (state == NULL || result == NULL || target == NULL)
	{
		return;
	}

	vec3_t direction;
	VectorSubtract(target, state->last_client_update.origin, direction);
	Vector2Angles(direction, result->ideal_viewangles);
	result->ideal_viewangles[ROLL] *= 0.5f;
	result->flags |= MOVERESULT_MOVEMENTVIEW;
	EA_UseItem(state->client_number, "Blaster");
	EA_Attack(state->client_number);
}

/*
=============
BotAI_EntityNumberForBspEntity

Resolves a parsed BSP activator back to the live AAS entity slot. Inline BSP
model names are zero-based (`*n`), while AAS stores the corresponding model
index one-based. Returning zero lets callers retain the blocked entity as a
safe fallback when a map omits a live activator model.
=============
*/
static int BotAI_EntityNumberForBspEntity(const aas_bspentity_t *entity)
{
	if (entity == NULL)
	{
		return 0;
	}

	const char *model = AAS_ValueForBSPEpairKey(entity, "model");
	if (model == NULL || model[0] == '\0')
	{
		return 0;
	}

	int modelindex = IndexFromModel(model);
	if (modelindex <= 0 && model[0] == '*')
	{
		/* AAS entity modelindex is one-based while inline model names are
		 * written as the zero-based BSP model number. */
		char *end = NULL;
		long model_number = strtol(model + 1, &end, 10);
		if (end != model + 1 && *end == '\0' &&
			model_number >= 0 && model_number < INT_MAX)
		{
			modelindex = (int)model_number + 1;
		}
	}
	if (modelindex <= 0)
	{
		return 0;
	}

	for (int entnum = AAS_NextEntity(0);
		entnum != 0;
		entnum = AAS_NextEntity(entnum))
	{
		if (AAS_EntityModelindex(entnum) == modelindex)
		{
			return entnum;
		}
	}
	return 0;
}

/*
=============
BotAI_StoreBlockedActivationGoal

Stores Gladiator's single ten-second activation goal after finding a reachable
ground point beside the blocking static BSP entity. A start-solid button uses
its untraced face point; a start-solid trigger consumes the block without
installing a goal.
=============
*/
static bool BotAI_StoreBlockedActivationGoal(bot_client_state_t *state,
	const bot_moveresult_t *result,
	const aas_bspentity_t *activation_entity,
	const vec3_t origin,
	const vec3_t mins,
	const vec3_t maxs,
	const vec3_t fallback_point,
	const vec3_t trace_start,
	const vec3_t trace_end,
	bool store_when_startsolid)
{
	if (state == NULL || result == NULL || origin == NULL || mins == NULL ||
		maxs == NULL || fallback_point == NULL || trace_start == NULL ||
		trace_end == NULL)
	{
		return false;
	}

	aas_trace_t trace = AAS_TraceClientBBox(trace_start,
		trace_end,
		PRESENCE_CROUCH,
		-1);
	if (trace.startsolid && !store_when_startsolid)
	{
		return true;
	}

	const float *goal_point = trace.startsolid ? fallback_point : trace.endpos;
	int areanum = AAS_PointAreaNum(goal_point);

	memset(&state->activation_goal, 0, sizeof(state->activation_goal));
	VectorCopy(origin, state->activation_goal.origin);
	state->activation_goal.areanum = areanum;
	VectorCopy(mins, state->activation_goal.mins);
	VectorCopy(maxs, state->activation_goal.maxs);
	int activation_entity_number = BotAI_EntityNumberForBspEntity(
		activation_entity);
	state->activation_goal.entitynum = activation_entity_number > 0 ?
		activation_entity_number : result->blockentity;
	state->activation_goal_time = AAS_Time() + 10.0f;
	if (AAS_AreaReachability(areanum) == 0)
	{
		if (state->ai_node == BOT_AI_NODE_SEEK_NBG)
		{
			state->nearby_goal_time = 0.0f;
		}
		else if (state->ai_node == BOT_AI_NODE_SEEK_LTG)
		{
			state->long_term_goal_time = 0.0f;
		}
		return true;
	}

	if (BotAI_CoopObjectiveControlEnabled())
	{
		/* This is the HTN's NavigateToButton step.  The blocker itself is
		 * the authorization to pursue the control; no global button search is
		 * started here. */
		BotAI_CoopSetControlObjectivePhase(state,
			BOT_COOP_CONTROL_NAVIGATE,
			state->activation_goal.entitynum,
			areanum);
	}
	BotAI_EnterNode(state, BOT_AI_NODE_ACTIVATE_ENTITY);
	return true;
}

/*
=============
BotAI_ButtonMoveDirection

Recreates Quake II brush-button angle decoding for the face point used by the
retail blocked-entity response.
=============
*/
static void BotAI_ButtonMoveDirection(const aas_bspentity_t *entity,
	vec3_t direction)
{
	if (direction == NULL)
	{
		return;
	}

	float angle = AAS_FloatForBSPEpairKey(entity, "angle");
	if (angle == -1.0f)
	{
		VectorSet(direction, 0.0f, 0.0f, 1.0f);
		return;
	}
	if (angle == -2.0f)
	{
		VectorSet(direction, 0.0f, 0.0f, -1.0f);
		return;
	}

	float radians = angle * ((float)M_PI / 180.0f);
	VectorSet(direction, cosf(radians), sinf(radians), 0.0f);
}

/*
=============
BotAI_HandleBlockedStaticEntity

Handles the static brush classes recognized by retail BotAIBlocked: doors are
shot immediately, shootable buttons are aimed at their exposed face, and
buttons/triggers with a reachable contact point enter the dedicated activation
node.
=============
*/
static bool BotAI_HandleBlockedStaticEntity(bot_client_state_t *state,
	bot_moveresult_t *result,
	const aas_bspentity_t *entity)
{
	if (state == NULL || result == NULL || entity == NULL)
	{
		return false;
	}

	const char *model = AAS_ValueForBSPEpairKey(entity, "model");
	if (model == NULL)
	{
		return false;
	}
	const char *classname = AAS_ValueForBSPEpairKey(entity, "classname");
	if (classname == NULL)
	{
		return false;
	}

	bool is_door = strcmp(classname, "func_door") == 0 ||
		strcmp(classname, "func_door_secret") == 0 ||
		strcmp(classname, "func_door_rotating") == 0;
	bool is_button = strcmp(classname, "func_button") == 0;
	bool is_trigger = strcmp(classname, "trigger_multiple") == 0 ||
		strcmp(classname, "trigger_once") == 0;
	if (!is_door && !is_button && !is_trigger)
	{
		return false;
	}

	vec3_t model_mins;
	vec3_t model_maxs;
	if (!BotAI_BlockedBspModelBounds(model,
		is_trigger,
		model_mins,
		model_maxs))
	{
		return true;
	}

	vec3_t center;
	VectorAdd(model_mins, model_maxs, center);
	VectorScale(center, 0.5f, center);

	if (is_door)
	{
		BotAI_SetBlockedAttack(state, result, center);
		return true;
	}

	vec3_t goal_mins;
	vec3_t goal_maxs;
	for (int axis = 0; axis < 3; ++axis)
	{
		goal_mins[axis] = model_mins[axis] - center[axis];
		goal_maxs[axis] = model_maxs[axis] - center[axis];
	}

	if (is_button)
	{
		vec3_t move_direction;
		BotAI_ButtonMoveDirection(entity, move_direction);
		(void)AAS_FloatForBSPEpairKey(entity, "lip");
		float half_extent = 0.5f *
			(fabsf(move_direction[0]) * (model_maxs[0] - model_mins[0]) +
			fabsf(move_direction[1]) * (model_maxs[1] - model_mins[1]) +
			fabsf(move_direction[2]) * (model_maxs[2] - model_mins[2]));
		if (AAS_FloatForBSPEpairKey(entity, "health") != 0.0f)
		{
			vec3_t face;
			VectorMA(center, -half_extent, move_direction, face);
			BotAI_SetBlockedAttack(state, result, face);
			return true;
		}

		vec3_t trace_start;
		vec3_t trace_end;
		vec3_t client_mins;
		vec3_t client_maxs;
		AAS_PresenceTypeBoundingBox(PRESENCE_CROUCH,
			client_mins,
			client_maxs);
		float client_extent = 0.0f;
		for (int axis = 0; axis < 3; ++axis)
		{
			float bound = move_direction[axis] < 0.0f ?
				client_maxs[axis] : client_mins[axis];
			client_extent += fabsf(bound) * fabsf(move_direction[axis]);
		}
		VectorMA(center,
			-(half_extent + client_extent),
			move_direction,
			trace_start);
		vec3_t fallback_point;
		VectorCopy(trace_start, fallback_point);
		trace_start[2] += 24.0f;
		VectorCopy(trace_start, trace_end);
		trace_end[2] -= 100.0f;
		for (int axis = 0; axis < 3; ++axis)
		{
			goal_mins[axis] -= 5.0f;
			goal_maxs[axis] += 5.0f;
		}
		return BotAI_StoreBlockedActivationGoal(state,
			result,
			entity,
			center,
			goal_mins,
			goal_maxs,
			fallback_point,
			trace_start,
			trace_end,
			true);
	}

	if (is_trigger)
	{
		vec3_t trace_start;
		vec3_t trace_end;
		VectorCopy(center, trace_start);
		trace_start[2] = model_maxs[2] + 24.0f;
		VectorCopy(trace_start, trace_end);
		trace_end[2] -= 100.0f;
		return BotAI_StoreBlockedActivationGoal(state,
			result,
			entity,
			center,
			goal_mins,
			goal_maxs,
			center,
			trace_start,
			trace_end,
			false);
	}

	return false;
}

/*
=============
BotAI_FindBspEntityByModel

Finds the map entity paired with the blocked game's inline BSP model, using
the same literal `*n` model name that retail BotEntityToActivate constructs.
=============
*/
static const aas_bspentity_t *BotAI_FindBspEntityByModel(
	const aas_bspentity_t *entities,
	const char *model)
{
	if (model == NULL)
	{
		return NULL;
	}

	for (const aas_bspentity_t *entity = entities;
		entity != NULL;
		entity = entity->next)
	{
		const char *entity_model = AAS_ValueForBSPEpairKey(entity, "model");
		if (entity_model != NULL && strcmp(entity_model, model) == 0)
		{
			return entity;
		}
	}

	return NULL;
}

/*
=============
BotAI_FindBspEntityByTarget

Finds the next map entity whose `target` points at a saved targetname. The
caller retains the successor cursor so it can reproduce retail's backtracking
through maps with multiple matching target links.
=============
*/
static const aas_bspentity_t *BotAI_FindBspEntityByTarget(
	const aas_bspentity_t *entity,
	const char *target)
{
	if (target == NULL)
	{
		return NULL;
	}

	for (; entity != NULL; entity = entity->next)
	{
		const char *entity_target = AAS_ValueForBSPEpairKey(entity, "target");
		if (entity_target != NULL && strcmp(entity_target, target) == 0)
		{
			return entity;
		}
	}

	return NULL;
}

/*
=============
BotAI_EntityToActivate

Reconstructs retail BotEntityToActivate: beginning at a blocked inline model,
walk reverse target links through trigger_counter and trigger_relay entities
until a usable button/trigger is found. It preserves the retail depth limit,
diagnostics, and trigger_key rejection.
=============
*/
const aas_bspentity_t *BotAI_EntityToActivate(
	const aas_bspentity_t *entities,
	int blockentity)
{
	aas_entityinfo_t entity_info;
	memset(&entity_info, 0, sizeof(entity_info));
	AAS_EntityInfo(blockentity, &entity_info);
	const char *model = AAS_ModelFromIndex(entity_info.modelindex);
	if (model == NULL || model[0] == '\0')
	{
		return NULL;
	}

	const aas_bspentity_t *entity = BotAI_FindBspEntityByModel(entities, model);
	if (entity == NULL)
	{
		BotLib_Print(PRT_ERROR,
			"BotEntityToActivate: no entity found with model %s\n",
			model);
		return NULL;
	}

	const char *classname = AAS_ValueForBSPEpairKey(entity, "classname");
	if (classname == NULL)
	{
		BotLib_Print(PRT_ERROR,
			"BotEntityToActivate: entity with model %s has no classname\n",
			model);
		return NULL;
	}

	const char *targetname = AAS_ValueForBSPEpairKey(entity, "targetname");
	if (strcmp(classname, "func_door_secret") == 0 &&
		(targetname == NULL ||
			(AAS_IntForBSPEpairKey(entity, "spawnflags") & 1) != 0))
	{
		return entity;
	}
	if (strcmp(classname, "func_door") == 0 &&
		AAS_FloatForBSPEpairKey(entity, "health") != 0.0f)
	{
		return entity;
	}
	if (targetname == NULL)
	{
		return NULL;
	}

	const char *target_stack[10] = {targetname};
	const aas_bspentity_t *search_stack[10] = {entities};
	int depth = 0;
	const char *last_classname = classname;
	while (depth >= 0)
	{
		const aas_bspentity_t *match = BotAI_FindBspEntityByTarget(
			search_stack[depth], target_stack[depth]);
		if (match == NULL)
		{
			if (target_stack[depth] != NULL)
			{
				BotLib_Print(PRT_ERROR,
					"BotEntityToActivate: no entity with target \"%s\"\n",
					target_stack[depth]);
			}
			--depth;
			continue;
		}
		search_stack[depth] = match->next;

		classname = AAS_ValueForBSPEpairKey(match, "classname");
		if (classname == NULL)
		{
			BotLib_Print(PRT_ERROR,
				"BotEntityToActivate: entity with target \"%s\" has no classname\n",
				target_stack[depth] != NULL ? target_stack[depth] : "");
			return NULL;
		}
		last_classname = classname;

		if (strcmp(classname, "trigger_counter") == 0 ||
			strcmp(classname, "trigger_relay") == 0)
		{
			if (depth >= 9)
			{
				BotLib_Print(PRT_ERROR,
					"BotEntityToActivate: stacked up more than %d trigger_counter or trigger_relay\n",
					depth);
				return NULL;
			}
			++depth;
			target_stack[depth] = AAS_ValueForBSPEpairKey(match, "targetname");
			search_stack[depth] = entities;
			continue;
		}

		if (strcmp(classname, "func_button") == 0 ||
			strcmp(classname, "trigger_multiple") == 0 ||
			strcmp(classname, "trigger_once") == 0 ||
			strcmp(classname, "func_door_rotating") == 0)
		{
			return match;
		}
		if (strcmp(classname, "trigger_key") == 0)
		{
			return NULL;
		}
		--depth;
	}

	BotLib_Print(PRT_ERROR,
		"BotEntityToActivate: unkown activator with classname \"%s\"\n",
		last_classname != NULL ? last_classname : "");
	return NULL;
}

/*
=============
BotAI_HandleBlockedMovement

Implements retail BotAIBlocked's direct static-activator handling and its
perpendicular alternate movement when no recognized activator applies. The
successful BotMoveInDirection call remains authoritative in the EA record.
=============
*/
bool BotAI_HandleBlockedMovement(bot_client_state_t *state,
	bot_moveresult_t *result,
	bool allow_activation,
	vec3_t alternate_direction)
{
	if (alternate_direction != NULL)
	{
		VectorClear(alternate_direction);
	}
	if (state == NULL || result == NULL || result->blocked == 0)
	{
		return false;
	}

	if (allow_activation && result->blockentity > 0)
	{
		aas_entityinfo_t entity_info;
		memset(&entity_info, 0, sizeof(entity_info));
		AAS_EntityInfo(result->blockentity, &entity_info);
		if (entity_info.valid && entity_info.solid == SOLID_BSP &&
			entity_info.modelindex > 0)
		{
			aas_bspentity_t *entities = BotInterface_CoopMapEntities();
			if (entities != NULL)
			{
				const aas_bspentity_t *entity = BotAI_EntityToActivate(
					entities, result->blockentity);
				if (entity != NULL && BotAI_HandleBlockedStaticEntity(state,
					result,
					entity))
				{
					return false;
				}
			}
		}
	}

	if (alternate_direction == NULL)
	{
		return false;
	}

	vec3_t forward;
	VectorCopy(result->movedir, forward);
	forward[2] = 0.0f;
	float forward_length = sqrtf(DotProduct(forward, forward));
	if (forward_length != 0.0f)
	{
		VectorScale(forward, 1.0f / forward_length, forward);
	}

	VectorSet(alternate_direction, forward[1], -forward[0], 0.0f);
	if (state->blocked_avoid_right)
	{
		VectorScale(alternate_direction, -1.0f, alternate_direction);
	}
	bool moved = BotMoveInDirectionHandle(state->move_handle,
		alternate_direction,
		400.0f,
		MOVE_WALK) != 0;
	if (!moved)
	{
		VectorScale(alternate_direction, -1.0f, alternate_direction);
		state->blocked_avoid_right = !state->blocked_avoid_right;
		moved = BotMoveInDirectionHandle(state->move_handle,
			alternate_direction,
			400.0f,
			MOVE_WALK) != 0;
	}

	if (state->ai_node == BOT_AI_NODE_SEEK_NBG)
	{
		state->nearby_goal_time = 0.0f;
	}
	else if (state->ai_node == BOT_AI_NODE_SEEK_LTG)
	{
		state->long_term_goal_time = 0.0f;
	}
	return moved;
}

/*
=============
BotAI_HandleFightMoveResult

Consume BotAttackMove's dormant chase result before Battle Fight aims or fires,
matching the retail failure-reset and non-activating blocked-response order.
=============
*/
static void BotAI_HandleFightMoveResult(void *context,
	const struct bot_moveresult_s *move_result,
	bool attack_chase_active)
{
	bot_client_state_t *state = (bot_client_state_t *)context;
	if (state == NULL || move_result == NULL || !attack_chase_active)
	{
		return;
	}

	bot_moveresult_t result = *move_result;
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
	state->last_move_result = result;
	state->has_move_result = true;
}

/*
=============
BotAI_RunTeamGoalMovement

Runs direct goal movement through the shared retail result bridge, including
the node-specific failed-move avoidance and goal-clock resets.
=============
*/
static int BotAI_RunTeamGoalMovement(bot_client_state_t *state,
	float thinktime,
	const bot_goal_t *goal,
	bot_input_t *input,
	bool defer_view_turn,
	bool *view_turn_pending)
{
	if (view_turn_pending != NULL)
	{
		*view_turn_pending = false;
	}
	if (state == NULL || goal == NULL || input == NULL)
	{
		return BLERR_INVALIDIMPORT;
	}

	int status = BotInterface_PrepareMoveState(state, thinktime);
	if (status != BLERR_NOERROR)
	{
		return status;
	}

	/*
	 * Retail hands the node's own var_7c mask to both BotMoveToGoal and
	 * BotMovementViewTarget. The activate node builds it without the
	 * rocket-jump term (0x1001efad), Seek NBG and Seek LTG with it
	 * (0x1001f327, 0x1001f83e).
	 */
	int travel_flags = state->ai_node == BOT_AI_NODE_ACTIVATE_ENTITY ?
		BotAI_ActivateEntityTravelFlags() :
		BotAI_LongTermGoalTravelFlags(state);

	bot_moveresult_t result;
	BotClearMoveResult(&result);
	BotMoveToGoalHandle(&result,
		state->move_handle,
		goal,
		travel_flags);
	if (result.failure)
	{
		BotResetAvoidReachHandle(state->move_handle);
		if (state->ai_node == BOT_AI_NODE_ACTIVATE_ENTITY ||
			state->ai_node == BOT_AI_NODE_SEEK_NBG)
		{
			state->nearby_goal_time = 0.0f;
		}
		else
		{
			state->long_term_goal_time = 0.0f;
		}
	}
	vec3_t alternate_direction;
	BotAI_HandleBlockedMovement(state,
		&result,
		true,
		alternate_direction);
	if (state->ai_node == BOT_AI_NODE_ACTIVATE_ENTITY && result.failure &&
		state->coop_control_phase == BOT_COOP_CONTROL_NAVIGATE)
	{
		BotAI_CoopSetControlObjectivePhase(state,
			BOT_COOP_CONTROL_RETRY,
			state->coop_control_entity,
			state->coop_control_goal_area);
	}

	status = EA_GetInput(state->client_number, thinktime, input);
	if (status != BLERR_NOERROR)
	{
		return status;
	}
	BotInterface_ApplyMoveResult(&result, input);
	bool advance_view = BotAI_ApplyLongTermMoveResultView(state,
		goal,
		travel_flags,
		thinktime,
		&result,
		input);
	state->last_move_result = result;
	state->has_move_result = true;
	state->active_goal_number = goal->number;
	status = EA_SubmitInput(state->client_number, input);
	if (status != BLERR_NOERROR)
	{
		return status;
	}
	if (advance_view)
	{
		if (defer_view_turn)
		{
			if (view_turn_pending != NULL)
			{
				*view_turn_pending = true;
			}
		}
		else
		{
			AI_DMState_SetEnemyContext(state->dm_state,
				state->combat.current_enemy,
				state->combat.enemy_sight_time,
				state->combat.last_enemy_area,
				state->combat.last_enemy_origin);
			AI_DMState_ChangeViewAngles(state->dm_state, state, thinktime);
		}
	}
	return BLERR_NOERROR;
}

/*
=============
BotAI_RunLongTermIdleView

Handles Seek LTG's no-goal and completed direct-team branches. Retail submits
their stationary actions, preserves a direct branch's ideal look target, then
advances sub_10029150's private view turn.
=============
*/
static int BotAI_RunLongTermIdleView(bot_client_state_t *state,
	float thinktime,
	const vec3_t held_viewangles,
	bool held_view_set,
	int held_actionflags,
	bot_input_t *input)
{
	if (state == NULL || input == NULL)
	{
		return BLERR_INVALIDIMPORT;
	}

	memset(input, 0, sizeof(*input));
	input->thinktime = thinktime;
	VectorCopy(state->last_client_update.viewangles, input->viewangles);
	input->actionflags = held_actionflags;
	int status = EA_SubmitInput(state->client_number, input);
	if (status != BLERR_NOERROR)
	{
		return status;
	}

	if (held_view_set && held_viewangles != NULL)
	{
		AI_DMState_SetIdealViewAngles(state->dm_state, held_viewangles);
	}
	AI_DMState_SetEnemyContext(state->dm_state,
		state->combat.current_enemy,
		state->combat.enemy_sight_time,
		state->combat.last_enemy_area,
		state->combat.last_enemy_origin);
	AI_DMState_ChangeViewAngles(state->dm_state, state, thinktime);
	return BLERR_NOERROR;
}

/*
=============
BotAI_ConsumeDeferredNodeSwitch

Accounts for the retail scheduler dispatch consumed when Seek LTG hands a
selected nearby goal to Seek NBG after the reconstructed terminal-work split.
=============
*/
static bool BotAI_ConsumeDeferredNodeSwitch(bot_client_state_t *state)
{
	if (state == NULL)
	{
		return false;
	}

	state->ai_node_switches++;
	if (state->ai_node_switches < BOT_AI_MAX_NODE_SWITCHES)
	{
		return true;
	}

	state->ai_node_overflow = true;
	/* Same retail overflow sequence as BotAI_Think: the two goal dumps at
	   0x10028ba7/0x10028bad then BotDumpNodeSwitches' level-4 ClientName
	   line at 0x1001d302/0x1001d358. */
	BotDumpGoalStack(state->goal_handle);
	BotDumpAvoidGoals(state->goal_handle);
	BotInterface_Printf(PRT_FATAL,
		"%s at %1.1f switched more than %d AI nodes\n",
		BotState_ClientName(state->client_number),
		AAS_Time(),
		BOT_AI_MAX_NODE_SWITCHES);
	return false;
}

/*
=============
BotAI_RunGoalMovement

Runs the retail terminal goal-node movement path for one bot AI frame.
=============
*/
static int BotAI_RunGoalMovement(bot_client_state_t *state,
	float thinktime,
	ai_goal_selection_t *selection,
	bot_input_t *input,
	bool *view_turn_pending,
	bool *post_acquire_enemy)
{
	(void)selection;
	if (view_turn_pending != NULL)
	{
		*view_turn_pending = false;
	}

	if (state->ai_node == BOT_AI_NODE_ACTIVATE_ENTITY)
	{
		if (post_acquire_enemy != NULL)
		{
			*post_acquire_enemy = true;
		}
		BotAI_UseItems(state);
		return BotAI_RunTeamGoalMovement(state,
			thinktime,
			&state->activation_goal,
			input,
			true,
			view_turn_pending);
	}
	if (state->ai_node == BOT_AI_NODE_SEEK_NBG)
	{
		if (post_acquire_enemy != NULL)
		{
			*post_acquire_enemy = true;
		}
		bot_goal_t nearby_goal;
		if (state->goal_handle <= 0 ||
			AI_GoalBotlib_GetTopGoal(state->goal_handle, &nearby_goal) == 0)
		{
			return BLERR_INVALIDIMPORT;
		}

		BotAI_UseItems(state);
		int status = BotAI_RunTeamGoalMovement(state,
			thinktime,
			&nearby_goal,
			input,
			true,
			view_turn_pending);
		if (state->has_move_result && state->last_move_result.failure)
		{
			state->nearby_goal_time = 0.0f;
		}
		return status;
	}

	BotAI_SelectAutomaticCTFGoal(state);

	bot_goal_t team_goal;
	vec3_t held_viewangles;
	bool held_view_set = false;
	int held_actionflags = 0;
	bot_team_goal_result_t team_result = BotAI_ResolveTeamLongTermGoal(state,
		thinktime,
		&team_goal,
		held_viewangles,
		&held_view_set,
		&held_actionflags,
		0);
	if (team_result == BOT_TEAM_GOAL_READY)
	{
		if (BotAI_TryLongTermNearbyGoal(state,
			&team_goal,
			BotAI_LongTermGoalTravelFlags(state)))
		{
			/* Retail re-enters Seek NBG and moves the selected item this frame. */
			if (!BotAI_ConsumeDeferredNodeSwitch(state))
			{
				return BLERR_NOERROR;
			}
			return BotAI_RunGoalMovement(state,
				thinktime,
				selection,
				input,
				view_turn_pending,
				post_acquire_enemy);
		}
		BotAI_UseItems(state);
		return BotAI_RunTeamGoalMovement(state,
			thinktime,
			&team_goal,
			input,
			false,
			NULL);
	}
	if (team_result == BOT_TEAM_GOAL_HANDLED)
	{
		return BotAI_RunLongTermIdleView(state,
			thinktime,
			held_viewangles,
			held_view_set,
			held_actionflags,
			input);
	}

	bot_goal_t long_term_goal;
	if (BotAI_GetItemLongTermGoal(state,
		&long_term_goal,
		BotAI_LongTermGoalTravelFlags(state)))
	{
		if (BotAI_TryLongTermNearbyGoal(state,
			&long_term_goal,
			BotAI_LongTermGoalTravelFlags(state)))
		{
			/* Retail re-enters Seek NBG and moves the selected item this frame. */
			if (!BotAI_ConsumeDeferredNodeSwitch(state))
			{
				return BLERR_NOERROR;
			}
			return BotAI_RunGoalMovement(state,
				thinktime,
				selection,
				input,
				view_turn_pending,
				post_acquire_enemy);
		}
		BotAI_UseItems(state);
		return BotAI_RunTeamGoalMovement(state,
			thinktime,
			&long_term_goal,
			input,
			false,
			NULL);
	}

	return BotAI_RunLongTermIdleView(state,
		thinktime,
		NULL,
		false,
		0,
		input);
}


/*
=============
BotAI_CoopResultUsesElevator

BotMoveToGoal copies the reach travel type before calling BotTravel_Elevator,
but the returned elementary mover result can clear that field. Preserve the
semantic elevator check through the result type and retained reach as well so
coop wait/timeout/retry logic observes the actual mover route.
=============
*/
/*
=============
BotAI_CoopElevatorRoute

Return the concrete reachability selected for a coop regroup. This is
diagnostic evidence only; movement remains owned by BotMoveToGoalHandle.
Keeping the source and destination AAS areas in the event makes a real
two-client stuck episode distinguishable from a route-selection failure.
=============
*/
bool BotAI_CoopObjectiveControlEnabled(void)
{
	return BotAI_CoopMode() != 0 &&
		LibVarGetValue("coopbot_objective_htn") != 0.0f;
}

void BotAI_CoopSetControlObjectivePhase(
	bot_client_state_t *state,
	bot_coop_control_phase_t phase,
	int control_entity,
	int goal_area)
{
	BotInterface_CoopSetControlObjectivePhase(state,
		BotAI_CoopObjectiveControlEnabled(),
		phase,
		control_entity,
		goal_area);
}

/*
=============
BotAI_CoopProbeControlRoute

Probe the current AAS route from the companion to the human after a control
activation. This is advisory evidence for the Objective HTN: a transiently
unknown player area (for example while riding a lift) is not treated as a
failed activation, and the existing bounded wait remains authoritative.
=============
*/
/*
=============
BotAI_CoopEntityLeadsToChangelevel

Follow the static target graph used by Quake II triggers.  The bounded depth
keeps malformed or cyclic map links from turning a per-frame safety check into
an unbounded traversal.
=============
*/
/*
=============
BotAI_ApplyCoopHardLeash

The companion leash is an opt-in coop overlay. When the bot exceeds the hard
leash, it stops the current attack command and uses the same AAS movement path
as ordinary goal navigation to return to the human player. The retail/DM path
is untouched unless coopbot_leash is enabled.
=============
*/
static int BotAI_Think(bot_client_state_t *state, float thinktime)
{
	if (state == NULL)
	{
		return BLERR_AIUPDATEINACTIVECLIENT;
	}

	if (!state->client_update_valid)
	{
		BotInterface_Printf(PRT_WARNING,
			"[bot_interface] BotAI: no snapshot for client %d\n",
			state->client_number);
		return BLERR_AIUPDATEINACTIVECLIENT;
	}

	if (state->goal_state == NULL || state->move_handle <= 0)
	{
		return BLERR_INVALIDIMPORT;
	}

	/* Feed area and intent observations before node selection can acquire a target. */
	BotAI_UpdateCoopBotArea(state);
	BotAI_UpdateCoopPlayerIntent(state);
	BotAI_UpdateCoopPlayerStyle(state);
	BotAI_UpdateCoopJointRetreat(state);
	BotAI_UpdateCoopRole(state);

	/*
	 * Retail BotDeathmatchAI runs the inventory pass (0x10028b15), the console
	 * message pass (0x10028b1b) and the enter-game chat probe (0x10028b4c)
	 * before it ever looks at the current node.  The observer, intermission
	 * and dead tests live inside the nodes, so a dead, spectating or
	 * intermission bot still drains its console queue every frame.
	 */
	BotAI_UpdateBattleInventory(state);
	BotCheckConsoleMessages(state);
	if (state->enter_game_time > AAS_Time() - 8.0f)
	{
		if (BotAI_ConstructLifecycleChat(state,
			"enter_game",
			CHARACTERISTIC_CHAT_ENTEREXITGAME,
			true))
		{
			BotAI_SetLifecycleStand(state, BotAI_ChatTime(state));
		}
	}

	pmtype_t pm_type = state->last_client_update.pm_type;
	if (pm_type == PM_SPECTATOR)
	{
		if (state->ai_node != BOT_AI_NODE_OBSERVER)
		{
			BotAI_EnterObserver(state);
		}
		return BotAI_RunLifecycleFrame(state, thinktime, false);
	}
	if (pm_type == PM_FREEZE)
	{
		if (state->ai_node != BOT_AI_NODE_INTERMISSION)
		{
			BotAI_EnterIntermission(state);
		}
		return BotAI_RunLifecycleFrame(state, thinktime, false);
	}
	if (pm_type == PM_DEAD || pm_type == PM_GIB)
	{
		if (!state->respawn_requested)
		{
			BotAI_ResetRespawnState(state);
			state->respawn_requested = true;
			state->respawn_action_sent = false;
			state->respawn_time = AAS_Time();
			if (BotAI_ConstructDeathChat(state))
			{
				state->respawn_time += BotAI_ChatTime(state);
			}
		}

		bool request_respawn = !state->respawn_action_sent &&
			AAS_Time() > state->respawn_time;
		return BotAI_RunLifecycleFrame(state, thinktime, request_respawn);
	}
	if (state->respawn_requested)
	{
		state->respawn_requested = false;
		state->respawn_action_sent = false;
		state->respawn_time = 0.0f;
		BotAI_EnterNode(state, BOT_AI_NODE_SEEK_LTG);
		/* Retail's respawn node changes state, then ends this first alive frame. */
		return BotAI_RunLifecycleFrame(state, thinktime, false);
	}
	if (state->ai_node == BOT_AI_NODE_OBSERVER)
	{
		BotAI_SetLifecycleStand(state, 0.0f);
		return BotAI_RunLifecycleFrame(state, thinktime, false);
	}
	if (state->ai_node == BOT_AI_NODE_INTERMISSION)
	{
		bool started_chat = BotAI_ConstructLifecycleChat(state,
			"start_level",
			CHARACTERISTIC_CHAT_STARTENDLEVEL,
			false);
		BotAI_SetLifecycleStand(state,
			started_chat ? BotAI_ChatTime(state) : 2.0f);
		return BotAI_RunLifecycleFrame(state, thinktime, false);
	}

	bot_ai_node_frame_t frame;
	memset(&frame, 0, sizeof(frame));
	frame.thinktime = thinktime;
	BotAI_InitEnemyInfo(&frame.enemy);
	if (!BotAI_RunNodeSwitchLoop(state, BotAI_NodeStep, &frame))
	{
		/*
		 * Retail emits three diagnostics on overflow: BotDumpGoalStack
		 * (0x10028ba7), BotDumpAvoidGoals (0x10028bad) and BotDumpNodeSwitches
		 * (0x10028bb3), the last of which formats with ClientName and prints
		 * at level 4 (0x1001d358).
		 */
		BotDumpGoalStack(state->goal_handle);
		BotDumpAvoidGoals(state->goal_handle);
		BotAI_DumpNodeSwitches(state);
	}

	if (frame.work == BOT_AI_FRAME_WORK_STAND)
	{
		bot_input_t coop_input;
		int stand_status;

		memset(&coop_input, 0, sizeof(coop_input));
		coop_input.thinktime = thinktime;
		VectorCopy(state->last_client_update.viewangles, coop_input.viewangles);
		if (BotAI_ApplyCoopHardLeash(state, thinktime, &coop_input))
		{
			BotAI_ConfigureBattleCombat(state);
			AI_DMState_ChangeViewAngles(state->dm_state, state, thinktime);
			BotState_EmitPendingClientCommands(state);
			stand_status = EA_EndRegular(state->client_number, thinktime);
			if (stand_status == BLERR_NOERROR)
			{
				state->client_update_valid = false;
			}
			return stand_status;
		}
		return BotAI_RunStand(state, thinktime);
	}

	ai_goal_selection_t selection = {0};
	bot_input_t input = {0};
	bool view_turn_pending = false;
	bool post_acquire_enemy = frame.post_acquire_enemy;
	int status = BLERR_NOERROR;
	if (frame.work == BOT_AI_FRAME_WORK_GOAL)
	{
		status = BotAI_RunGoalMovement(state,
			thinktime,
			&selection,
			&input,
			&view_turn_pending,
			&post_acquire_enemy);
		if (status != BLERR_NOERROR)
		{
			return status;
		}

		if (post_acquire_enemy)
		{
			ai_dm_enemy_info_t delayed_enemy;
			if (BotAI_FindEnemy(state, &delayed_enemy))
			{
				BotAI_EnterFoundEnemy(state, true);
			}
		}
		if (view_turn_pending)
		{
			AI_DMState_SetEnemyContext(state->dm_state,
				state->combat.current_enemy,
				state->combat.enemy_sight_time,
				state->combat.last_enemy_area,
				state->combat.last_enemy_origin);
			AI_DMState_ChangeViewAngles(state->dm_state, state, thinktime);
		}
	}
	else if (frame.work == BOT_AI_FRAME_WORK_FIGHT ||
		frame.work == BOT_AI_FRAME_WORK_CHASE)
	{
		if (frame.work == BOT_AI_FRAME_WORK_CHASE)
		{
			BotAI_UpdateEnemyBattleInventory(state,
				state->combat.current_enemy);
			BotAI_UseItems(state);
			status = BotAI_RunBattleChaseMovement(state,
				thinktime,
				&input);
			if (status != BLERR_NOERROR)
			{
				return status;
			}
		}
		else
		{
			input.thinktime = thinktime;
			VectorCopy(state->last_client_update.viewangles, input.viewangles);
			selection.valid = true;
			selection.candidate.travel_flags = BotAI_BattleChaseTravelFlags(state);
			BotAI_SelectBattleWeapon(state);

			if (state->dm_state != NULL)
			{
				AI_DMState_SetEnemyContext(state->dm_state,
					state->combat.current_enemy,
					state->combat.enemy_sight_time,
					state->combat.last_enemy_area,
					state->combat.last_enemy_origin);
				BotAI_BattleUseItems(state);
				BotAI_UseItems(state);
				AI_DMState_UpdateWithMoveResult(state->dm_state,
					state,
					&selection,
					&frame.enemy,
					&input,
					g_botInterfaceFrameTime,
					BotAI_HandleFightMoveResult,
					state);
				BotInterface_SynchroniseCombatState(state);
			}
		}

		if (BotAI_WantsToRetreat(state))
		{
			BotAI_EnterNode(state, BOT_AI_NODE_BATTLE_RETREAT);
		}
	}
	else if (frame.work == BOT_AI_FRAME_WORK_BATTLE_NBG)
	{
		BotAI_UseItems(state);
		status = BotAI_RunBattleNBGMovement(state,
			thinktime,
			&input,
			&frame.enemy);
		if (status != BLERR_NOERROR)
		{
			return status;
		}
	}
	else if (frame.work == BOT_AI_FRAME_WORK_BATTLE_RETREAT)
	{
		status = BotAI_RunBattleRetreatMovement(state,
			thinktime,
			&input,
			&frame.enemy,
			frame.has_movement_goal ? &frame.movement_goal : NULL);
		if (status != BLERR_NOERROR)
		{
			return status;
		}
	}
	else if (frame.work == BOT_AI_FRAME_WORK_BATTLE_RETREAT_IDLE)
	{
		status = BotAI_RunBattleRetreatIdle(state, thinktime, &input);
		if (status != BLERR_NOERROR)
		{
			return status;
		}
	}

	if (!BotAI_ApplyCoopHardLeash(state, thinktime, &input) &&
		!BotAI_ApplyCoopChangelevelGate(state, thinktime, &input) &&
		!BotAI_ApplyCoopControlWait(state, thinktime, &input) &&
		!BotAI_ApplyCoopControlReturnPath(state) &&
		!BotAI_ApplyCoopAreaAdvanceGate(state, thinktime, &input) &&
		!BotAI_ApplyCoopRescuePositioning(state, thinktime, &input) &&
		!BotAI_ApplyCoopRolePositioning(state, thinktime, &input) &&
		!BotAI_ApplyCoopFirelineAvoidance(state, thinktime, &input) &&
		!BotAI_ApplyCoopDoorwayAvoidance(state, thinktime, &input) &&
		!BotAI_ApplyCoopBasicCover(state, thinktime, &input))
	{
		(void)BotAI_ApplyCoopPersonalSpace(state, thinktime, &input);
	}

	BotState_EmitPendingClientCommands(state);

	status = EA_EndRegular(state->client_number, thinktime);
	if (status != BLERR_NOERROR)
	{
		return status;
	}

	state->client_update_valid = false;
	return BLERR_NOERROR;
}

/*
=============
BotAI

Runs the client think first, then the retail global entity-item update gate.
=============
*/
static int BotAI(int client, float thinktime)
{
	if (!BotInterface_EnsureLibraryReady("BotAI"))
	{
		return BLERR_LIBRARYNOTSETUP;
	}

	if (!BotInterface_ValidateClientNumber(client, "BotAI"))
	{
		return BLERR_INVALIDCLIENTNUMBER;
	}
	if (!AAS_Initialized())
	{
		return BLERR_NOERROR;
	}

	bot_client_state_t *state = BotState_Get(client);
	if (state == NULL || !state->active)
	{
		BotInterface_Printf(PRT_FATAL,
			"client %d hasn't been setup\n",
			client);
		return BLERR_AICLIENTNOTSETUP;
	}

	int status = BotAI_Think(state, thinktime);
	BotUpdateEntityItemsThrottled(g_botInterfaceFrameTime);
	if (LibVarGetValue("coopbot_log") >= 3.0f)
	{
		BotLib_LogWriteTimeStamped(
			"event=ai client=%d name=\"%s\" status=%d node=%d enemy=%d",
			client, BotState_ClientName(client), status,
			state->ai_node, state->combat.current_enemy);
	}
	return status;
}

/*
=============
BotConsoleMessage

Queues a console message while preserving the retail inactive-client failure.
=============
*/
static int BotConsoleMessage(int client, int type, char *message)
{
	if (!BotInterface_EnsureLibraryReady("BotConsoleMessage"))
	{
		return BLERR_LIBRARYNOTSETUP;
	}

	if (!BotInterface_ValidateClientNumber(client, "BotConsoleMessage"))
	{
		return BLERR_INVALIDCLIENTNUMBER;
	}

	bot_client_state_t *state = BotState_Get(client);
	if (state == NULL || !state->active)
	{
		BotInterface_Printf(PRT_ERROR,
			"recieved console message for inactive bot client\n");
		return BLERR_AICMFORINACTIVECLIENT;
	}

    if (state->chat_state == NULL)
    {
        BotInterface_Printf(PRT_WARNING,
                             "[bot_interface] BotConsoleMessage: client %d missing chat state\n",
                             client);
        return BLERR_CANNOTLOADICHAT;
    }

    if (message != NULL)
    {
        BotQueueConsoleMessage(state->chat_state, type, message);
    }

    return BLERR_NOERROR;
}
static bot_export_extended_t *BotInterface_GetBotAPI(const void *import,
	size_t import_size)
{
	static bot_export_extended_t exportTable;
	g_botExtendedExportTable = &exportTable;

	memset(&exportTable, 0, sizeof(exportTable));

	BotInterface_FreeImportCache();
	BotInterface_InitialiseImportTable(import, import_size);
	BotInterface_BuildImportTable(import);
	BotMemory_SetAllocatorCallbacks(
		BotInterface_GetEngineImport() != NULL ?
			BotInterface_GetEngineImport()->GetMemory : NULL,
		BotInterface_GetEngineImport() != NULL ?
			BotInterface_GetEngineImport()->FreeMemory : NULL);

	/*
	 * HLIL ordering: capture the allocator imports, translate compatibility
	 * shims, install the botlib import table, feed the bridge import table, and
	 * reset cached update translation state.
	 */
	BotInterface_SetImportTable(BotInterface_GetInterfaceImportTable());
	Q2Bridge_SetImportTable(BotInterface_GetEngineImport());
	Bridge_ResetCachedUpdates();
	Q2Bridge_SetDebugLinesEnabled(g_botInterfaceDebugDrawEnabled);
	assert(BotInterface_GetEngineImport() != NULL);

    exportTable.BotVersion = BotVersion;
    exportTable.BotSetupLibrary = BotSetupLibraryWrapper;
    exportTable.BotShutdownLibrary = BotShutdownLibraryWrapper;
    exportTable.BotLibraryInitialized = BotLibraryInitializedWrapper;
    exportTable.BotLibVarSet = BotLibVarSetWrapper;
    exportTable.BotDefine = BotDefineWrapper;
    exportTable.BotLoadMap = BotLoadMap;
    exportTable.BotSetupClient = BotSetupClient;
    exportTable.BotShutdownClient = BotShutdownClient;
    exportTable.BotMoveClient = BotMoveClient;
    exportTable.BotClientSettings = BotClientSettings;
    exportTable.BotSettings = BotSettings;
    exportTable.BotStartFrame = BotStartFrame;
    exportTable.BotUpdateClient = BotUpdateClient;
    exportTable.BotUpdateEntity = BotUpdateEntity;
    exportTable.BotAddSound = BotAddSound;
    exportTable.BotAddPointLight = BotAddPointLight;
    exportTable.BotAI = BotAI;
    exportTable.BotConsoleMessage = BotConsoleMessage;
    exportTable.BotAllocGoalState = AI_GoalBotlib_AllocState;
    exportTable.BotFreeGoalState = AI_GoalBotlib_FreeState;
    exportTable.BotResetGoalState = AI_GoalBotlib_ResetState;
    exportTable.BotLoadItemWeights = AI_GoalBotlib_LoadItemWeights;
    exportTable.BotFreeItemWeights = AI_GoalBotlib_FreeItemWeights;
    exportTable.BotPushGoal = AI_GoalBotlib_PushGoal;
    exportTable.BotPopGoal = AI_GoalBotlib_PopGoal;
    exportTable.BotEmptyGoalStack = AI_GoalBotlib_EmptyGoalStack;
    exportTable.BotGetTopGoal = AI_GoalBotlib_GetTopGoal;
    exportTable.BotGetSecondGoal = AI_GoalBotlib_GetSecondGoal;
    exportTable.BotChooseLTGItem = AI_GoalBotlib_ChooseLTG;
    exportTable.BotChooseNBGItem = AI_GoalBotlib_ChooseNBG;
    exportTable.BotResetAvoidGoals = AI_GoalBotlib_ResetAvoidGoals;
    exportTable.BotAddAvoidGoal = AI_GoalBotlib_AddAvoidGoal;
    exportTable.BotRemoveFromAvoidGoals = AI_GoalBotlib_RemoveFromAvoidGoals;
    exportTable.BotAvoidGoalTime = AI_GoalBotlib_AvoidGoalTime;
    exportTable.BotSetAvoidGoalTime = AI_GoalBotlib_SetAvoidGoalTime;
    exportTable.BotDumpAvoidGoals = AI_GoalBotlib_DumpAvoidGoals;
    exportTable.BotDumpGoalStack = AI_GoalBotlib_DumpGoalStack;
    exportTable.BotGoalName = AI_GoalBotlib_GoalName;
    exportTable.BotGetLevelItemGoal = AI_GoalBotlib_GetLevelItemGoal;
    exportTable.BotGetNextCampSpotGoal = AI_GoalBotlib_GetNextCampSpotGoal;
    exportTable.BotGetMapLocationGoal = AI_GoalBotlib_GetMapLocationGoal;
    exportTable.BotInterbreedGoalFuzzyLogic = AI_GoalBotlib_InterbreedGoalFuzzyLogic;
    exportTable.BotSaveGoalFuzzyLogic = AI_GoalBotlib_SaveGoalFuzzyLogic;
    exportTable.BotMutateGoalFuzzyLogic = AI_GoalBotlib_MutateGoalFuzzyLogic;
    exportTable.BotUpdateGoalState = AI_GoalBotlib_Update;
	exportTable.BotUpdateEntityItems = AI_GoalBotlib_UpdateEntityItems;
    exportTable.BotRegisterLevelItem = AI_GoalBotlib_RegisterLevelItem;
    exportTable.BotUnregisterLevelItem = AI_GoalBotlib_UnregisterLevelItem;
    exportTable.BotMarkLevelItemTaken = AI_GoalBotlib_MarkItemTaken;
	BotInterface_FillExportedWrappers(&exportTable);

	return &exportTable;
}

/*
=============
GetBotAPI

Copies only the immutable ten-callback Gladiator 0.96 import prefix and
returns a physically separate 20-callback retail export table.
=============
*/
GLADIATOR_API bot_export_t *GetBotAPI(bot_import_t *import)
{
	static bot_export_t retailExportTable;
	g_botRetailExportTable = &retailExportTable;
	bot_export_extended_t *extendedExportTable = BotInterface_GetBotAPI(import,
		BOT_IMPORT_RETAIL_SIZE);

	/* Do not return the first bytes of the extension table to retail callers. */
	memcpy(&retailExportTable, extendedExportTable, sizeof(retailExportTable));
	return &retailExportTable;
}

/*
=============
GetBotAPIEx

Accepts the reconstruction's optional import tail through an explicit size.
This in-repo seam deliberately has no dllexport marker: the retail DLL has
only the GetBotAPI export.
=============
*/
bot_export_extended_t *GetBotAPIEx(bot_import_extended_t *import,
	size_t import_size)
{
	return BotInterface_GetBotAPI(import, import_size);
}
