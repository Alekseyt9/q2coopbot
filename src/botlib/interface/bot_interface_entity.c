#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "shared/q_platform.h"
#include "q2bridge/aas_translation.h"
#include "q2bridge/botlib.h"
#include "q2bridge/bridge.h"
#include "q2bridge/update_translator.h"
#include "botlib/aas/aas_map.h"
#include "botlib/aas/aas_local.h"
#include "botlib/aas/aas_sound.h"
#include "botlib_interface.h"
#include "bot_interface_entity.h"
#include "bot_interface.h"
#include "bot_state.h"

typedef struct botinterface_entity_snapshot_s
{
    qboolean valid;
    bot_updateentity_t state;
} botinterface_entity_snapshot_t;

static botinterface_entity_snapshot_t g_botInterfaceEntityCache[BOT_INTERFACE_MAX_ENTITIES];

bool BotInterface_ReadEntityCache(int ent, bot_updateentity_t *state)
{
    if (state == NULL || ent < 0 || ent >= BOT_INTERFACE_MAX_ENTITIES ||
        !g_botInterfaceEntityCache[ent].valid)
    {
        return false;
    }

    *state = g_botInterfaceEntityCache[ent].state;
    return true;
}

void BotInterface_ResetEntityCache(void)
{
    for (size_t index = 0; index < BOT_INTERFACE_MAX_ENTITIES; ++index)
    {
        g_botInterfaceEntityCache[index].valid = qfalse;
    }
}/*
=============
BotInterface_EnqueueSound

Forward one sound update through the retail status-returning sound leaf.
=============
*/
static int BotInterface_EnqueueSound(const vec3_t origin,
	int ent,
	int channel,
	int soundindex,
	float volume,
	float attenuation,
	float timeofs)
{
	return AAS_SoundSubsystem_UpdateSound(origin,
		ent,
		channel,
		soundindex,
		volume,
		attenuation,
		timeofs);
}

static void BotInterface_EnqueuePointLight(vec3_t origin,
                                           int ent,
                                           float radius,
                                           float r,
                                           float g,
                                           float b,
                                           float time,
                                           float decay)
{
    if (!AAS_SoundSubsystem_RecordPointLight(origin, ent, radius, r, g, b, time, decay))
    {
        BotInterface_Printf(PRT_WARNING,
                             "[bot_interface] BotAddPointLight: point light queue capacity exceeded\n");
    }
}/*
=============
BotInterface_ValidateEntityNumber

Applies the shared retail entity-number guard before exported entity work.
=============
*/
qboolean BotInterface_ValidateEntityNumber(int ent, const char *function_name)
{
	const botlib_library_variables_t *variables = BotInterface_GetLibraryVariables();
	int max_entity_number = BOT_INTERFACE_MAX_ENTITIES;
	if (variables != NULL)
	{
		max_entity_number = variables->maxentities;
	}

	if (ent >= 0 && ent <= max_entity_number)
	{
		return qtrue;
	}

	BotInterface_Printf(PRT_ERROR,
		"%s: invalid entity number %d, [0, %d]\n",
		function_name,
		ent,
		max_entity_number);
	return qfalse;
}/*
=============
BotUpdateEntity

Validates and forwards an entity snapshot to the bridge and AAS world.
=============
*/
int BotUpdateEntity(int ent, bot_updateentity_t *bue)
{
	if (!BotInterface_EnsureLibraryReady("BotUpdateEntity"))
	{
		return BLERR_LIBRARYNOTSETUP;
	}

	if (!BotInterface_ValidateEntityNumber(ent, "BotUpdateEntity"))
	{
		return BLERR_INVALIDENTITYNUMBER;
	}

	if (bue == NULL)
	{
		return AAS_UpdateEntity(ent, NULL);
	}

	int status = Bridge_UpdateEntity(ent, bue);
	if (status != BLERR_NOERROR)
	{
		return status;
	}

	AASEntityFrame translated = {0};
	if (!Bridge_ReadEntityFrame(ent, &translated))
	{
		return BLERR_INVALIDIMPORT;
	}

	status = AAS_UpdateEntity(ent, &translated);
	if (status != BLERR_NOERROR)
	{
		return status;
	}

	if (bue != NULL && ent >= 0 && ent < BOT_INTERFACE_MAX_ENTITIES)
	{
		g_botInterfaceEntityCache[ent].state = *bue;
		g_botInterfaceEntityCache[ent].valid = qtrue;
	}

	aasworld.entitiesValid = qtrue;
	return BLERR_NOERROR;
}/*
=============
BotAddSound

Validates the source entity then forwards the raw sound status and diagnostics.
=============
*/
int BotAddSound(vec3_t origin,
	int ent,
	int channel,
	int soundindex,
	float volume,
	float attenuation,
	float timeofs)
{
	if (!BotInterface_EnsureLibraryReady("BotUpdateSound"))
	{
		return BLERR_LIBRARYNOTSETUP;
	}

	if (!BotInterface_ValidateEntityNumber(ent, "BotUpdateSound"))
	{
		return BLERR_INVALIDENTITYNUMBER;
	}

	return BotInterface_EnqueueSound(origin,
		ent,
		channel,
		soundindex,
		volume,
		attenuation,
		timeofs);
}

/*
=============
BotAddPointLight

Validates the source entity before recording a retail point light update.
=============
*/
int BotAddPointLight(vec3_t origin,
	int ent,
	float radius,
	float r,
	float g,
	float b,
	float time,
	float decay)
{
	if (!BotInterface_EnsureLibraryReady("BotAddPointLight"))
	{
		return BLERR_LIBRARYNOTSETUP;
	}

	if (!BotInterface_ValidateEntityNumber(ent, "BotAddPointLight"))
	{
		return BLERR_INVALIDENTITYNUMBER;
	}

	BotInterface_EnqueuePointLight(origin, ent, radius, r, g, b, time, decay);
	return BLERR_NOERROR;
}
