#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_ENTITY_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_ENTITY_H

#include <stdbool.h>

#include "bot_interface.h"

#define BOT_INTERFACE_MAX_ENTITIES 1024

#ifdef __cplusplus
extern "C" {
#endif

void BotInterface_ResetEntityCache(void);
bool BotInterface_ReadEntityCache(int ent, bot_updateentity_t *state);
void BotInterface_Printf(int priority, const char *fmt, ...);
bool BotInterface_EnsureLibraryReady(const char *function_name);
qboolean BotInterface_ValidateEntityNumber(int ent,
	const char *function_name);
int BotUpdateEntity(int ent, bot_updateentity_t *bue);
int BotAddSound(vec3_t origin,
	int ent,
	int channel,
	int soundindex,
	float volume,
	float attenuation,
	float timeofs);
int BotAddPointLight(vec3_t origin,
	int ent,
	float radius,
	float r,
	float g,
	float b,
	float time,
	float decay);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_ENTITY_H */

