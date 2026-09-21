#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_EXPORTS_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_EXPORTS_H

#include <stdbool.h>

#include "q2bridge/botlib.h"

#ifdef __cplusplus
extern "C" {
#endif

bool BotInterface_EnsureLibraryReady(const char *function_name);
void BotInterface_FillExportedWrappers(bot_export_extended_t *export_table);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_EXPORTS_H */
