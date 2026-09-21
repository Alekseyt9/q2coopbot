#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_IMPORT_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_IMPORT_H

#include <stdbool.h>
#include <stddef.h>

#include "q2bridge/botlib.h"
#include "botlib_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

bot_import_extended_t *BotInterface_GetEngineImport(void);
botlib_import_table_t *BotInterface_GetInterfaceImportTable(void);
botlib_import_table_t *BotInterface_GetBotlibImportTable(void);
void BotInterface_ClearEngineImportState(void);

void BotInterface_FreeImportCache(void);
bool BotInterface_UpdateImportCache(const char *name, const char *value);
void BotInterface_InitialiseImportTable(const void *imports, size_t import_size);
void BotInterface_BuildImportTable(const void *import_table);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_IMPORT_H */
