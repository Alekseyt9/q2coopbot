#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_ASSETS_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_ASSETS_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct botinterface_asset_list_s
{
    char **entries;
    size_t count;
} botinterface_asset_list_t;

bool BotInterface_RecordMapAssets(const char *mapname,
	int modelindexes,
	char *modelindex[],
	int soundindexes,
	char *soundindex[],
	int imageindexes,
	char *imageindex[]);

void BotInterface_ResetMapAssets(void);

char **BotInterface_MapModelEntries(void);
size_t BotInterface_MapModelCount(void);

const char *BotInterface_ModelNameForIndex(int modelindex);
const char *BotInterface_ImageNameForIndex(int imageindex);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_ASSETS_H */
