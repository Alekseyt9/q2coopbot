#include <stdlib.h>
#include <string.h>

#include "botlib/aas/aas_local.h"
#include "bot_interface_assets.h"

typedef struct botinterface_map_cache_s
{
	char map_name[MAX_FILEPATH];
	botinterface_asset_list_t models;
	botinterface_asset_list_t sounds;
	botinterface_asset_list_t images;
} botinterface_map_cache_t;

static botinterface_map_cache_t g_botInterfaceMapCache;

static char *BotInterface_AssetCopyString(const char *text)
{
	size_t length;
	char *copy;

	if (text == NULL)
	{
		return NULL;
	}
	length = strlen(text);
	copy = (char *)malloc(length + 1);
	if (copy == NULL)
	{
		return NULL;
	}
	memcpy(copy, text, length);
	copy[length] = '\0';
	return copy;
}

static void BotInterface_FreeAssetList(botinterface_asset_list_t *list)
{
	if (list == NULL)
	{
		return;
	}
	if (list->entries != NULL)
	{
		for (size_t index = 0; index < list->count; ++index)
		{
			free(list->entries[index]);
		}
		free(list->entries);
	}
	list->entries = NULL;
	list->count = 0;
}

static bool BotInterface_CopyAssetList(botinterface_asset_list_t *target,
	int count,
	char *source[])
{
	if (target == NULL)
	{
		return false;
	}
	BotInterface_FreeAssetList(target);
	if (count <= 0 || source == NULL)
	{
		return true;
	}

	size_t allocation = (size_t)count;
	char **table = (char **)calloc(allocation, sizeof(char *));
	if (table == NULL)
	{
		return false;
	}
	for (size_t index = 0; index < allocation; ++index)
	{
		if (source[index] == NULL)
		{
			continue;
		}
		table[index] = BotInterface_AssetCopyString(source[index]);
		if (table[index] == NULL)
		{
			for (size_t rollback = 0; rollback < index; ++rollback)
			{
				free(table[rollback]);
			}
			free(table);
			return false;
		}
	}
	target->entries = table;
	target->count = allocation;
	return true;
}

bool BotInterface_RecordMapAssets(const char *mapname,
	int modelindexes,
	char *modelindex[],
	int soundindexes,
	char *soundindex[],
	int imageindexes,
	char *imageindex[])
{
	botinterface_asset_list_t models = {0};
	botinterface_asset_list_t sounds = {0};
	botinterface_asset_list_t images = {0};

	if (!BotInterface_CopyAssetList(&models, modelindexes, modelindex) ||
		!BotInterface_CopyAssetList(&sounds, soundindexes, soundindex) ||
		!BotInterface_CopyAssetList(&images, imageindexes, imageindex))
	{
		BotInterface_FreeAssetList(&models);
		BotInterface_FreeAssetList(&sounds);
		BotInterface_FreeAssetList(&images);
		return false;
	}
	BotInterface_FreeAssetList(&g_botInterfaceMapCache.models);
	BotInterface_FreeAssetList(&g_botInterfaceMapCache.sounds);
	BotInterface_FreeAssetList(&g_botInterfaceMapCache.images);
	g_botInterfaceMapCache.models = models;
	g_botInterfaceMapCache.sounds = sounds;
	g_botInterfaceMapCache.images = images;
	if (mapname != NULL)
	{
		strncpy(g_botInterfaceMapCache.map_name,
			mapname,
			sizeof(g_botInterfaceMapCache.map_name) - 1);
		g_botInterfaceMapCache.map_name[
			sizeof(g_botInterfaceMapCache.map_name) - 1] = '\0';
	}
	return true;
}

void BotInterface_ResetMapAssets(void)
{
	BotInterface_FreeAssetList(&g_botInterfaceMapCache.models);
	BotInterface_FreeAssetList(&g_botInterfaceMapCache.sounds);
	BotInterface_FreeAssetList(&g_botInterfaceMapCache.images);
	g_botInterfaceMapCache.map_name[0] = '\0';
}

char **BotInterface_MapModelEntries(void)
{
	return g_botInterfaceMapCache.models.entries;
}

size_t BotInterface_MapModelCount(void)
{
	return g_botInterfaceMapCache.models.count;
}

const char *BotInterface_ModelNameForIndex(int modelindex)
{
	return AAS_ModelFromIndex(modelindex);
}

const char *BotInterface_ImageNameForIndex(int imageindex)
{
	return AAS_ImageFromIndex(imageindex);
}
