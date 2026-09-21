#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "q2bridge/botlib.h"
#include "botlib_interface.h"
#include "bot_interface.h"
#include "bot_interface_import.h"

static bot_import_extended_t g_botImportStorage;
static bot_import_extended_t *g_botImport = NULL;
static botlib_import_table_t g_botInterfaceImportTable;

bot_import_extended_t *BotInterface_GetEngineImport(void)
{
    return g_botImport;
}

botlib_import_table_t *BotInterface_GetInterfaceImportTable(void)
{
    return &g_botInterfaceImportTable;
}

void BotInterface_ClearEngineImportState(void)
{
    memset(&g_botInterfaceImportTable, 0, sizeof(g_botInterfaceImportTable));
    memset(&g_botImportStorage, 0, sizeof(g_botImportStorage));
    g_botImport = NULL;
}

typedef struct botinterface_import_libvar_s {
    char *name;
    char *value;
    struct botinterface_import_libvar_s *next;
} botinterface_import_libvar_t;

static botinterface_import_libvar_t *g_botInterfaceLibVars = NULL;

static void BotInterface_FreeImportLibVar(botinterface_import_libvar_t *entry)
{
    if (entry == NULL)
    {
        return;
    }

    free(entry->name);
    free(entry->value);
    free(entry);
}

static void BotInterface_ResetImportLibVars(void)
{
    botinterface_import_libvar_t *entry = g_botInterfaceLibVars;
    while (entry != NULL)
    {
        botinterface_import_libvar_t *next = entry->next;
        BotInterface_FreeImportLibVar(entry);
        entry = next;
    }

    g_botInterfaceLibVars = NULL;
}

static char *BotInterface_CopyString(const char *text);

static botinterface_import_libvar_t *BotInterface_FindImportLibVar(const char *name)
{
    botinterface_import_libvar_t *entry = g_botInterfaceLibVars;
    while (entry != NULL)
    {
        if (entry->name != NULL && name != NULL && strcmp(entry->name, name) == 0)
        {
            return entry;
        }

        entry = entry->next;
    }

    return NULL;
}

static botinterface_import_libvar_t *BotInterface_EnsureImportLibVar(const char *name)
{
    if (name == NULL)
    {
        return NULL;
    }

    botinterface_import_libvar_t *entry = BotInterface_FindImportLibVar(name);
    if (entry != NULL)
    {
        return entry;
    }

    entry = calloc(1, sizeof(*entry));
    if (entry == NULL)
    {
        return NULL;
    }

    entry->name = BotInterface_CopyString(name);
    if (entry->name == NULL)
    {
        free(entry);
        return NULL;
    }

    entry->value = BotInterface_CopyString("");
    if (entry->value == NULL)
    {
        free(entry->name);
        free(entry);
        return NULL;
    }

    entry->next = g_botInterfaceLibVars;
    g_botInterfaceLibVars = entry;
    return entry;
}

static char *BotInterface_CopyString(const char *text)
{
    if (text == NULL)
    {
        return NULL;
    }

    size_t length = strlen(text);
    char *copy = (char *)malloc(length + 1);
    if (copy == NULL)
    {
        return NULL;
    }

    memcpy(copy, text, length);
    copy[length] = '\0';
    return copy;
}

/*
=============
BotInterface_PrintWrapper

Formats print output and forwards it through the engine import table.
=============
*/
static void BotInterface_PrintWrapper(int type, const char *fmt, ...)
{
	if (fmt == NULL)
	{
		return;
	}

	if (g_botImport == NULL || g_botImport->Print == NULL)
	{
		return;
	}

	va_list args;
	va_start(args, fmt);

	char buffer[1024];
	vsnprintf(buffer, sizeof(buffer), fmt, args);

	va_end(args);

	buffer[sizeof(buffer) - 1] = '\0';

	const botlib_import_capture_t *capture = BotInterface_GetImportCapture();
	if (capture != NULL && capture->Print != NULL)
	{
		capture->Print(type, buffer);
	}

	g_botImport->Print(type, "%s", buffer);
}

/*
=============
BotInterface_DPrintWrapper

Formats developer output and forwards it through the engine print callback.
=============
*/
static void BotInterface_DPrintWrapper(const char *fmt, ...)
{
	if (fmt == NULL)
	{
		return;
	}

	if (g_botImport == NULL || g_botImport->Print == NULL)
	{
		return;
	}

	va_list args;
	va_start(args, fmt);

	char buffer[1024];
	vsnprintf(buffer, sizeof(buffer), fmt, args);

	va_end(args);

	buffer[sizeof(buffer) - 1] = '\0';

	const botlib_import_capture_t *capture = BotInterface_GetImportCapture();
	if (capture != NULL && capture->DPrint != NULL)
	{
		capture->DPrint(buffer);
	}

	g_botImport->Print(PRT_MESSAGE, "%s", buffer);
}

static void BotInterface_AddCommandWrapper(const char *name, void (*function)(void))
{
    if (g_botImport == NULL || g_botImport->AddCommand == NULL || name == NULL || function == NULL)
    {
        return;
    }

    g_botImport->AddCommand(name, function);
}

static void BotInterface_RemoveCommandWrapper(const char *name)
{
    if (g_botImport == NULL || g_botImport->RemoveCommand == NULL || name == NULL)
    {
        return;
    }

    g_botImport->RemoveCommand(name);
}

static int BotInterface_CmdArgcWrapper(void)
{
    if (g_botImport == NULL || g_botImport->CmdArgc == NULL)
    {
        return 0;
    }

    return g_botImport->CmdArgc();
}

static const char *BotInterface_CmdArgvWrapper(int index)
{
    if (g_botImport == NULL || g_botImport->CmdArgv == NULL)
    {
        return NULL;
    }

    return g_botImport->CmdArgv(index);
}

/*
=============
BotInterface_BotLibVarGetWrapper

Returns cached libvar values from the import-side shim table.
=============
*/
static int BotInterface_BotLibVarGetWrapper(const char *var_name, char *value, size_t size)
{
	int status = -1;

	if (value == NULL || size == 0)
	{
		return -1;
	}

	value[0] = '\0';

	if (var_name == NULL)
	{
		return -1;
	}

	botinterface_import_libvar_t *entry = BotInterface_FindImportLibVar(var_name);
	if (entry != NULL && entry->value != NULL)
	{
		size_t length = strlen(entry->value);
		if (length >= size)
		{
			length = size - 1;
		}

		memcpy(value, entry->value, length);
		value[length] = '\0';
		status = 0;
	}

	const botlib_import_capture_t *capture = BotInterface_GetImportCapture();
	if (capture != NULL && capture->BotLibVarGet != NULL)
	{
		capture->BotLibVarGet(var_name, value, status);
	}

	return status;
}

/*
=============
BotInterface_BotLibVarSetWrapper

Updates cached libvar values supplied through the import-side shim table.
=============
*/
static int BotInterface_BotLibVarSetWrapper(const char *var_name, const char *value)
{
	int status = -1;

	if (var_name == NULL || value == NULL)
	{
		return -1;
	}

	botinterface_import_libvar_t *entry = BotInterface_EnsureImportLibVar(var_name);
	if (entry != NULL)
	{
		char *copy = BotInterface_CopyString(value);
		if (copy != NULL)
		{
			free(entry->value);
			entry->value = copy;
			status = 0;
		}
	}

	const botlib_import_capture_t *capture = BotInterface_GetImportCapture();
	if (capture != NULL && capture->BotLibVarSet != NULL)
	{
		capture->BotLibVarSet(var_name, value, status);
	}

	return status;
}

void BotInterface_BuildImportTable(const void *import_table)
{
    (void)import_table;

    BotInterface_ResetImportLibVars();

    g_botInterfaceImportTable.Print = BotInterface_PrintWrapper;
    g_botInterfaceImportTable.DPrint = BotInterface_DPrintWrapper;
    g_botInterfaceImportTable.BotLibVarGet = BotInterface_BotLibVarGetWrapper;
    g_botInterfaceImportTable.BotLibVarSet = BotInterface_BotLibVarSetWrapper;
    g_botInterfaceImportTable.AddCommand = BotInterface_AddCommandWrapper;
    g_botInterfaceImportTable.RemoveCommand = BotInterface_RemoveCommandWrapper;
    g_botInterfaceImportTable.CmdArgc = BotInterface_CmdArgcWrapper;
    g_botInterfaceImportTable.CmdArgv = BotInterface_CmdArgvWrapper;
}

typedef struct botlib_import_cache_entry_s {
    struct botlib_import_cache_entry_s *next;
    char *name;
    char *value;
} botlib_import_cache_entry_t;

static botlib_import_cache_entry_t *g_botImportCache = NULL;
static botlib_import_table_t g_botlibImportTable = {0};

botlib_import_table_t *BotInterface_GetBotlibImportTable(void)
{
    return &g_botlibImportTable;
}

/*
=============
BotInterface_FreeImportCache

Releases the local copy of imported library variables.
=============
*/
void BotInterface_FreeImportCache(void)
{
	botlib_import_cache_entry_t *entry = g_botImportCache;
	while (entry != NULL)
	{
		botlib_import_cache_entry_t *next = entry->next;
		free(entry->name);
		free(entry->value);
		free(entry);
		entry = next;
	}

	g_botImportCache = NULL;
}

/*
=============
BotInterface_ImportCacheEntry

Report the index'th host-set libvar held in the bridge import cache.

BotSetupLibrary uses this to re-materialise the local libvar list after its
reset, restoring the retail invariant that host values are already present
before the first getter runs.  Iteration is by index rather than by name
because the shim matches with case-sensitive strcmp while retail's LibVarGet
uses _strcmpi.
=============
*/
bool BotInterface_ImportCacheEntry(int index, const char **name, const char **value)
{
	if (index < 0 || name == NULL || value == NULL)
	{
		return false;
	}

	/*
	 * The cache prepends, so walking it directly yields the most recently set
	 * name first.  Index from the oldest entry instead: retail never clears
	 * libvarlist, so its entries sit in the order the host pushed them, and
	 * the seed has to reproduce that order rather than invert it.
	 */
	int count = 0;
	for (botlib_import_cache_entry_t *entry = g_botImportCache;
		entry != NULL;
		entry = entry->next)
	{
		++count;
	}
	if (index >= count)
	{
		return false;
	}

	int remaining = count - 1 - index;
	for (botlib_import_cache_entry_t *entry = g_botImportCache;
		entry != NULL;
		entry = entry->next)
	{
		if (remaining-- == 0)
		{
			*name = entry->name;
			*value = entry->value;
			return *name != NULL && *value != NULL;
		}
	}

	return false;
}

bool BotInterface_UpdateImportCache(const char *name, const char *value)
{
    if (name == NULL || value == NULL)
    {
        return false;
    }

    for (botlib_import_cache_entry_t *entry = g_botImportCache; entry != NULL; entry = entry->next)
    {
        if (strcmp(entry->name, name) == 0)
        {
            char *copy = BotInterface_CopyString(value);
            if (copy == NULL)
            {
                return false;
            }

            free(entry->value);
            entry->value = copy;
            return true;
        }
    }

    botlib_import_cache_entry_t *fresh = (botlib_import_cache_entry_t *)calloc(1, sizeof(*fresh));
    if (fresh == NULL)
    {
        return false;
    }

    fresh->name = BotInterface_CopyString(name);
    fresh->value = BotInterface_CopyString(value);
    if (fresh->name == NULL || fresh->value == NULL)
    {
        free(fresh->name);
        free(fresh->value);
        free(fresh);
        return false;
    }

    fresh->next = g_botImportCache;
    g_botImportCache = fresh;
    return true;
}

/*
=============
BotInterface_BotLibVarGetShim

Fetches cached libvar values for the bootstrapped import table.
=============
*/
static int BotInterface_BotLibVarGetShim(const char *name, char *buffer, size_t buffer_size)
{
	int status = BLERR_INVALIDIMPORT;

	if (name == NULL || buffer == NULL || buffer_size == 0)
	{
		return BLERR_INVALIDIMPORT;
	}

	for (botlib_import_cache_entry_t *entry = g_botImportCache; entry != NULL; entry = entry->next)
	{
		if (strcmp(entry->name, name) == 0)
		{
			strncpy(buffer, entry->value, buffer_size - 1);
			buffer[buffer_size - 1] = '\0';
			status = BLERR_NOERROR;
			break;
		}
	}

	if (status != BLERR_NOERROR)
	{
		buffer[0] = '\0';
	}

	const botlib_import_capture_t *capture = BotInterface_GetImportCapture();
	if (capture != NULL && capture->BotLibVarGet != NULL)
	{
		capture->BotLibVarGet(name, buffer, status);
	}

	return status;
}

/*
=============
BotInterface_BotLibVarSetShim

Stores cached libvar values for the bootstrapped import table.
=============
*/
static int BotInterface_BotLibVarSetShim(const char *name, const char *value)
{
	int status = BLERR_INVALIDIMPORT;

	if (name == NULL || value == NULL)
	{
		return BLERR_INVALIDIMPORT;
	}

	if (BotInterface_UpdateImportCache(name, value))
	{
		status = BLERR_NOERROR;
	}

	const botlib_import_capture_t *capture = BotInterface_GetImportCapture();
	if (capture != NULL && capture->BotLibVarSet != NULL)
	{
		capture->BotLibVarSet(name, value, status);
	}

	return status;
}

/*
=============
BotInterface_PrintShim

Formats and forwards print output during the import table handshake.
=============
*/
static void BotInterface_PrintShim(int priority, const char *fmt, ...)
{
	if (g_botImport == NULL || g_botImport->Print == NULL || fmt == NULL)
	{
		return;
	}

	va_list args;
	va_start(args, fmt);

	char buffer[1024];
	vsnprintf(buffer, sizeof(buffer), fmt, args);

	va_end(args);

	const botlib_import_capture_t *capture = BotInterface_GetImportCapture();
	if (capture != NULL && capture->Print != NULL)
	{
		capture->Print(priority, buffer);
	}

	g_botImport->Print(priority, "%s", buffer);
}

/*
=============
BotInterface_DPrintShim

Formats and forwards developer output during the import table handshake.
=============
*/
static void BotInterface_DPrintShim(const char *fmt, ...)
{
	if (g_botImport == NULL || g_botImport->Print == NULL || fmt == NULL)
	{
		return;
	}

	va_list args;
	va_start(args, fmt);

	char buffer[1024];
	vsnprintf(buffer, sizeof(buffer), fmt, args);

	va_end(args);

	const botlib_import_capture_t *capture = BotInterface_GetImportCapture();
	if (capture != NULL && capture->DPrint != NULL)
	{
		capture->DPrint(buffer);
	}

	g_botImport->Print(PRT_MESSAGE, "%s", buffer);
}

/*
=============
BotInterface_InitialiseImportTable

Prepares the bootstrapped import table used during library setup.
=============
*/
void BotInterface_InitialiseImportTable(const void *imports,
	size_t import_size)
{
	memset(&g_botImportStorage, 0, sizeof(g_botImportStorage));
	if (imports != NULL)
	{
		size_t copy_size = import_size;
		if (copy_size > sizeof(g_botImportStorage))
		{
			copy_size = sizeof(g_botImportStorage);
		}
		memcpy(&g_botImportStorage, imports, copy_size);
		g_botImport = &g_botImportStorage;
	}
	else
	{
		g_botImport = NULL;
	}

	memset(&g_botlibImportTable, 0, sizeof(g_botlibImportTable));
	g_botlibImportTable.Print = BotInterface_PrintShim;
	g_botlibImportTable.DPrint = BotInterface_DPrintShim;
	g_botlibImportTable.BotLibVarGet = BotInterface_BotLibVarGetShim;
	g_botlibImportTable.BotLibVarSet = BotInterface_BotLibVarSetShim;
	g_botlibImportTable.AddCommand = BotInterface_AddCommandWrapper;
	g_botlibImportTable.RemoveCommand = BotInterface_RemoveCommandWrapper;
	g_botlibImportTable.CmdArgc = BotInterface_CmdArgcWrapper;
	g_botlibImportTable.CmdArgv = BotInterface_CmdArgvWrapper;
}
