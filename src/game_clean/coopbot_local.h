/*
 * Compatibility preinclude for the clean Yamagi game module.
 *
 * The gameplay implementation comes from Yamagi's current baseq2 sources.
 * Only the bot bridge is added here; the bridge is deliberately kept out of
 * the gameplay structs and does not replace the engine's game ABI.
 */
#ifndef COOPBOT_CLEAN_LOCAL_H
#define COOPBOT_CLEAN_LOCAL_H

#define BOT
#define BOT_IMPORT
#define C_ONLY

#if defined(WIN32) || defined(_WIN32)
#include <windows.h>
#ifdef hyper
#undef hyper
#endif
#endif

#include "header/local.h"

#define FL_BOT            0x00002000
#define FL_BOTINPUT       0x00004000
#define FL_OLDORGNOTSET   0x00008000

#include "botlib.h"
#include "bl_main.h"
#include "bl_cmd.h"
#include "bl_spawn.h"
#include "bl_redirgi.h"
#include "bl_debug.h"
#include "coopbot_diag.h"

extern int paused;

#endif
