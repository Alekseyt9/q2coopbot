#include "g_local.h"

/* The old bridge exposed optional bot-menu helpers. They are not part of
 * Yamagi's baseq2 client state, so keep these optional UI hooks harmless. */
void ToggleBotMenu(edict_t *ent)
{
	(void)ent;
}

void SendStatusBar(edict_t *ent, char *bar)
{
	(void)ent;
	(void)bar;
}
