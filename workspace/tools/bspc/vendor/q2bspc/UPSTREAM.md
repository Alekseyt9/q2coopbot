# Quake II BSPC source

Source: https://github.com/RaZeR-RBI/bspc at commit `e5e1c95dc23e2fb51e147b34c1922a9c841ae3a3`.
License: GPL-2.0-or-later; see `LICENSE` in this directory. This utility is
built separately from the Go client and does not enter the game runtime.

The local Windows build fixes are limited to C17 selection, compatible
Win32 thread IDs, current `time_t`, avoiding a free of `ctime`'s static buffer,
and Quake II collision loader declarations/signatures and buffer cleanup.
The Quake II reachability implementation itself is upstream's.
