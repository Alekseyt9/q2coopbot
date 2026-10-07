# MG/Shotgun/Blaster: native fixture и первое обучение

07.10.2026. В существующий Go/PowerShell харнес добавлены `weapons` (MG40 bullets + Shotgun20 shells + Blaster) и `weapons-scarce` (MG10 + Shotgun6 + Blaster). Стартовое оружие MG, native release100/idle gunframe6 сохранены. Политика допускается к Shotgun в этих isolated synchronous fixtures. Harness требует explicit PPO masked weapon head; Go Run также проверяет тип/версию provider. Старые fixed-loadout режимы сохранены.

## Изменения и проверки

Test-only setup выдаёт объявленное оружие/ammo. Native barrier требует fresh inventory age0..2, наличие всех3 оружий и точные ammo counts/HUD. `ResetExpectation` умеет задавать inventory list; offline reset proof проверяет перечисленные counts и freshness, но не объявляет unlisted inventory/полный мир/RNG/AI восстановленными. PPO exporter проверяет MG start phase для новых loadouts.

Shotgun firing проходит тот же прямой usercmd путь. Barrel spread guard использует консервативный18-degree envelope, включающий stock Shotgun1000/500 spread и2-degree kick. Он меняет только attack, не aim/movement. При наблюдаемом напарнике Shotgun fire в этом isolated pilot блокируется; co-op acceptance остаётся отдельным этапом.

Fixture/ownership/pellet-guard tests прошли, ownership test выполнен с реальным base1 BSP из существующего Steam pak. Reset test отклоняет неизвестный/устаревший inventory и shells mismatch даже при правильном HUD. `go test ./cmd/... ./internal/...` прошёл после final edits. `go test ./...` также запускался: старые generated duplicate main по-прежнему ломают общий проход; при глобальном Steam-only `Q2_SEARCH_SCAN_ROOT` дополнительные campaign tests не находят AAS. Этот root использован лишь для focused BSP test; полный product pass выполнен без него. Лог общей попытки сохранён в artifact root, ошибки не скрыты.

Первый boot на seeds44500–44503 остановился до gameplay из-за прежнего fixture whitelist в Go Run. Whitelist расширен; failed receipts сохранены, повторный запуск использовал новые seeds и новый output directory.

## Native diagnostic, исключённый из обучения

Artifacts: `workspace/artifacts/combat-multiweapon-v1-20261007/`. Probe — явно не обученная PPO diagnostic network: запрашивает Shotgun, отпускает fire до UDP-observed Shotgun feature18, после этого стреляет. Ни демонстрацией, ни боевым кандидатом эти weights не являются.

4 server/client instances x2, distinct seeds44600–44603, Mixed, scarce inventory. Все4 captures/reset/native execution/dispatch приняты.377 first-life PPO rows прошли Go replay. Во всех4 случаях первая UDP-equipped Shotgun observation на frame103. Native Shotgun mod2 health damage175 на episode; расход shells6,6,4,6. MG не стрелял перед сменой, поэтому этот probe показывает request-driven смену, а не ammo-empty fallback.

## On-policy batch и Update26

Source Update25 SHA256 `07725dbbf3f462063f716948bfea3d3499667d7403723d46552621409c32fe38`. Fresh stochastic Mixed `weapons`, seeds44700–44703,4 instances x2, stock HP, release100/max300, unchanged recoil-v5 reward. Все4 captures/dispatch/reset accepted;310 eligible first-life/context transitions,4 death terminals. Diagnostic rows не включены.

Samples:302 keep,5 Blaster,2 Machinegun,1 Shotgun. Native real-policy Shotgun damage144 на44700 и48 на44702. На44702 policy запросила Shotgun при4 оставшихся bullets; subsequent equipped Shotgun наблюдался, но рядом с ammo exhaustion. Stock NoAmmoWeaponChange тоже умеет выбирать Shotgun, поэтому это не самостоятельное доказательство выученной полезной смены. Все4 training episodes завершились first-life death, wins0. Не сравнивать это напрямую с прежними MG100 episodes: resource fixture изменён.

CUDA-only PPO resume на RTX5070 от Update25:26 completed updates,249 total actor steps;9 accepted actor steps,40 critic steps. Final approximate KL0.009998187<0.01. Head и inventory weights участвуют в loss; useful weapon-choice acceptance пока нет.

Update26 weights SHA256 `b7e3b60bfc17bea5fd38c8c8acc35af881702082d1a0ffac47d5a454f5bb7062`; checkpoint `a7a4f72bd6db0122fbd21980465ed5fd1edf5d2c42459710723e10f1a1e229f8`. Durable update completion receipt/report сохранены. Curriculum24 остаётся подтверждённым fixed-MG reference; Update26 — experimental multiweapon training checkpoint, live/default не переключены.

## Следующая работа

Больше свежих batches с normal/scarce ammo и Solo/Mixed,4 instances x2; обучение только CUDA с сохранением Adam/RNG. Отдельно измерять explicit request, stock auto-switch, actual equip delay, release fire и damage/ammo по каждому оружию. Провести paired evaluation на новых seeds с одинаковыми loadouts, подтвердить пользу оружия и отсутствие MG100/Solo regressions. Затем расширять оружия, типы монстров/геометрию/вертикальные действия и кооп; R5–R9 не завершены.
