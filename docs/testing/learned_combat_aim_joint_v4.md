# Совместное обучение представления прицела с сохранением других команд

06.10.2026. Продолжение [aim-head v4](learned_combat_aim_head_v4.md): frozen encoder +2 fitted outputs не улучшили закрытый игровой цикл. Новый `--mode joint` в `scripts/fork_combat_aim.py` обучает весь actor810→64→64→8. Supervised aim targets прежние: observed nearest valid bbox angular errors, step±20°, latentatanh(step/180). Другие6 raw outputs distill из parent на тех же inputs, weight1.0. Loss=aim latent MSE+non-aim raw MSE. Это offline geometric supervision, **не PPO/RL без меток**, без server truth/human demonstrations/runtime corrections.

Ветка начата от сохранённого маневрирования: `combat-entity-types-v4-fork-20261006`, weights SHA `18f38979ec9028d3702662171b3bbb42bc99b16fa90d585ef437c1b45612d40f`, checkpoint SHA `bb9d3ce0c4d564bd8ef8677af48c1b2d78ba3719c367eadafba4fda8ea439327`, cumulative45 PPO updates/446 actor steps. Это явный возврат к сохранённой reference ветке, не продолжение failed54-update chain. Architecture/features/action range180*tanh/std/reward objective прежние. Critic output zero, оба Adam reset; parent RNG/history/counters retained. Все старые модели сохранены отдельно.

4437 verified training rows:1477 Mixed seeds16300–16315 +2960 isolated HP60 seeds16500–16515. Это данные состояний нескольких прошлых policies для supervised fit, не on-policy PPO. Eval16200/16400/16600/16700 не используются. Frozen source SHA/row count/feature version/duplicate rollout checks обязательны. Fixed1500 full-batch Adam steps lr0.0003, без выбора best epoch. Training-state retention gate: mean absolute normalized movement command drift≤0.05, attack probability drift≤0.05, mean vertical KL≤0.02. Gate не гарантирует сохранение closed-loop behavior.

Train bounded-command MAE yaw18.18→1.87°,pitch19.20→1.09°. Movement command MAE0.00178/0.00199, attack probability MAE7.95e-5, vertical KL2.49e-7, non-aim raw max error0.08373. Tensor layers encoder действительно изменяются; контрольные тесты не требуют impossible exact non-aim parity для joint. CPU9.33ms/step vs CUDA2.09ms/step; **выбран RTX5070 CUDA**. Fork weights SHA `f8f9639c82e3345c45a72084f77def87f1af9b599b9a5435c1d1adef34c77503`, root `workspace/artifacts/combat-aim-joint-v4-fork-20261006`.24 Python tests прошли, включая joint encoder change/retention/checkpoint/parent preservation и прежний head-only режим.

Протокол до игровых результатов: paired deterministic parent/fork isolated stock Parasite HP175, release100, held-out seeds16900–16903; paired parent/fork Mixed Parasite+Gunner stock HP, release0,seeds17000–17003. Каждый batch **4 independent instances x2**,300 game frames. Roots `combat-aim-joint-v4-solo-eval-20261006` и `combat-aim-joint-v4-mixed-eval-20261006`. Затем fixed4 fresh PPO updates от joint fork, isolated HP60/release100,training seeds16800–16815,4 instances x2; fixed fourth update vs initial joint fork на held-out HP175 seeds17100–17103. Root `combat-ppo-aim-joint-v4-20261006`. Reward v4/gamma0.99 сохранены; eval не обучает, не выбирает checkpoint и не меняет заданный бюджет. Не заявлены full-world reset/statistical generalization. Пользовательский live не заменяется.

| Stock HP175 solo seed | Outgoing parent → joint | Incoming parent → joint | Kills parent → joint |
|---|---:|---:|---:|
|16900|20 → 160|23 → 100|0 → 0|
|16901|20 → 175|25 → 69|0 → 1|
|16902|20 → 175|18 → 84|0 → 1|
|16903|20 → 175|14 → 95|0 → 1|

Итог первой solo проверки: outgoing80→685,kills0→3,deaths0→1,incoming80→348. **Убийства обычного175HP Parasite подтверждены под provider ownership**: available native kill reward+5 при frames249/269/269 для seeds16901/2/3. Handoff к rules только на следующем frame250/270/270 после убийства; поэтому это не rules-assisted kill. До убийств других handoff у этих after lives нет. У parent16900 было13 visibility/control handoffs, что ограничивает attribution его survival. У after16900 handoff0, observed death с160 damage/без kill. Критерий4/4 kill+no death пока не выполнен.

Provider bbox angular error>15° снизился562/570→0/644 attack samples; mean visible Parasite range356.22→184.29. Это существенное улучшение текущего observed прицела, но не measurement shot accuracy/lead. **Маневрирование ухудшилось**: parent держал дистанцию, joint чаще сближался. Non-aim distillation сохраняет локальные команды на training inputs, но другой yaw меняет их мировое направление; она не сохраняет физическую траекторию. Реальная survivor/kill проверка важнее training retention gate.

Mixed17000–17003: capture/provenance/native proof valid8/8, first-life handoffs0; outgoing parent30/30/30/30→joint60/60/130/90, суммарно120→340. Incoming400→400,deaths4→4,kills0→0. Bbox angular error>15°313/325→0/270 attack samples. Прицел улучшился и в Mixed наблюдениях, однако этого недостаточно для победы над группой. Подход к Parasite под огнём Gunner остаётся нерешённым.

После initial joint fork выполнены4 fresh HP60 PPO updates:1720 confirmed transitions (505/341/568/306),40 actor steps. **15 provider kills из16 training episodes**, все15 kill reward rows подтверждены native proof и включены в consumed PPO. Scalar kill component+5 сохранён; дополнительно доступны damage/aim/spacing компоненты. CPU быстрее CUDA на этих меньших on-policy batch (последний3.07ms vs5.28ms); GPU использовался для большего supervised fit. Final counters49 updates/486 steps этой reference ветки. Final weights SHA `0ea1f80bf0c46bc56195f6fbf16a760a6e4d3b8c795b5a6d3e8b66a377202c55`; checkpoint SHA `7f5474f6d650d6305fa95d13797bbee0f419b829ac88ab73e4ae5743c8a99424`.

| Held-out HP175 seed, initial joint → fixed fourth PPO | Outgoing | Incoming | Kills |
|---|---:|---:|---:|
|17100|175 → 175|82 → 50|1 → 1|
|17101|130 → 175|100 → 30|0 → 1|
|17102|130 → 175|100 → 25|0 → 1|
|17103|140 → 175|100 → 21|0 → 1|

Итог solo held-out17100–17103: **kills1→4,deaths3→0**, outgoing575→700,incoming382→126. Все4 final first lives дожили до trace end. Provider attack bbox angular error0/611→2/340 samples; mean observed horizontal range184.22→143.48. Не утверждать, что здесь научилось дальнее retreat: финальная policy быстрее добивает при более близкой дистанции. Native kills под provider ownership проверяются отдельно от последующего перехода к navigation rules. Fixture4/4 kill+no death criterion пройден на этой четвёрке; это ограниченный pilot на base1, не универсальная combat acceptance и не automatic live promotion.

Независимые paired audits проверили source/native fingerprints, seed/dispatch/provenance и owner segments; reward audit всех24 PPO training/eval captures проверил каждый available aim/spacing component и score1720 consumed rows (max error6.7e-16),15/15 training kill rows included.24 Python tests прошли. Пользовательский live сохранён.

До дополнительной проверки зафиксирован supplemental transfer test: initial joint vs fixed final PPO на **новых Mixed seeds17200–17203**,stock HP,release0,4 instances x2,300 frames. Root `workspace/artifacts/combat-aim-joint-v4-transfer-eval-20261006`. Это сравнение переноса после isolated PPO; оно не меняет checkpoint/бюджет и не используется для обучения/выбора модели. Успех solo не распространяется заранее на группу.

Supplemental Mixed transfer завершён: capture/native proof/provenance8/8, first-life handoffs0. Outgoing initial joint120/130/150/70→final50/40/40/40, суммарно470→170;incoming400→400,deaths4→4,kills0→0. Provider frames246→311,bbox angular error>15°1/233→5/80 applied visible attack samples. Финальный isolated checkpoint **не улучшил группу**.

Диагностика уточнила источник низкого outgoing: сеть запрашивала attack во всех246/311 provider frames, но attack guard `barrel_blast_risk` подавил13→231 команд; applied attack233→80. `static_hull_blocked` movement interventions183→228, unsupported motion27→27. Следовательно нельзя объяснять падение outgoing только потерей точности aim или отказом attack head: бот попадает в положения, где безопасный выстрел заблокирован близкой взрывоопасной бочкой, и часто упирается в геометрию. Guards сохранены, overrides записаны. Прицел без подходящего маневрирования/обхода препятствий недостаточен. Обстоятельства выводятся из capture interventions, без отключения защит или подмешивания rules aiming.

`kill-owner-audit.json` финального solo дополнительно подтвердил available native kill+5 под provider на frames183/183/179/179; navigation handoff только на следующих184/184/180/180. Всего48 capture этого этапа проверены по native provenance/dispatch и paired/owner conditions; независимый component/consumed-score reward audit охватывает24 PPO training/eval captures. Ни один held-out observation не использовался для fitting или PPO. Файлы исходной reference и неудачных веток сохранены, порты33100–33103 освобождены, live marker/config не изменены. Следующий этап — Mixed curriculum с явным контролем proximity/обхода препятствий и сохранением solo kill навыка; хороший прицел теперь есть, универсальная тактика для состава группы ещё не обучена.
