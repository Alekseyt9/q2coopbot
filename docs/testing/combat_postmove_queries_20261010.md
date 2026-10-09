# Прицел и перемещение одной команды: 10.10.2026

## Исправление предыдущего native аудита

В `audit_combat_blaster_muzzle_cuda.py` и `report_combat_blaster_hits.py` было неверное сопоставление по одинаковому numeric frame. RunFrame увеличивает server frame; событие F+1 внутри beginF/endF+1 относится к командеF, а не следующей. Теперь `native_projectile_window.py` связывает события по порядку begin/end, spawncount/actor/sequence, проверяет границы и незакрытые окна. Boundary regression проверяет событие на end boundary, malformed sequence, выход за frame window и незакрытый step.

Старые audit/hit JSON сохранены под `*-frame-join-v1-superseded.json`. Native quality/win reports не меняются. Исправленные показатели:

| Cohort | Вариант | Live monster blaster hits |
| --- | --- | ---: |
| Precision v1 | before | 100/142=70.42% |
| Precision v1 | eye after | 106/149=71.14% |
| Precision v1 | FireBC | 112/151=74.17% |
| Precision v1 | rules | 93/95=97.89% |
| Muzzle v2 | before | 100/142=70.42% |
| Muzzle v2 | muzzle after | 97/144=67.36% |
| Muzzle v2 | eye after | 97/142=68.31% |
| Muzzle v2 | FireBC | 100/142=70.42% |
| Muzzle v2 | rules | 94/94=100% |

На533 и658 соответственно выстрелах без terminal death orientation CUDA audit показывает mismatch0: native direction соответствует **новому** post-command angle, muzzle — **post-command** position; максимальная position error0.000817units. Средняя direction vector error≈0.0000005. Вывод о задержке прицела отменён. Движок ClientThink выполняет Pmove и обновляет origin/v_angle до Think_Weapon; fallback Think_Weapon в ClientBeginServerFrame также происходит после команды. Числа shot/hit не включают machinegun; неизвестные outcomes0 в этих cohorts.

## Абляция postmove-query v3

Exporter `--center-muzzle --post-move-labels` создаёт `observed_center_blaster_postmove_query_v3`. Учебная метка использует достигнутое перемещение и stance из следующего **клиентского** наблюдения. Из исходных observed enemy relative positions вычитается этот delta, затем решается center-muzzle intercept equation с исходной observed enemy velocity. Следующая enemy position не используется. Это offline corrective supervision, не информация, доступная агенту заранее.

**Future client position/stance находится только в labels (`query_move_label`, `query_ducked_label`), не в854 policy features.** GPU подтвердил точное равенство всех features и identity/seed порядков исходному corpus. Missing/dead/cross-life/gapped next contexts не дают меток;29train/10validation context frames masked. Сохранены все6268/1499 context frames. Known aim pairs3818/1052; Go serialized queries и CUDA feature-derived labels различаются максимум на0.0000153°.

Важное ограничение: displacement — фактический от prior-policy команды. Изменение aim меняет базис forward/side и способно изменить достигнутое движение. Эти labels не являются точным counterfactual rollout новой команды. Только paired native trial может подтвердить пользу; не объявлять сам dataset решением aim/movement coupling.

Обучен сильный FireBC before45,100epochs RTX5070; старые45rows/encoder/value/std фиксированы, новые36 fine/mode rows. Fine validation query RMSE8.24→6.25°,383 fine labels; mode accuracy63.59→64.83%. Эти RMSE не сравниваются напрямую с v1/v2: targets и subsets различаются.

Завершено80/80paired native боёв, строгий native/source/binary proof принят,16slots/x2: before81/postmove-after81/muzzle-after81/FireBC/rules,validation28,4families×4seeds. Final test отложен; baseline не заменён. Pipeline автоматически включает corrected ordered-window projectile hit report.

| Вариант | Победы | Смерти | Applied firing-ray error | Native blaster hits |
| --- | ---: | ---: | ---: | ---: |
| Before81 | 7/16 | 9 | 11.363° | 100/142=70.42% |
| Postmove after81 | 7/16 | 9 | 12.075° | 91/138=65.94% |
| Muzzle after81 | 7/16 | 9 | 11.861° | 97/144=67.36% |
| Legacy FireBC | 7/16 | 9 | target не объявлен | 100/142=70.42% |
| Rules | 8/16 | 5 | target не объявлен | 94/95=98.95% |

Unknown projectile outcomes0 у всех;16 before/after исходов совпали. **Postmove after не принят:** ни побед, ни hit fraction больше не стало. Меньшая offline fine-query ошибка не доказала пользу в native бою. Почему именно не перенеслось — не установлено; factual displacement и frozen encoder являются ограничениями эксперимента, не доказанными причинами.

Следующая архитектурная абляция: отдельная общая для enemy slots обучаемая aim-ветка с прямым доступом к observed range/relative vector/bbox angular features/velocity и выбранному движению. Это даст возможность учиться связи move→aim без зависимости только от старого замороженного encoder. Старые movement/fire/weapon/target outputs сохранить как control; новую ветку проверять на GPU и paired native hits/wins. Не добавлять будущие position или server shot outcomes в её входы. Сначала проверить сильный prior; не расходовать большие corpora всех архитектур до положительного контрольного результата.

Артефакты под `workspace/artifacts/`: `postmove-queries-v3-20261010/{report,cuda-query-audit}.json`, `precision-postmove-v3-20261010/{weights,report}.json`, `precision-postmove-eval-v3-20261010/`; process receipt `workspace/build/precision-postmove-eval-v3-20261010-process.json`.
