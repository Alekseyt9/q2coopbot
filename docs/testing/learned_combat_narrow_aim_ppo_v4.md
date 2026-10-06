# Более узкая exploration прицела в Mixed PPO

06.10.2026. Продолжение [Mixed PPO после maneuver curriculum](learned_combat_maneuver_mixed_ppo_v4.md). Parent: `combat-ppo-maneuver-mixed-v4-20261006/iteration-4/update`, weights SHA `f55325f08d1185bfe2f5998c658470b18feef2e38832c6f549e3fdad6b20cdfe`, checkpoint SHA `1ac4ec64271e51fc870f9086300a921e8e98bb9e2a8d25c87ea6d029f270a451`,53 PPO updates/524 actor steps.

Гипотеза, а не установленная причина: stochastic angular noise мешает получать kill transitions. Фактический parent yaw/pitch log_std≈-3.9014, latent std≈0.0202. Вблизи нулевого latent mean `180*tanh(z)` даёт приблизительно3.64° angular std. `scripts/fork_combat_exploration.py` создаёт отдельную ветку с yaw/pitch log_std=-5.3, latent std≈0.004992/near-zero angular std≈0.899°. Это уменьшение stochastic noise примерно в4 раза. **Mean actor/value tensors и движение сохраняются**, actor Adam reset, critic Adam/RNG/consumed history/counters retained. Нет runtime aim correction или изменений Go guards. Это distribution fork, не optimizer update. Его deterministic policy совпадает с parent.

Fork root `workspace/artifacts/combat-narrow-aim-v4-fork-20261006`, weights SHA `06d7507ffd3e05ba45397ce2fd903d3626d49fc4702cf33a46318b61eb298245`. Training config/reward v4/objective/gamma0.99 и architecture810→64→64 прежние. Только новые on-policy rows могут обучать fork; старые rollout distributions не подходят.

Протокол до результатов: fixed4 PPO updates, training18100–18115, paired deterministic initial fork/fixed fourth model Mixed18200–18203. Дополнительно paired solo stock175HP/release100 на18300–18303. Каждый batch4 независимых server/client instances x2,300 game frames; Mixed stock175HP Parasite+Gunner/release0/Blaster. Eval не используется для fit или выбора best checkpoint, live promotion отсутствует. Root `workspace/artifacts/combat-ppo-narrow-aim-mixed-v4-20261006`.

Для проверки начального distribution effect дополнительно запланирован **stochastic wide-parent control** на18100–18103 с той же mean policy, seeds и условиями, сравнимый с first narrow training batch. Wide control **не используется для PPO**. Это отдельная diagnostic paired проверка исходного шума, не held-out acceptance; full-world reset equivalence не доказана. Дальнейшие PPO updates меняют mean policy, поэтому их результаты сами по себе не изолируют эффект изменения std.

## Завершённая проверка после обслуживания диска

Четыре updates завершены: **2330 fresh PPO rows, 40 actor steps**, итоговые
счётчики **57 updates / 564 actor steps**. По итерациям 341, 614, 513 и 862
перехода. Из-за заполненного диска `iteration-3/batch` не дал usable capture;
использован только `iteration-3-retry`. Повторный native export всех четырёх
batch побайтно совпал с consumed rollout. Проверены parent weights/checkpoint,
Adam/RNG resume, уникальность rollout, objective/gamma и KL (0.00444–0.00659).
Все updates использовали CPU по замеру: CUDA на этих малых batch медленнее
примерно в 1.7–2.5 раза. Это не отменяет применения GPU для больших supervised fits.

Финальные веса SHA256:
`6183518c866e894933ae03ba24a26ca3528628d27ec8940ef785fe2833d86373`.
Checkpoint SHA256:
`a2861de16afc86fd4957632130ec0d3146a1ccdc88f35418d90a68a62cdc5e0e`.

Прежние evaluation были прерваны обслуживанием и не используются в итоговом
сравнении. Все шесть свежих evaluation batch находятся в
`workspace/artifacts/combat-narrow-aim-v4-eval-20261006`: 24 native capture,
4 server/client instances одновременно, x2, 300 кадров, независимые сиды.
У всех batch одинаковы текущие source/native fingerprints; парные проверки
подтверждают остальные условия. Eval и noise controls не участвуют в fit.

| Проверка | До | После |
|---|---:|---:|
| Mixed 18200–18203: полных побед без смерти | 0/4 | 0/4 |
| Mixed: native kills | 3 | 1 |
| Mixed: смерти | 4 | 4 |
| Mixed: исходящий / входящий health damage | 925 / 400 | 665 / 400 |
| Solo 18300–18303: убийства / смерти | 4 / 0 | 4 / 0 |
| Solo: исходящий / входящий health damage | 700 / 163 | 700 / 178 |

Все kills принадлежат learned provider. В Mixed нет rules handoff первой жизни.
В Solo handoff во всех эпизодах происходит **после убийства**, на следующем кадре.
На seed 18201 исходная Mixed policy убила обоих, но затем умерла — это не победа
без смерти. Сравнение с прежними 2/4 победами на17900–17903 не является парным:
там другие сиды, поэтому изменение 2/4→0/4 нельзя приписывать этому update.

В Mixed ошибки bbox >15° равны0/432 до и0/296 после. Это геометрический proxy,
не измерение точности выстрелов. Static hull interventions131→19, неподвижность
при предложенном движении130→19, но provider first-life steps433→296: более
короткую жизнь нужно учитывать при сравнении абсолютных счётчиков. Средняя
наблюдаемая дистанция до Parasite208.2→215.0; почти все кадры внутри288.
Уменьшение блокировок не стало улучшением завершения боя.

Offline attribution входящего урона по классам проверена против суммы first-life
health damage: Gunner150→191, Parasite250→209. Файл
`incoming-threat-audit.json` содержит также noise controls. Это показывает
необходимость учитывать обе угрозы; данные server truth не добавлены во вход
policy. Дальнейшие выводы о конкретном способе уклонения требуют отдельного опыта.

## Контроль самого angular noise

Первоначальный wide-control не завершён до обслуживания. Вместо сравнения
с capture на другой версии подготовки выполнены **две свежие пары batch**
`noise/evaluation-before` (wide parent) и `noise/evaluation-after` (initial narrow
fork) на18500–18503. Это stochastic диагностические эпизоды, не deterministic
acceptance и не training. Mean actor/value совпадают точно; movement log_std
совпадает в float32 (JSON-fork сохраняет ничтожное округление исходных двух
float64 значений); angular log_std меняется с примерно-3.9014 на-5.3.

| Noise control | Wide | Narrow |
|---|---:|---:|
| Полные победы без смерти | 0/4 | 0/4 |
| Убийства / смерти | 0 / 4 | 0 / 4 |
| Исходящий / входящий health damage | 470 / 400 | 440 / 400 |
| Bbox errors >15° / samples | 1 / 323 | 0 / 232 |

На этих четырёх сидах польза сужения шума не подтверждена. Четырёх эпизодов
недостаточно для статистического вывода о любом составе, карте или policy.
Training содержит7 provider kill reward rows, все7 вошли в PPO, но сравнение
с1 kill прежнего wide-training на других сидах не изолирует эффект noise.

## Аудиты и решение

Независимый audit пересчитал aim/spacing potentials всех40 валидных training/eval
capture; максимальная ошибка компонента8.88e-16. Проверены все2330 consumed
reward scores, принадлежность kills provider, source SHA и seed exclusion.
Неудачные/прерванные capture исключены. Полная эквивалентность reset мира не
доказана; нет статистической генерализации или переноса в пользовательскую игру.

Доказательства: training root `ppo-lineage-audit.json`, `replay-checks/report.json`,
`shaping-audit.json`; eval root `summary.json`, `noise-contract-audit.json`,
`paired-audit.json`, `guard-audit.json`, `kill-ownership-audit.json`, `diagnostics.json`
и соответствующие файлы в `solo/` и `noise/`.

**Ветка сохранена как отрицательный результат и не принята вместо parent.**
Одиночный kill-навык сохраняется, но group fixture criterion не выполнен.
Следующий опыт должен адресовать групповое маневрирование, опасность второго
монстра и переключение между целями; повторять тот же PPO budget только ради
суженного angular noise оснований нет. Runtime Go/guards не менялись,
GRU/attention и обучаемый выбор оружия в этом опыте не добавлены.
