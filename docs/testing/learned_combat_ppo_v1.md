# Первый PPO-Clip update

05.10.2026. R4b частично реализован: Go stochastic actor/value, свежие rollout, проверка поведения и один offline PPO update. Боевой приёмки и обучаемого выбора оружия пока нет. Формулы clipping и on-policy сбор сверены с первичным описанием [OpenAI Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

## Политика и исполнение

`combat_ppo_v1`: actor — прежняя MLP 386→64→64→8, инициализированная BC Blaster v2; value — отдельная MLP 386→64→64→1 с начальным нулевым выходом. Четыре непрерывных pre-tanh действия выбираются из Normal с обучаемым log_std; attack — Bernoulli; vertical — Categorical release/jump/crouch. Оружие фиксировано Blaster. Начальные log_std [-2.3,-2.3,-5.3,-5.3] задают малый разброс прицела и более широкий разброс движения.

Go делает inference и sampling, отправляет команды через прежний исполнитель. Python не управляет игрой и не реализует второй харнес. Серверная награда остаётся отдельным offline-сигналом. Source наблюдений и 386 признаков не получают seed, log probability, value или серверные эффекты.

Selection/Step сохраняют `sample`: исходный pre-tanh latent, выбранные attack/vertical, log probability, value, sampling_seed и policy version. Sampling seed отдельный для каждого worker/эпизода и равен его зафиксированному seed; Go RNG независим от native engine RNG. Версия весов — SHA канонической структуры actor/value/log_std без sampling_seed и deterministic, поэтому разные RNG-потоки не создают ложные разные версии одной модели. Полный SHA каждого provider-файла отдельно фиксируется manifest.

PPO ratios используют likelihood исходного latent: Jacobian фиксированного tanh-преобразования сокращается в отношении. Entropy-регуляризатор относится к latent Normal и двум дискретным распределениям. Quantization, pitch clamp и guards — часть перехода среды; их применённое действие не подставляется вместо sample в log probability. Setup/death/noncombat остаются у rules и исключаются из PPO loss. Смена оружия не обучается.

## Проверка и подготовка batch

`cmd/q2ppo-data` принимает только completed/provenance-valid synchronous learned PPO batch с frozen моделью. Проверяет SHA original/episode provider, sampling seed, одну версию весов и stochastic mode. Запускает исходный batch exporter после проверки его SHA, повторно доказывает native execution/reset/reward и сравнивает steps/rewards/server_outcomes с первоначальным экспортом. Дополнительно Go пересчитывает предложенное действие, log probability и value каждого sample. Input hashes проверяются повторно перед завершением.

В обучение включаются только provider-owned переходы первой жизни с reward и next; смерть сохраняется как terminal с нулевым bootstrap. GAE: gamma 0.99, lambda 0.95; recurrence разрывается на gap, смене seed, terminal и truncated. При доступном next и нетерминальной границе value используется для bootstrap, последующие правила не входят в recurrence. Хвост с отсутствующим next/reward не получает выдуманную награду. Это ограниченный гибридный pilot с передачей управления rules, а не непрерывное управление всей картой.

Начальная модель: `workspace/artifacts/combat-ppo-initial-v1-20261005/weights.json`, SHA256 `870101522d5f4677413061be9dd541d286362b36877d98dd6aa0e6cd03f845c2`. Policy `ppo:b2e21400cd0b39d721703c8aa6036dfb67462d5eb675f1b2e3a47cffa79ecdf6`.

Свежий batch: `learned-combat-baseline-20261005-225056-290`; 4 инстанса × 1 эпизод, x2, 300 кадров, seed 13400–13403, base1/Blaster/один Parasite. Capture/provenance 4/4. До обновления: урон врагу [20,30,30,30], три первые смерти, одна жизнь дожила до конца трассы; это sampling исходного BC, ещё не результат PPO.

Проверенный rollout: `workspace/artifacts/combat-ppo-rollout-v1-20261005`, 797 переходов, 3 terminal, 266 неиспользованных строк (rules/последующие жизни/неполные переходы). Rollout SHA256 `1597c2e061f3a620115f5d6d3f211dba1a653210efda076fbfea3de362603240`. Исходные старые BC/teacher JSONL не использовались как on-policy опыт.

## Update и контроль шага

`scripts/ppo_combat.py` и `scripts/scenarios/combat-ppo-v1.json`: full-batch PPO clip 0.2, Adam actor lr 0.0003/value lr 0.001, максимум 10 actor + 40 value шагов, target KL 0.01, entropy 0.001, max grad norm 0.5, seed 20261005. Конфиг зафиксирован до сбора. Нет выбора по held-out игровым результатам.

Первое выполнение `combat-ppo-update-v1-20261005` выявило дефект контроля: проверка KL только перед очередным шагом оставляла слишком большой уже выполненный первый шаг (KL 100.95). Эти веса не запускались в игре. Артефакт сохранён как отклонённый proposal. Это исправление алгоритма на training batch, а не подбор по test.

Теперь каждый actor proposal проверяется после шага; при превышении KL, nonfinite или ухудшении clipped objective восстанавливаются actor/std/optimizer и шаг уменьшается вдвое, до 12 уменьшений. Если приемлемого шага нет, восстановленная политика сохраняется и обновление останавливается. Limiting KL по этому batch не гарантирует безопасность во всех состояниях.

Исправленный update `workspace/artifacts/combat-ppo-update-v1-20261005-r1`: 10 actor шагов, 59 уменьшений шага, 0 окончательных отклонений, final approximate KL 0.00547773, value MSE 0.579666. PyTorch/Go old-logprob max error 0.00024223, old-value error 0; проверены все 797 samples. Веса SHA256 `535b025537c28b8b44918e034866cc50aeb86afb383fe55161fea0341a1cfc26`.

CPU benchmark actor-update 2.682 мс/шаг, RTX 5070 CUDA 5.852 мс/шаг: выбран CPU. Actual update с backtracking/value занял 0.256 с без запуска Python, benchmark и экспорта. Это измерение маленького batch, не общий вывод о GPU. Использован существующий PyTorch 2.10.0+cu128 на F, новые пакеты не скачивались.

Checkpoint содержит actor/value/log_std, optimizer states и CPU/CUDA RNG. При первом пилоте CLI восстановления optimizer ещё не было; этот отчёт фиксирует один update с новым optimizer. В следующем шаге добавлены [resume и цикл свежих batch](learned_combat_ppo_cycle_v1.md). Сохранена точная копия первоначального trainer.py и его SHA; гиперпараметры, source hashes и training report лежат рядом.

Тесты: независимые/повторяемые RNG-потоки, неизменность weight version при смене sampling seed, пересчёт sample likelihood и действия, отказ чужой версии; Python GAE terminal, truncation bootstrap, gap и последовательные переходы.

## Оценка до/после

Детерминированные копии before/after созданы в `workspace/artifacts/combat-ppo-evaluation-v1-20261005`. Sampling отключён только для оценки; эти JSONL запрещены как PPO training data. Сохранён отдельный полный SHA каждого eval-provider, sampling_seed по-прежнему указан для provenance.

Batch after: `learned-combat-baseline-20261005-225454-357`; before: `learned-combat-baseline-20261005-225652-030`. Два последовательных запуска по четыре инстанса, x2, 300 кадров, отдельные seed 13500–13503 с повтором между режимами. Source/native fingerprints и условия совпали; capture/provenance/dispatch 8/8. Скрытая full-world эквивалентность reset всё ещё не доказана. Проверено отклонение чужих behavior weights и deterministic eval в `q2ppo-data`.

| Seed | Урон врагу до / после | Полученный урон до / после | Конец первой жизни до / после |
| --- | ---: | ---: | --- |
| 13500 | 50 / 60 | 100 / 100 | смерть / смерть |
| 13501 | 20 / 20 | 17 / 100 | конец трассы / смерть |
| 13502 | 100 / 30 | 100 / 100 | смерть / смерть |
| 13503 | 20 / 20 | 100 / 13 | смерть / конец трассы |

По три смерти до и после, native kills 0 в обоих режимах; gameplay 0/4 до и после. Selection p99 ≤1088 мкс, ниже текущего бюджета 5 мс. Четыре пары не устанавливают статистическое улучшение; одиночный update не решает бой. Новый checkpoint не включён в пользовательский live runtime. Собственные процессы завершены, порты 33100–33103 освобождены.

Сравнение с SHA исходных report: `workspace/artifacts/combat-ppo-evaluation-v1-20261005/comparison.json`. Общий `go test ./...` и четыре GAE-теста прошли. Всего за этот этап три batch, 12 capture-valid эпизодов: 4 stochastic training + 8 deterministic evaluation.

```powershell
# Fresh directory for each stage; repeat collection for every accepted version.
go run ./cmd/q2ppo-data --batch workspace/artifacts/learned-combat-baseline-20261005-225056-290 --model workspace/artifacts/combat-ppo-initial-v1-20261005/weights.json --out workspace/artifacts/ppo-recheck
& F:/src/strat/.venv-gpu/Scripts/python.exe scripts/ppo_combat.py --model workspace/artifacts/combat-ppo-initial-v1-20261005/weights.json --data workspace/artifacts/ppo-recheck --config scripts/scenarios/combat-ppo-v1.json --out workspace/artifacts/ppo-recheck-update
```

Следующая работа: управляемый цикл нескольких свежих batch со восстановлением optimizer/checkpoint, отдельные eval seeds и правила выбора принятой модели. Повторное использование этого batch с новым behavior model запрещено проверкой SHA. Голова выбора оружия, inventory mask и сравнение с RL с нуля требуют следующего эксперимента; демонстрации не обязательны для RL.
