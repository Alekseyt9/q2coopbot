# Цикл PPO и восстановление checkpoint

05.10.2026. Следующий шаг R4b: четыре свежие итерации и отдельная оценка. Это пилот фиксированного Blaster, а не полноценный обученный боец или обучение оружию.

## Продолжение состояния обучения

`scripts/ppo_combat.py --resume` восстанавливает actor/value Adam, CPU RNG и доступные CUDA RNG после benchmark. Проверяет соответствие checkpoint экспортированным JSON actor/value/log_std, SHA весов, конфиг и отсутствие rollout SHA в списке уже потреблённых batch. Загрузка `torch.load(weights_only=True)` ограничена tensor/primitive checkpoint.

`combat_ppo_checkpoint_v2` сохраняет weights SHA, consumed_rollouts, updates_completed и total_actor_steps. Старый checkpoint первого PPO-пилота мигрирует один раз с проверкой соседнего report.json, который фиксирует SHA весов и предыдущий rollout. Resume не меняет начальные параметры распределения: используются log_std текущих весов. Actor line search сохраняет и восстанавливает также состояние Adam при отклонении шага.

Тест resume сравнивает следующий шаг восстановленного Adam с непрерывным обучением и проверяет отказ повторного rollout и чужих весов. Четыре теста GAE остаются. Конфиг gamma/lambda/clip/lr/steps сохранён из первого пилота без настройки по evaluation.

## Оркестрация существующего харнеса

`scripts/run_combat_ppo.ps1` вызывает прежний baseline runner: четыре server/client worker, x2, отдельные seed, cold runtime на каждый эпизод. Каждая итерация: frozen behavior weights → свежий stochastic batch → `q2ppo-data` с повторной native-проверкой → offline update с checkpoint resume → следующий batch новой версии.

Конфиг и trainer SHA проверяются в ходе цикла; trainer копируется в artifact. Обновления пишутся в новые каталоги, progress.json сохраняет завершённые стадии. Ошибка сбора, native replay, resume или KL останавливает цикл с сохранением данных. Worker cleanup остаётся у существующего runner. Никаких загрузок пакетов или изменений пользовательского live runtime.

После всех обновлений отдельно выполняются deterministic before/after на одинаковых evaluation seeds. Они не пересекаются с training seeds и не используются в PPO update. `fixture_promotion_eligible` требует native first-life kill и отсутствие первой смерти во всех четырёх валидных after-эпизодах. Изначально дополнительно требовался legacy harness acceptance; этот rules-specific критерий исправлен после второго цикла, подробности в [отчёте второго цикла](learned_combat_ppo_cycle_v2.md). Это узкая проверка fixture, не статистическая generalization/боевой приёмка и не автоматическое включение в live. Продолжать тренировать candidate можно и при false.

Первый цикл: `workspace/artifacts/combat-ppo-cycle-20261005-230526-428`. Начальные веса/checkpoint — `combat-ppo-update-v1-20261005-r1`, четыре новых итерации, training seeds 13600–13615, evaluation 13700–13703, 300 игровых кадров. Версии предыдущих файлов не перезаписываются.

## Почему секунды update не означают обученного бойца

У текущего пилота маленькая 386→64→64 сеть и сотни переходов в batch. Секунды в trainer report относятся к оптимизации, отдельно от запуска Python, сбора игровой практики, native replay и оценки. Успех определяется игровыми результатами, а не временем шага optimizer. Пока бой не освоен; неизвестны необходимое число итераций и время до устойчивого результата.

Для масштаба: [OpenAI Five](https://openai.com/index/openai-five-defeats-dota-2-world-champions/) обучали 10 месяцев, накопив около 45 000 лет self-play; использовался PPO. Это долгие командные матчи с большим объёмом опыта, а не аналог нашего единственного короткого Parasite fixture. Малый пилот проверяет корректность контура до расширения задач; он не обещает обучить Quake-бота за несколько секунд.

## Результат первого цикла

| Итерация | Fresh transitions | Actor шаги | Final KL | Value MSE своего batch | Update, с |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 854 | 10 | 0.009238 | 0.182847 | 0.282 |
| 2 | 660 | 10 | 0.007777 | 0.071703 | 0.275 |
| 3 | 845 | 10 | 0.006049 | 0.049928 | 0.265 |
| 4 | 896 | 10 | 0.006959 | 0.046762 | 0.437 |

Собрано 3255 новых PPO-переходов; 40 новых actor и 160 value шагов. Финальный checkpoint: updates_completed=5, total_actor_steps=50, пять consumed rollout SHA с учётом первого пилота. Первое продолжение мигрировало legacy checkpoint, следующие три использовали v2. Во всех четырёх report указан resume SHA. MSE относится к разным batch/target, её снижение не доказывает улучшение одинаковой оценки.

Веса: `iteration-4/update/weights.json`, SHA256 `ba3714cf7f8d49d4ceedc07bb402346e10a0bf37ca7e848739f28773eb5b4f0f`; checkpoint рядом. CPU выбран в каждой итерации: benchmark 3.15–3.44 мс/actor-step, CUDA 5.94–7.49 мс. Сама оптимизация суммарно 1.26 с; сбор, replay, Python startup и восемь evaluation-эпизодов сюда не входят. Root trainer.py — точная frozen byte-копия скрипта с SHA из report; дополнительные per-update копии первоначально могли иметь нормализованные переводы строк, последующий код сохраняет их побайтово.

### Отдельная deterministic evaluation

| Seed | Урон врагу до / после | Полученный урон до / после | Конец первой жизни до / после |
| --- | ---: | ---: | --- |
| 13700 | 40 / 30 | 100 / 100 | смерть / смерть |
| 13701 | 30 / 30 | 100 / 100 | смерть / смерть |
| 13702 | 20 / 30 | 17 / 13 | конец трассы / конец трассы |
| 13703 | 20 / 30 | 100 / 7 | смерть / конец трассы |

Первая смерть: 3/4 до, 2/4 после. Native kills 0/4 и gameplay acceptance 0/4 в обоих режимах. Сравнение выполнено на четырёх парах одной геометрии: оно не подтверждает статистически устойчивое улучшение или перенос. `fixture_promotion_eligible=false`, автоматического переключения live нет.

Всего 24 capture/provenance-valid эпизода: 16 training и 8 evaluation. Повторные native replay и behavior logprob/value проверки прошли для всех четырёх batch. Source/native fingerprints и условия обеих eval совпали; full-world reset equivalence остаётся неподтверждённой. Общий `go test ./...`, пять Python GAE/resume tests и PowerShell syntax check прошли. Собственные процессы завершены, порты 33100–33103 освобождены; пользовательский live runtime прежний.

```powershell
./scripts/run_combat_ppo.ps1 -Model workspace/artifacts/combat-ppo-cycle-20261005-230526-428/iteration-4/update/weights.json -Checkpoint workspace/artifacts/combat-ppo-cycle-20261005-230526-428/iteration-4/update/checkpoint.pt -Iterations 4 -Seed 13800 -EvalSeed 13900
```

Команда была выполнена во [втором цикле](learned_combat_ppo_cycle_v2.md): ещё 3506 переходов, диагностика остановок и прицела. Не повторять эти seeds как свежие. Перед существенным увеличением объёма нужно измерить причины попадания в стены/потери цели и качество самостоятельного движения, затем разнообразить стартовые условия. Наличие resume позволяет накапливать опыт; количество проходов само по себе не гарантирует обучение полезному бою. Голова оружия и RL с нуля по-прежнему отдельные незавершённые этапы.
