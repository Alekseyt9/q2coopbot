# Сохранение critic и Adam при небольшом изменении награды

Текущий v8/v9 A/B использовал одинаковый сброс critic output и обоих Adam
состояний. Получено61/80 против67/80 побед; преимущество над сильным
unchanged v8 parent не установлено. Пока идет отдельная400-бойная оценка,
подготовлен альтернативный режим. Гипотеза: сохранение обученной оценки
состояний и оптимизаторов может уменьшить потерю прежнего поведения.
Причина изменения качества текущего A/B этим не доказана.

`fork_combat_architecture_objective_cuda.py --preserve-critic-optimizer`
сохраняет actor, std, critic, оба Adam states, consumed rollout history и
счетчики обновлений. Меняется только явный objective config hash checkpoint.
Critic сначала предсказывает возврат прежней награды; дообучение на новой
награде необходимо. Все остальные hyperparameters обязаны совпадать.
Смена архитектуры/наблюдений/discount этим режимом не разрешается.

## CUDA проверка

`workspace/artifacts/native-waste-warm-fork-smoke-v1-20261010/report.json`
закрыт. Три режима прошли GPU forward и read-back checkpoint:

| Режим | Critic/Adam | Weights SHA256 |
|---|---|---|
| warm control v8 | сохранены | 211c2d6d8062620a983800b81b18320452aaba4e0a58ebd80c10270ecf2bd27b |
| warm quality v9 | сохранены | 211c2d6d8062620a983800b81b18320452aaba4e0a58ebd80c10270ecf2bd27b |
| default reset control v8 | сброшены | 11bbb789f6b2530e91171f0735629c4a7c6d4b14ca329f941c2e66ad8402469e |

Warm файлы весов побайтно равны сильному родителю. Actor/value/std tensors
и вложенные optimizer tensors после CUDA чтения совпали с сохраненными;
warm Adam states также совпали с родительскими. Default reset файл совпал
с прежним reset control fork. Это проверка переноса, не обучение и не
доказательство улучшения боев. CPU/Go численных NN проверок не было.

## Общая очередь A/B

Добавлен `run_combat_action_quality_ab.py --shared-training-pool`:
оба frozen actor корпуса передаются одной160-бойной очереди на16 slots.
После ее закрытия `process_combat_architecture_pool.py --model-id`
отдельно выбирает control и quality, проверяет исходный общий pool receipt,
перенумеровывает только локальный plan index и применяет свой objective/config.
Данные разных наград в одном PPO batch не смешиваются.
Существующий collector сжимает только завершенные member files с SHA проверкой.
Общий root содержит обе branch paths, что сохраняет проверку containment.

Оба флага опциональны. Python синтаксис проверен; GPU fork проверен живым
CUDA прогоном. Новая A/B интеграция shared pool еще не запущена в игре.
Предварительный следующий эксперимент — warm v8/v9 из общего parent,
свежие train offsets344..347,80 боев на ветку, затем common development eval.
Его выбор зависит от закрытой400-бойной проверки текущего v9 кандидата;
одновременно второй16-instance pool не запускается.

## Закрытая оценка и следующий запуск

`native-waste-parent-rules-eval-v1-20261010/progress.json` подтверждает
400 закрытых боёв и `diagnostics_complete=true`. Сильный parent выиграл
70/80 и67/80, v9 кандидат67/80 и69/80; смерти9/7 против11/8.
Правила выиграли64/80. Преимущество v9 над сильным parent не доказано;
публикуемые веса и человеческая сессия не изменены.

Запущен `workspace/artifacts/native-waste-warm-ab-v1-20261010`:
80 train боёв на ветку,20 семейств, offsets344..347;
затем80 common development боёв на ветку, offsets28..31,
stochastic sampling offset20261011. Один общий16-slot pool, timescale2.
Флаги `--preserve-critic-optimizer --shared-training-pool` включены.
Обе CUDA fork проверки прошли: critic/Adam сохранены,
checkpoint read-back exact; исходные weights SHA равны parent211c2d6d….
Обучение/оценка на момент записи еще не завершены.

Этот опыт проверяет v8/v9 при сохранении состояния обучения. Он не
изолирует влияние warm против reset: прежний reset опыт использовал другие
train seeds. Даже победа в A/B требует отдельного сравнения с сильным parent
и правилами; development бои не являются финальным отложенным тестом.

Для последующей проверки `run_combat_native_candidate_comparison.py`
принимает явный `--candidate-branch control|quality` (default quality).
Путь checkpoint обязан соответствовать выбранной закрытой ветке.
Это позволяет проверить и v8 control, если именно она покажет лучшие
результаты. До завершения A/B следующая оценка не запускается.

Общий training pool закрыт:160/160 jobs, ошибок0,
`source_unchanged=true`, receipt SHA
`9a9c7ee79ed0d462a95c3ef641aa3e9f694f375bce55837c452ad76dcefd8460`.
Сжатие160 завершенных members закрыто с проверкой сохранения SHA потоков.
`training-comparability.json` подтверждает80 пар условий и одинаковые
fork weights. Control processing начался с native-only экспорта20
семейств и привязан к исходному общему pool receipt; последующее численное
восстановление rollout/value/logprob и PPO выполняются на CUDA.
Полное shared-pool обучение и оценка на момент этой записи еще не закрыты.

Control CUDA finalize batch закрыт:20/20 корпусов,
`state=complete`, `device=cuda`; request SHA
`39e1e010128a0c3ccfe329f04e6a51ffb68bd606f63a964af49834cf0ad3fc93`.
Это завершение восстановления обучающих данных, не завершение PPO update.

Control processing затем закрыт:CUDA update6,4222 eligible transitions,
10 actor steps. Веса SHA
`385c10bd74209d13b71ea1da37338c9cbea0fffb39ca18bd1aad5ab0475c7813`,
checkpoint SHA
`00545c50d001136940c4d5889f8fa9bbfe2bee8474ee7fa0eff96d79a08dae9c`.
`complete.json` связывает веса, checkpoint и report; началась подготовка
quality ветки. Итоговый отдельный CUDA checkpoint audit и оценочные бои
еще не завершены; эти training числа не подтверждают улучшение политики.

Quality CUDA finalize также закрыт20/20, request SHA
`b216a3155b5bb8e546359e534496c9bf36d2c8571f49f41b191fdda062abb26a`.
Обе processing protocol записи имеют одинаковые `python_sources` SHA
и общий `pool_sha256`; выбранные binding/plan indexes различаются
соответственно control/quality. Quality дошла до `cuda-update`.

Обе ветки обучения закрыты; quality CUDA update6 содержит4281 eligible
transitions и10 actor steps. Quality weights SHA
`ff6036d472bd5ddd917dcff33d82240166e6b538d139d454cfe9710288736580`,
checkpoint SHA
`cf554e6d224645afb9439f8717c58f7c40909071fe51b1fbf4754ea5d020abfb`.
Для обеих веток отдельный CUDA checkpoint audit подтвердил
`actor_value_std_exact=true`, `optimizer_state_exact=true`.
Driver перешел к `paired_validation`,160 боёв в общей16-slot очереди.
Shared-pool сбор и раздельное GPU обучение теперь проверены живым запуском;
оценочное качество и финальная диагностическая приемка еще не закрыты.

Все160 evaluation captures закрыты без ошибок и с неизмененными исходниками.
Core quality report:control66/80 побед,9 смертей;quality63/80,12 смертей;
received damage16.95 против20.1875, outgoing75.925 против75.2625.
Пропусков capture frames0 в обеих ветках. Финальные native shot/movement
диагностики еще выполняются. Эти результаты не дают основания для
promotion quality ветки; отдельный сильный parent ранее70/80 в RNG arm a,
но финальная диагностическая приемка текущего A/B еще необходима.

## Финальный результат

Driver завершился exit0. `progress.json` закрыт, result SHA
`240940742210f4c3ef4adf50c23959e2abd2ffc40dc5ac73b2808b6329a986f0`.
Отдельно перечитаны оба CUDA checkpoint seals и все native report SHA;
`ownership_clean=true`. Полный warm A/B завершен, promotion отсутствует.

| Метрика | Warm v8 control | Warm v9 quality |
|---|---:|---:|
| Победы |66/80|63/80|
| Смерти |9|12|
| Machinegun попадания по живым монстрам |49.0%|43.6%|
| Machinegun confirmed waste |348/694 shots|419/752 shots|
| Без выбранной/видимой bbox цели: hits/shots |1/206|1/272|
| Blaster попадания |56.4%|59.7%|
| Provider held attack стоя |18.5%|19.6%|
| Средний абсолютный yaw команды /frame |10.31°|9.50°|
| Median первый native выстрел после видимой цели |0s|0s|
| Max первый native выстрел |4.1s|4.6s|

В обеих ветках native fire зарегистрирован79/80; один бой без выстрела
не подменен нулевой задержкой. Парные переходы:2 приобретенные победы,
5 потерянных. Меньше yaw и лучше Blaster accuracy не компенсируют
потери побед, рост смертей и ухудшение Machinegun. Quality ветка отклонена
для promotion. Это не доказывает бесполезность miss penalty вообще;
результат относится к этим coefficients, бюджету и parent.
Warm-vs-reset причинное сравнение не проводилось (разные train seeds).

Следующий опыт:из сильного v8 parent увеличить долю свежих training
mixed/group сцен, reward оставить v8. Сохранить retention сцены и общую
оценку всех20 семейств; читать actual native shot/damage/kill и движение.
Не обучать на validation captures из диагностического разбора. Новые
веса принимать только при улучшении относительно сильного parent и
без заметной деградации ранее освоенных сцен.
