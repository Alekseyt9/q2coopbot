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
