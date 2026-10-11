# Прямое обусловливание оружейной головы состоянием

Основание: температура2 сократила ожидание с неподходящим оружием, но на свежих учебных боях дала2452 запросов против160 и50,5% кадров подъёма/опускания против8,5%. Качество обученных control/exploration ещё проверяется на confirmation560; это основание для исследовательского прототипа, не решение о продвижении новой архитектуры.

Входы текущей v10-модели уже содержат gun_frame/30, инвентарь, текущий weapon one-hot, свой предыдущий запрос и его известный возраст. Проблема не в полном отсутствии этих признаков: ablation новых весов практически не показала их влияния на weapon-head. Общий total-policy KL преимущественно отражал движение/прицел.

GPU-прототип `scripts/combat_weapon_state_residual.py` обходит общий encoder для оружейного residual:154 входа →64 ReLU →32 ReLU →12 логитов. Он получает common20, observed inventory31, equipped/own-request26, мягко выбранные признаки текущей цели12+40, mean/max наблюдаемых врагов12+12 и вероятность none-target1. Контекст цели взвешивается существующим masked target head; не добавляются скрытые HP, серверный WeaponState или придуманный оптимальный weapon. Mask оружия, действия остальных голов и исполнитель сохраняют прежние контракты.

Последний слой инициализируется нулём. Пока residual не обучен, raw outputs должны совпасть с исходной политикой точно. После обучения меняются только weapon-логиты8:20; общий выбор цели может получать gradient через мягкий target context. Это устройство графа, не доказательство полезного обучения.

Прототип ещё не подключён к Go inference, PPO wrappers, сериализации моделей или миграции чекпоинтов. Он не публикует weights.json; текущая v10 runtime не должна молча игнорировать новую архитектуру. Результат отдельной CUDA graph-пробы фиксируется в `workspace/artifacts/combat-weapon-state-residual-prototype-probe-v1-20261011.json` после фактического закрытия. Проба использует реальные train histories исходного parent; случайная disposable perturbation проверяет связность градиента, не является обучением на демонстрациях и не сохраняет обученные веса.

CUDA graph-проба закрыта:13298 реальных кадров,12396 параметров; zero-init outputs совпали точно. После disposable random perturbation все неоружейные raw outputs также совпали точно. Проверены finite no-enemy inputs; все шесть наблюдаемых блоков дали ненулевой gradient. Состояние исходного attention воспроизведено с ошибкой<2e-5, хеши model/rollout/sequence проверены. Optimizer steps0, weights/checkpoints не публиковались. Это доказывает связность нового блока и сохранность выхода при инициализации, а не практическое использование признаков после PPO или улучшение боя.

На закрытом control-after PT выполнена отдельная CUDA-проверка схемы миграции:19 прежних actor/log_std параметров совпали с моделью и состоянием Adam по именам/формам; для6 новых тензоров (12396 scalars) выделены непересекающиеся ID нового optimizer group. Старый log_std остаётся в прежней группе, не сдвигается за добавленные параметры. Наличие собственного sequence objective требует явного fork. Receipt: `workspace/artifacts/combat-weapon-state-migration-layout-v1-20261011.json`; инструмент `scripts/prepare_combat_weapon_state_migration_cuda.py`. Это схема переноса, не опубликованный новый PT и не проверка его восстановления. Нативный LoadPPO использует DisallowUnknownFields: новая JSON-архитектура требует явного Go contract, иначе будет отклонена, а не молча проигнорирована.

## Экспериментальный PT сохранён и независимо проверен

`scripts/stage_combat_weapon_state_checkpoint_cuda.py` создал отдельный candidate из control-after. Файлы `workspace/artifacts/combat-weapon-state-candidate-control-v1-20261011/candidate-weights.json` и `candidate-checkpoint.pt` имеют отдельные имена и приватный checkpoint version: старый Go decoder отвергает новый residual field, старый PPO trainer —private checkpoint version. `ready_for_native_training=false` сохраняет эту границу явно.

На14049 реальных полных собственных control-after evaluation-кадрах CUDA sequence выводы исходной и расширенной политики совпали точно, state error<2e-5. Это проверка, не fitting на evaluation. Adam восстановлен с прежней группой параметров и отдельной residual group с lr0,0003, новыми нулевыми моментами/часами. Disposable zero-gradient Adam API step подтвердил восстановление двух групп; сохранённый PT не делал ни одного optimizer step. Config/history/critic/RNG/clocks сохранены, старый sequence objective архивирован для обязательного нового own-policy fork.

Независимый инструмент `workspace/artifacts/verify-weapon-state-candidate-persistence-20261011.py` заново загрузил обе пары сохранённых artifacts и подтвердил точное сохранение всех старых tensor/scalar/list/dict полей, включая critic и его optimizer state;6 новых actor tensors, нулевые новые моменты/steps, старые model fields без изменений. Receipt: `combat-weapon-state-candidate-persistence-v1-20261011.json`. Candidate seal: weights SHA `8dfac2306c1ed34676c5784186f5f65c09107ec516d5d9739496e6096a87bc45`, PT SHA `31292acaaf20b9e26f07b6bd89ba705c47cc1975853266949513f45fa57e7ad9`.

Нативная поддержка, стандартный trainer wrapper/export/restore и новое собственное обучение ещё не реализованы. Это промежуточный проверенный перенос, не готовая игровая модель и не основание для продвижения. Исходники работающей confirmation/CUDA chain не менялись.

После результатов текущей confirmation:

1. Уточнить критерий: recovery-switching быстрее без ущерба обычным боям; phase time и реальные native wins/damage, а не число attack intents.
2. Добавить явный model contract и Go residual inference, при сохранении полного управления движением/прицелом/атакой обучаемой политикой. Поддержка других архитектур проверяется отдельно, не предполагается по одному parent.
3. Внедрить Python actor wrapper и sealed CUDA checkpoint migration: старые параметры/Adam/RNG/history сохраняются, новые параметры и моменты вводятся явно; zero-init outputs сверяются на реальных полных последовательностях. Конфигурация нового optimizer group/head LR требует явной objective fork.
4. Собрать свежие собственные train-корпуса, full-actor обновление с отдельным учётом weapon gradient/состояния/retention. Старые temperature2 траектории не выдаются за on-policy данные нового residual.
5. Выполнить парную confirmation с обычными и recovery-сценами, rules и baseline; затем полные карты/реальный кооператив. Ни граф-проба, ни успешное восстановление PT не заменяют эти проверки.
