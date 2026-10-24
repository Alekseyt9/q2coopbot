# Опора, гранаты gunner и компактный тип снаряда

24.10.2026. Продолжение TRAIN-only разбора после отрицательного body/control confirmation. Нового обучения, игровых прогонов и замены основной модели на этом этапе нет.

## Опора: почему OnGround не совпадал с точечным тестом

Добавлен диагностический `cmd/q2support-audit` и `MapInfo.InspectGroundSupport`. Они сравнивают точечные пробы в центре/четырёх углах, смещение границы0.125 и полные hull sweeps вниз0.25/1/18.5/64 units: статический BSP, BSP с movers, затем с наблюдаемыми collision bodies. Не выдают разрешение на движение, не меняют guard и не вызывают нейронную сеть.

На двух заранее выбранных TRAIN состояниях:

| Состояние | Центр | Часть углов | Полный hull вниз0.25 |
|---|---|---|---|
| base3/1164165468/index113 | Пол примерно64units ниже | Есть контакт через0.125 | Есть ровная статическая опора |
| ware2/1172065468/index94 | В пределах64units не найден | Есть контакт через0.125 | Есть ровная статическая опора |

У обоих hull trace valid=true, startsolid/allsolid=false, fraction0.375, normal[0,0,1]. Movers и тела не изменяют результат короткой пробы. Клиентский OnGround согласуется с частичным перекрытием платформы полным объёмом игрока; это не означает ровную площадку под всем footprint.

`GroundFrictionStatus` намеренно требует ровную опору под центром и всеми углами, поэтому подробный predictor отказывается от этих состояний. Однако желаемое движение тоже нельзя автоматически разрешить: с учётом **delta_angles** и stock pitch/3 оно направлено к убываниюY, тогда как текущие близкие поддержанные пробы находятся на положительной сторонеY. В частности, простое чтение cmd.Yaw без delta_angles перевернуло бы направление base3.

Guard не ослаблен. Ошибочная остановка или безопасный будущий путь этими данными не доказаны. Артефакт `combat-body-control-support-details-v1-20261024`, report SHA `f958b409e70560599729571c752b45c062ba622d488596d8e7fd460f439e4f1b`.

Проверки частичной площадки, поддержки только телом, неизвестной геометрии, отсутствия мутации и корректного JSON прошли. Весь `go test ./internal/quake ./cmd/q2support-audit -count=1` прошёл с текущим BSP/AAS fixture. Первые попытки общего теста выявили инфраструктурные различия: в assets отсутствуетbase2.aas, исторический reference AAS не проходит текущий platform50 route fixture. Финальный тест использует сохранённый canonical AAS `42bd5036691c3eb4df038b3cb06fcf13edf953da095bc3a68fae4d6084b11f9a.aas` и symlinks к общим PAK. Наборы PAK/AAS не копировались, пользовательские runtime/config не менялись. Go тесты выполняли физику/метаданные, не NN forward.

## Какими атаками gunner убивает

На тех же96 собственных TRAIN боях и10952принятых переходах:

| Gunner атака | События с потерей HP | Потеря HP | Смертельные события |
|---|---:|---:|---:|
| Пули, MOD_UNKNOWN и attacker=inflictor |251|748|5|
| Grenade, MOD_GRENADE |26|762|10|
| Grenade splash, MOD_G_SPLASH |13|241|1|

Итого гранаты1003HP и11/16смертельных gunner событий. Нативные MOD/inflictor — только offline диагностические метки. Их значения и механика сверены с `yquake2/src/game/header/local.h`, `g_weapon.c` и gunner.c; они не стали входом модели.

Артефакт `combat-body-control-train-incoming-modes-v1-20261024`, SHA `1ffb623362b13e0663659236120ff7fb2f8dc38ffb09323c63488d29fcad0410`.

## Была ли граната видна модели

Из39TRAIN grenade HP событий: любая граната была в текущем наблюдении37раз, same-ID grenade33раз, все эти33попали в первые4 projectile слота, скорость same-ID была известна30раз. Среди11смертельных: любая граната11раз, same-ID/top4 —10, её скорость —8. Это совпадение entity ID и класса, не доказательство отсутствия переиспользования ID/нового intra-window spawn; прежние5кадров учитывают только принятые наблюдения.

Скорость не потеряна между environment и policy: `History.Enrich` после `EnrichEnvironment` вычисляет её по соседним клиентским кадрам в игровом времени, сбрасывая при разрывах/смене жизни. Нельзя объявлять velocity всегда отсутствующей только по конструктору в environment.go.

Но обнаружен конкретный пробел **кодирования типа**. Legacy environment encoder использует для projectiles общий Enemy encoder: в type bits есть parasite/gunner, но нет grenade/rocket/blaster. При одинаковой геометрии/скорости grenade и rocket дают одинаковый старый вектор; это проверено scalar feature тестом. Текстовый `class` присутствует в JSON наблюдения, однако сам по себе не является входом нейронной сети.

Артефакт availability: `combat-body-control-grenade-availability-v1-20261024`, SHA `f72250f9ea390143b2037768768bc8724fa58e85ad902db63e0915ed66ac6850`. Это основание для новой проверяемой гипотезы, а не доказательство, что типы уже улучшат выживание.

## Новый encoder: реализован, PPO пока не подключён

Экспериментальные `combat_features_v12_projectile_types`, `_neutral`, `_ablated`. Точный parent prefix —1469признаков `combat_features_v11_body_ablated`; новая ширина1489. Дополнительные20бит — четыре projectile слота по пяти типам: grenade,hand_grenade,rocket,blaster_bolt,other. Порядок совпадает с legacy stable distance sort, без entity IDs. Старые версии сохраняются.

Основной matched control — **neutral**, не all-zero. Он даёт ровно один other-bit на каждый занятый слот, скрывая класс и сохраняя presence/возможность обучать занятые слоты. Если в данных есть только один вид снаряда, такие входы отличаются только перестановкой активной колонки; нельзя выдавать дополнительный trainable presence gate за пользу типа. All-zero остаётся техническим дополнительным контролем, не главным сравнением.

Go scalar encoder и CUDA encoder проверены на128реальных наблюдениях изTRAIN captures и6synthetic проверках типов/пустого входа. Typed и neutral совпали точно, prefix не изменился, presence по слотам одинаков. NN policy не исполнялась, fit не выполнялся. V1 fixture с all-zero контролем сохранён вместе с исходниками; основной technical receipt — `combat-projectile-type-encoder-fixture-v2-20261024`, SHA `ed5814d22137ac328f9d7bfacc13e7d92fcedcf43948383c09152adb700881a7`.

На этапе encoder (около16:55) `LoadPPO` ещё отвергал новые версии: admission, auxiliary-prefix compatibility, новая ширина input layers и checkpoint migration не были добавлены. Не объявлять encoder готовой обученной системой. Более позднее подключение описано ниже.

## Следующие шаги до нового fitting

1. Добавить полную совместимость нового feature schema, сохранив1147-префикс существующих auxiliary heads и threat-memory capability. Typed/neutral должны отличаться только классом, не владением боем или guard.
2. На GPU одинаково расширить actor/value нулевыми20колонками. Исходный experimental parent: `combat-body-training-workflow-v2-20261024/round-04/ablated/update`, weights SHA `dca2e0cb8c7805324c4291cb3ad5dae4e3d01338bb94259dd0bb9a4424bf44fe`. Он не принят как превосходящий rules.
3. Отдельный versioned training scope/checkpoint: сохранить семь старых Adam групп/часы/состояния и общий обучаемый bias в обеих ветках; новая input группа одинакова в typed/neutral. Для active critic расширить моменты input weight нулевыми колонками с сохранением старого prefix и step counters; старые optimizer anchors не менять. Не сбрасывать скрыто историю critic одной ветки.
4. Проверить CUDA исходную эквивалентность, сохранение/возобновление actor/critic и native parity до реального изменения весов. Текущие encoder receipts не являются этими проверками.
5. Только затем зафиксировать равный собственный TRAIN бюджет и свежую оценку. Считать наличие и разнообразие projectile типов в обеих ветках: отсутствие разнообразия ограничит проверяемую гипотезу. Не подавать эти просмотренные confirmation состояния в fit, не менять награду одновременно и не подменять движение правилами grenade evade.

Полезность нового входа и устойчивое превосходство над rules не установлены. Обучение и численные проверки сети остаются только CUDA; этот этап добавил диагностику и scalar encoder.

## GPU migration и admission — около17:11 МСК

Добавлена совместимость1489входов в Go PPO/auxiliary admission и Python target/spatial helpers. Вспомогательные heads продолжают читать прежние1147признаков; body scope WIDTH1469 не менялся. `UsesThreatMemory` сохраняется у всех v12 режимов; отдельный metadata test предотвращает прежний класс ошибок раннего handoff из-за feature version. CUDA finalizer сохраняет новый helper вместе с frozen verifier sources.

Новый scope `combat_projectile_type_input_ppo_v1`: семь исходных actor Adam групп сохранены полностью, восьмая группа — новые20input columns. Общий input bias одинаково обучаем в обеих ветках. Active critic Adam получил нулевые моменты только для новых input columns; старые моменты, step counters и старые optimizer anchors сохранены. Config/history/consumed rollouts/RNG исходного checkpoint сохранены, новая история scope пуста, scope_updates=0. Не было сброса critic между ветками.

Обе миграции завершены CUDA:

| Ветка | Initial weights SHA | Raw/value/log-prob max error к parent |
|---|---|---|
| typed |4867d17ac6385a4cc78a3f97f86d782b2a1e07616b4295d5c4f6dd90f91d40e5|0 / 0 / 0|
| neutral |f99f49f28d2d6b0b566d259484fb58f15a4b181e5477f9d32e9a2a9aed961d77|0 / 0 / 0|

Парная CUDA проверка подтвердила одинаковые начальные actor/value/std tensors, активные/старые actor и critic optimizers, RNG и consumed history. Checkpoint roundtrip и следующий пробный actor/critic Adam step после загрузки совпали. Новые input-градиенты ненулевые у обеих веток (typed0.174929,neutral0.176577); общий bias-gradient одинаков0.697617. Пробные шаги не сохранены как обученные веса.

Использованы2737переходов последнего собственного TRAIN корпуса parent, уже consumed им ранее. Его behavior предшествует final parent update; сравнивались пересчитанные **текущим parent** log probabilities, не заявлялось совпадение нового parent с прежней behavior policy. Эти записи служили только equivalence/resume proof и не были повторным on-policy обновлением. Source unchanged, fitting=false.

Протокол `docs/combat_projectile_type_initializer_protocol_20261024.json`, SHA `0cdcd455c0bd98176cb35004b1b97913e5eb857d11a85e756cc6eb50baf6f2d8`. Артефакты `combat-projectile-type-initial-v1-20261024/{typed,neutral}` и `pair-proof.json`. Go metadata tests фактически загрузили обе модели и отвергли их новую ширину под старой схемой; neural forward на Go не выполнялся.

Запущен `combat-projectile-type-parity-workflow-v1-20261024`: source-v11/typed-initial/neutral-initial, четыре свежих test recipes,12native/12CUDA планово,16slots,x2,port49500. Spec `docs/combat_projectile_type_parity_spec_20261024.json`, SHA `dc417931a0f26aad6e44aeabb3c64a62c34771c430f942f3691b6eda3694bb2f`. Supervisor PID6848, evaluator child23516. Веса и исходники зафиксированы до завершения. Это technical initial parity, не новая оценка преимущества. До её закрытия полноценное fitting не запускается; актуального нового TRAIN бюджета и trainer workflow ещё нет.

### Восстановление после занятого UDP диапазона

Первая попытка остановилась до native запуска: port49515 был занят посторонним процессом. Ни pool output, ни capture roots ещё не существовали, native_cases_run=0; исходные failed progress/log сохранены. Посторонний процесс не останавливался.

Добавлен `scripts/recover_prepared_combat_evaluation.py`: принимает только доказанный pre-native port collision, проверяет исходный source fence, SHA fixture/spec/plans/models и отсутствие любых capture outputs. Использует **ровно те же уже замороженные планы**, меняя только UDP диапазон. Он не генерирует новые сцены/сиды и не обходит историческую проверку для нового опыта; это продолжение уже проверенной подготовки. Повторно применять к частично сыгранному пулу нельзя.

Recovery запущен около17:19 в `combat-projectile-type-parity-recovery-v1-20261024`, supervisor10360, recovery runner19968, фактические порты41000–41015. Родительский spec с первоначальным port49500 не переписан; operational `recovery.json` явно связывает объявленный/фактический порт и исходный failure log SHA. Первоначальный overall workflow остаётся failed, активен новый recovery workflow. Evaluation captures/audits продолжаются в исходном evaluation root; окончательный parity receipt будет в recovery root. Около17:22 все12native jobs завершены, идёт последняя CUDA обработка. Результата parity gate на момент записи ещё нет.

### Native parity закрыт — около17:24 МСК

Recovery workflow завершён:12native/12CUDA,1293принятые строки суммарно. Все8пар source→typed/neutral на четырёх мирах совпали по фактически отправленным командам, предцельному состоянию мира, ownership и наборам кадров. Старый1469-префикс совпал, новый20-канальный typed/neutral хвост из Go записи совпал с CUDA encoder (max error0). До первого goal/death shutdown tail исключён одинаково. Source unchanged, fitting=false, promotion=false.

Parity report SHA `712f7e22b75aedc1cf1cf24504cd82f8cd7f69390980360d012d820a9622edd4`; recovery complete SHA связывает parity seal `8a277d67d1b819e43716f196aae33e0faa1430b4f940063d30e111629a108359`. Сами модели по-прежнему имеют нулевые новые input weights: это проверка неизменности управления при переносе, не доказательство лучшего поведения или выживания.

Теперь можно реализовать новый own-policy trainer. Важно: initializer имеет отдельный protocol SHA `0cdcd455...`, а будущий training protocol должен быть отдельным зафиксированным документом. Не переписывать initializer receipts и не требовать, чтобы их initialization protocol SHA вдруг совпал с новым training protocol. Сохранить initialization binding, добавить явный training binding в новый checkpoint; на update0 допустить только отсутствие training binding, дальше требовать точного совпадения. Старые body scope/history/optimizers остаются frozen anchors, активные поля — projectile scope/optimizer/value_optimizer. Новые корпуса должны быть1489ширины, body-префикс1147:1469нулевой, neutral скрывает только тип, все собственные TRAIN сиды новые.

Полноценного trainer/нового fitting пока нет. Перед fitting зафиксировать бюджет, награду и свежий paired screen; не выбирать промежуточные checkpoint по оценке и не объявлять технический parity боевым улучшением.
