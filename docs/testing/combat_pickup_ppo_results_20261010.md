# Исправленный ownership: результат CUDA continuation

Сравнение `workspace/artifacts/pickup-ppo-eval-v1-20261010`:1040/1040 native
боёв завершены без ошибок, source_unchanged=true, все member receipts приняты.
Core quality complete; все9 диагностик и итоговый ownership/storage acceptance
ещё выполняются. Promotion отсутствует; independent test не запускался.

Сравниваются исходные и дообученные m1/parent3 под одним исправленным runtime:
20 семейств ×4 validation seeds, deterministic и два stochastic RNG offset.
Каждая модель перед оценкой получила80 own-policy train боёв и CUDA
continuation update1→2 с сохранением Adam/config/objective. Оба checkpoint
reload audit прошли.960 actual provider configs проверены, repeated stochastic
execution configs=0. Rules имеет80 боёв, а не160 независимых повторов.

| Модель / режим | Победы до → после /80 | Смерти до → после | Полученный урон до → после, средний |
| --- | ---: | ---: | ---: |
| m1 deterministic | 61 →62 | 17 →17 | 25.7 →26.0 |
| m1 stochastic-a | 64 →66 | 15 →13 | 28.8 →26.5 |
| m1 stochastic-b | 63 →65 | 13 →13 | 26.8 →25.9 |
| parent3 deterministic | 61 →58 | 14 →17 | 21.7 →24.4 |
| parent3 stochastic-a | 68 →70 | 9 →7 | 18.8 →15.6 |
| parent3 stochastic-b | 61 →63 | 13 →11 | 19.8 →22.6 |
| rules | 68 | 8 | 15.5 |

У обеих моделей stochastic ветки прибавили по2 победы, суммарно+4 на двух
RNG arms. Эти arms используют одни engine conditions и не дают160 независимых
геометрий. Parent3-after в среднем66.5 побед/80 против rules68; один arm70
не является устойчивым превосходством. Deterministic parent3 ухудшился,
поэтому результат зависит от режима исполнения. Возрастание урона parent3-b
при большем числе побед — отдельный tradeoff, не однозначное улучшение.

Для следующего curriculum A/B наиболее вероятный родитель — parent3-after:
133 победы в сумме двух stochastic arms против m1-after131. Запущенный
selection driver подтвердит выбор только после всех текущих gates, включая
нулевой rules fallback при видимой цели и нулевой equip fallback. Это выбор
на reused development, не независимая оценка. Обе ветки нового A/B начнут
с одинакового actor/value/std/Adam и получат одинаковый бюджет80 боёв.

SHA core quality:
`40706fadf73baff783f677f487338c29399be7f05360913c94b60b273f51e847`.
SHA member proof:
`dc030c2922be7aab03ca2b00e63e4ccd44ecab1f8c14489c31820037f30c3e2f`.
Полные эпизоды/парные изменения — quality-episodes.json и quality-report.json;
окончательный acceptance будет зафиксирован после завершения диагностики.

## Предварительная завершённая Machinegun диагностика

`machinegun-hits.json` уже рассчитан; общий9-report acceptance ещё pending.
Показатель — доля native Machinegun shots с damage живому монстру в
экспортированных окнах первой жизни, не общий kill rate или ошибка прицела.

| Вариант | Выстрелов до → после | Доля с damage живому монстру до → после |
| --- | ---: | ---: |
| m1 stochastic-a | 748 →798 | 43.72% →41.73% |
| m1 stochastic-b | 821 →805 | 43.73% →43.48% |
| parent3 stochastic-a | 637 →652 | 62.01% →61.66% |
| parent3 stochastic-b | 567 →577 | 56.44% →59.97% |
| rules | 280 | 98.93% |

Состав выстрелов, расстояния и моменты огня отличаются, поэтому это не
причинный эффект обучения прицелу. У parent3-a moving shots доля59.69→59.89%,
stationary67.98→69.42%; у parent3-b moving59.45→59.66%, stationary46.09→61.32%.
Descriptive strata не выравнивают состав целей и геометрию. Больший win rate
не означает, что все режимы стрельбы улучшились. Прежде чем менять reward
или усложнять сеть, нужен общий curriculum A/B и анализ выбранных промахов.

## Дробовик: команды и подтверждённый native урон

Все9 основных diagnostic reports завершены; общий ownership/storage closure
ещё pending. Дополнительный strict audit всех alive Shotgun frames отклонил
all-provider acceptance: встречаются rules frames без видимой цели.
Он сохраняет `pickup-command-observations.json` до assert; этот файл не
подменяет accepted proof. Ни в одном rules-Shotgun кадре нет clear target.

| Вариант | Provider Shotgun frames | Provider attack sent frames | Rules Shotgun frames с clear target |
| --- | ---: | ---: | ---: |
| m1 stochastic-a before | 381 | 186 | 0 |
| m1 stochastic-b before | 253 | 138 | 0 |
| m1 stochastic-a after | 338 | 193 | 0 |
| m1 stochastic-b after | 250 | 140 | 0 |
| parent3 stochastic-a before | 269 | 257 | 0 |
| parent3 stochastic-b before | 242 | 186 | 0 |
| parent3 stochastic-a after | 244 | 215 | 0 |
| parent3 stochastic-b after | 514 | 218 | 0 |

Первый разобранный rules frame: m1-a-before, parasite-blaster-generated,
seed700026/frame183, no observed enemies, fallback=system2_noncombat.
Наблюдаемая defeated запись содержит parasite351; native goal_stop kill_frame183.
Это handoff после наблюдаемой смерти в конкретном случае. Другие примеры
включают случаи без goal_stop; их нельзя автоматически отнести к хвосту
победы. Основной full ownership reporter классифицирует причины отдельно.
Никакие runtime gates для обучения не ослаблены.

На том же native эпизоде подтверждён реальный Shotgun damage:14 damage callbacks
с MOD_SHOTGUN=2,55 health damage живому monster_parasite,1 добивание, внутри
exclusive matched provider-owned equipped Shotgun attack windows. Входные
SHA и полные события сохранены в `shotgun-native-example.json`, native/report
SHA связан с member proof. Callback может быть отдельной дробиной, не считать
это14 выстрелами. Ранее монстр получал урон другим оружием;55 — вклад только
проверенных Shotgun событий. Это положительное подтверждение в одном эпизоде
исходного m1-a, а не глобальная метрика точности или улучшение новых весов.

Full `control-ownership.json` затем завершился: во всех12 learned вариантах
rules_with_clear_target=0 и pilot_equip_not_ready=0. Оставшиеся rules frames
помечены system2_noncombat или combat_visibility_timeout, а не недопуском
дробовика. Это acceptance выбранного combat ownership scope; не утверждение,
что каждый alive кадр первой жизни управлялся сетью. Итоговое закрытие
driver ещё ожидает visible-target/storage passes.

## Окончательное закрытие

Driver завершился штатно. Все1040 member proofs,960 actual sampling configs,
9 diagnostic reports, ownership/visible-target и физический storage audit
закрыты; ссылки на SHA из progress.json сверены с фактическими файлами.
Диагностический strict all-alive Shotgun audit отдельно остался rejected
из-за noncombat frames; он не является обязательным scoped ownership gate.
Ни один runtime gate не был ослаблен для перехода к следующему обучению.

SHA diagnostics acceptance:
`f178d6f2ce6079c5c3fa0044edffedd50123db831f056588f69e11dc59e79c9e`.
SHA ownership acceptance:
`f66589bc833b34496c83255c6f248c866254e65e83620d736039b4e56cfc3b1f`.
SHA storage final:
`793da9b40e84d6945898f2eb49ca6ae8c782633013d70e7804834bc19b948df1`.

Сжатие4979 записей/логов/JSON:11641114107 logical bytes →3570741248 physical
bytes, с сохранением SHA. Это snapshot выбранных файлов, не размер всего
репозитория или всего диска. Новый A/B автоматически начал160 own-policy
боёв: parent3-after выбран по сумме133 побед/18 смертей двух RNG arms.
Копии исходных весов и checkpoint SHA обеих веток проверены идентичными.
Новых результатов качества или независимого test пока нет.
