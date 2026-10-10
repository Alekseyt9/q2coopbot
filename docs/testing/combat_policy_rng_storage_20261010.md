# Policy RNG и хранение новых captures

В предыдущем sampling пилоте харнес заменял seed из weights.json engine
seed каждого боя. Поэтому stochastic-a/b фактически исполняли одинаковые
конфиги. Это обнаружено SHA проверкой48 effective provider configs,
а не нейросетевым replay. Восстановленный пилот остаётся исторической
проверкой deterministic против одного stochastic режима с повторами.

Теперь frozen plan допускает `policy_sampling_seed_offset`, default0.
Effective policy seed = engine seed + offset; engine seed, геометрия,
состав монстров и native RNG не меняются. Диапазон effective seed0..2^31-1
проверяется до запуска. Настройка проходит через independent pool и
обычный registered combat-baseline runner. Manifest записывает offset.
`audit_combat_sampling_configs.py` сверяет actual config SHA с native
receipt, его параметры с исходными весами и seed с frozen plan/manifest.
Независимость arms проверяется по различию фактических конфигов, не именам.

Новый пилот56: m1/parent3 ×deterministic/stochastic-a/stochastic-b ×8
site02 условий плюс правила8. Offsets0/20261011/20261012. Веса меняют
только deterministic флаг; policy seed вычисляется в харнесе и сохраняется
как фактический конфиг. GPU обучение в этом эксперименте не проводится.
Live Go inference остаётся частью управления настоящими боями.

После полного пилота и проверки distinct configs поставлена в очередь
широкая оценка560: те же семь вариантов ×20 development семейств ×4
engine seeds validation24..27. Она удерживает OS handle пилота и начнётся
только после complete diagnostics и audit без повторных stochastic
конфигов. Повторно проверяются текущие source hashes пилота; source/native
изменение запрещает запуск. Это знакомая development выборка, не test.
Независимый test остаётся зарезервированным.

## Хранение

Общий bundle ранее при достижении предела1024 hardlinks NTFS молча копировал
бинарник каждому новому участнику. Теперь при заполнении группы создаётся
одна дополнительная общая копия, затем ссылки на неё. Именованный mutex
сериализует создание групп параллельными работниками. SHA каждого
установленного бинарника проверяется. Между разными томами установка
отклоняется, чтобы не скрывать массовое создание отдельных копий.

Практическая проверка создала1050 ссылок с parallelism16. Всего1052 пути
включая canonical и дополнительную копию соответствуют двум inode;
все installed hashes совпадают. Результат:
`workspace/artifacts/binary-link-shards-smoke-v1-20261010/report.json`.

Во время новых сборов отдельный orchestration поток сжимает JSONL/log
только участников с опубликованным успешным terminal job receipt и
capture_complete/provenance_valid report. LZX прозрачен для существующих
readers. SHA256 каждого файла проверяется до и после; список сохраняется
в stream-compression-receipts.json. Активные участники не обрабатываются.
Перед широким сбором требуется не менее10ГиБ свободного места; для
отдельного56-case smoke минимум2ГиБ, с тем же поэтапным сжатием.

Проверки: `go test ./internal/trainingepisodes` passed, включая допустимый
offset, отрицательный offset и переполнение; только генератор/реестр,
без CPU/Go нейросетевого replay. Синтаксис всех PS scripts проверен.
Live pilot завершён56/56 без ошибок; все девять диагностик завершены.
Audit48 actual provider configs подтвердил ноль повторных stochastic
условий и effective seeds согласно frozen offset. Все56 client paths
ссылаются на один inode; все56 exporter paths тоже на один inode.
В новом сборе нет прежних отдельных копий бинарника каждому участнику.

| Вариант | Победы /8 | Смерти |
| --- | ---: | ---: |
| m1 deterministic | 0 | 8 |
| m1 stochastic-a | 5 | 3 |
| m1 stochastic-b | 5 | 2 |
| parent3 deterministic | 0 | 5 |
| parent3 stochastic-a | 5 | 2 |
| parent3 stochastic-b | 4 | 3 |
| Правила | 8 | 0 |

Quality SHA256: `3aa4fbda9a8716a690c8d4b6bee5d8f0078e2c6ec28646eb0350e7e385a59d68`.
Diagnostics SHA256: `6f9b19331a8b0f1547f727b1884db2ff6fd2f90a178498dd554b43cb6ad82c6b`.
Вывод ограничен site02 development: stochastic помогает, но ещё уступает
правилам; источник выигрыша отдельных компонентов actions не изолирован.
Широкий560 сбор автоматически начал работу после полного пилота и source
проверок. Результаты широкой оценки пока не подтверждены. На старте
оставалось около12.7ГиБ свободного места; closed-member LZX идёт по ходу.

Пилот: `workspace/artifacts/action-sampling-offset-smoke-v1-20261010`.
Очередь560: `workspace/artifacts/action-sampling-offset-wide-v1-20261010`.
Драйвер: `scripts/run_combat_sampling_evaluation.py`.
