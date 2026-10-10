# Отбор stochastic кандидата перед independent test

`scripts/select_combat_stochastic_test_candidate.py` выполняет только
development selection. Он не открывает seed inventory, не резервирует и не
запускает test, не обучает и не продвигает веса. Для actual independent test
потребуется adapter к зарезервированным conditions и predeclared controls;
старый `run_combat_independent_test.py` ожидает deterministic `*-after` и
не должен напрямую получать новую stochastic selection.

Groups фиксируются до закрытия development в selection-protocol.json. Каждый
group — две stochastic ветки одного source_weights SHA, разные declared
policy RNG offsets, одинаковые generated fixtures/engine seeds. Required
gates: complete core,9 diagnostics,ownership/storage acceptance SHA,
zero clear-target rules и zero equip fallback. Actual model/plan SHA проверены.

Выбор по сумме побед двух arms, затем смертям и среднему полученному урону.
Допуск к будущему test: среднее число побед выше rules; при равных победах
среднее число смертей ниже rules. Это development gate, не статистическое
доказательство превосходства. Повторение одной геометрии с другим policy RNG
не увеличивает число независимых generated условий вдвое.

Реальная проверка на закрытом corrected-runtime1040:
`pickup-stochastic-test-gate-v1-20261010`, selected=parent3-after,
mean_wins66.5 versus rules68, eligible=false. Таким образом, удачная
отдельная ветка70/80 не запускает independent test. Native test не тронут.

Для текущего common curriculum560 за фактическим evaluation PID21684
поставлен отбор parent/uniform/wall, по stochastic-a/b каждый. Каталог:
`wall-curriculum-test-selection-v1-20261010`. OS handle удерживается, после
его закрытия проверяются seals, а не просто stage файл. Итоговый decision
будет опубликован после полной оценки. Positive gate потребует отдельного
test preparation/dispatch и актуальной проверки зарезервированного inventory;
он не будет означать автоматическую production promotion.

## Подготовка независимого stochastic test

`stochastic-heldout-preparation-v1-20261010/protocol.json` имеет состояние
`static_prepared_not_dispatched`: три контрольных плана по160 условий
подготовлены и проверены настоящим registry/BSP planner. Все20 рецептов и
тестовые сиды совпадают с исходным зарезервированным inventory. Native test
не запускался; `dispatch-reservation.json` отсутствует.

Добавлен `scripts/run_combat_stochastic_heldout.py`: две stochastic ветки
единственного выбранного actor плюс исходные parent3-before, FireBC и rules,
всего800 боёв,16 слотов,×2. Перед запуском повторяет sealed development
selection, обновляет workspace metadata inventory и требует точного совпадения
зарезервированных условий. Если условия уже использованы, отказывает вместо
смены test seeds. Все веса и пять планов фиксируются до native dispatch;
reservation создаётся эксклюзивно через file mode `x` в исходном preparation.
После dispatch проверяет member proofs, actual sampling configs, качество,
девять diagnostics, ownership, outcome behavior и физическое хранение.
Автоматического продвижения модели нет.

Проверка реализации: Python syntax проходит; реально отклонённый corrected1040
кандидат (mean66.5 versus rules68) отвергается до создания test output или
reservation. Повторная metadata-проверка трёх подготовленных control plans
подтвердила все160 условий каждого. Положительный полный native запуск нового
драйвера ещё не проверен: текущий560 development не закончен и eligibility
неизвестна. Это готовый к последующей проверке драйвер, не доказательство
качества или завершённого independent test. Перед запуском требуется8GiB
свободного места; закрытые native streams сжимаются существующим сборщиком.

Перед dispatch также дважды сверяются source bindings development cohort.
На закрытом corrected1040 реальная проверка текущего дерева прошла для704
Go/harness записей и268 native source записей. Это проверка SHA источников,
без численного NN replay. Обе stochastic ветки нового test имеют общий
`model=selected`, разные labels и offsets: существующий actual-config auditor
будет группировать их в пары и обнаруживать повторившиеся execution configs.
