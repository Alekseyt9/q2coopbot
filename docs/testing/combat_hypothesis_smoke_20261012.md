# Первый короткий цикл гипотез, 2026-10-12

Живой smoke:24/24 принятых native боя,16/16 CUDA replay cases, 1928 переходов. Время evaluator: 177.35 секунд, без внешнего build/generation/prepare-only и завершающего Markdown reporter. Пул16 x2; отдельный seed каждой сцены, одинаковые сцены/seeds для сравниваемых вариантов. Четыре карты base3/ware2/jail1/city1,4 ordinary+4 recovery условия на вариант. Обучения и продвижения весов не было.

| Вариант | Победы | Ordinary | Recovery | Полученный урон |
|---|---:|---:|---:|---:|
| control1 after stochastic |1/8|0/4|1/4|629|
| conditional2 after stochastic |2/8|0/4|2/4|668|
| rules |6/8|2/4|4/4|426|

Это проверка работоспособности короткого pipeline, не доказательство качества кандидата. Семейства, режимы и weights взяты из уже обученных закрытых обновлений; новая гипотеза движения ещё не исполнялась. Smoke использовал confirmation468 из исходного template; эти seeds считаются просмотренными screening-данными и не должны попасть в будущую независимую confirmation. После этого генератор исправлен: smoke/screen/coverage используют validation, final confirmation — confirmation. Источник первой версии генератора архивирован рядом с evaluation.

## Реализация

`prepare_combat_hypothesis_stage.py` выдаёт specs для8/24/56/112 условий на вариант, по три варианта; равномерно выбирает карты отдельно для ordinary/recovery. Короткий subset фиксируется до боя, а не выбирается по результатам. Planner и episode pool теперь принимают1..80 seeds на семейство; старый cohort scheduler при немножественном4 бюджете сообщает, что нужен episode scheduler. Полные старые планы сохраняют прежний бюджет. Driver проверяет фактические native/ordinary/recovery budgets до запуска pool.

Прошли3 metadata теста генератора, Go small-cohort test с verify-plan для1/2/3/4 seeds, build planner и pool dry-run с24 заданиями. Никакой NN в этих тестах не исполнялась. Все численные проверки моделей в smoke выполнялись на CUDA.

Для screen подготовлен spec:12 семейств x2 seeds x3 варианта =72 боя,16ordinary/8recovery per variant, validation472–473. Prepare-only завершён: модели/SHA/условия/исторические seeds проверены, ни одного боя не запускалось. Не запускать его ради повторной оценки уже известного слабого кандидата; использовать новую гипотезу исполнителя или обучения.

Artifacts: `combat-hypothesis-smoke-evaluation-v1-20261012/complete.json`, `wall-time.json`, `combat-hypothesis-smoke-report-v1-20261012/results.md`, `combat-hypothesis-screen-prepared-v1-20261012/prepared.json`. Auto-stop по механическим метрикам, quality cache и human recorder ещё не реализованы. Ступенчатый [план](combat_fast_hypothesis_cycle_plan_20261012.md) сохраняет эти следующие требования.
