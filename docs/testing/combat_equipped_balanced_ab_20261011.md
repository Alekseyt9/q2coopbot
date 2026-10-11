# Обучение и сравнение признаков оружия v9/v10

Исходные линии — parent update10 и shared update11 после полного actor update.
У каждой линии контроль v9 и его CUDA migration v10 (1147 признаков).
Дополнены текущее оружие и собственная история последнего `use`-запроса;
нулевые новые колонки до обучения сохраняют прежние outputs. Архитектуру,
историю Adam и бюджеты сравнивают внутри каждой линии. Cross-line результаты
не доказывают превосходство архитектуры, поскольку исходная история различается.

Train pool **320/320** закрыт: 20 зарегистрированных семейств, четыре независимых
сида на семейство, четыре варианта. Все сцены и сиды между вариантами совпали.
Train seed offset44, 16 инстансов ×2, wall899,25 с; источник не изменился,
ошибок нет. Четыре модели получают только свои 80 боёв. Модель v9 не получает
действия или историю v10, и наоборот.

Настройки обучения прежние: actor_lr3e-5, actor_steps10, critic_steps40,
весь actor обучаем, own-history sequence retention weight1 с равной массой
семейств/сидов. Численная обработка, обучение и checkpoint audit только CUDA.
Новый retention spec явно оформляет fork прежней цели, сохранённой в lineage.

Оба обновления parent закрыты и проверены: v9 —9219 кадров,10 принятых
actor steps,sequence KL0,0030361061; v10 —9211 кадров,10 шагов,
sequence KL0,0030287823. Encoder/движение/прицел и Adam clocks проверены.
Оба продолжили счётчик до update11. История разных native own-policy боёв
имеет небольшую разницу в числе кадров; seed/fixture pairing совпадает.

CUDA input-use parent-v10 подтвердил влияние новых входов по настоящим
histories: при обнулении current weapon band средняя policy KL1,21300934e-5,
own request band4,85134657e-5. До обучения оба значения были0.
Это абляция наблюдений при фиксированных масках, а не боевой результат.
Все четыре обновления закрыты, GPU checkpoint audits и seals проверены.

| Вариант | Кадры train | Принятые actor steps | Update | Sequence KL |
|---|---:|---:|---:|---:|
| parent-v9 | 9219 | 10 | 11 | 0,0030361061 |
| parent-v10 | 9211 | 10 | 11 | 0,0030287823 |
| shared-v9 | 8942 | 10 | 12 | 0,0041465269 |
| shared-v10 | 8936 | 10 | 12 | 0,0046442243 |

Всего36308 кадров. Shared-v10 также использует новые входы: CUDA ablation
current weapon KL1,60062495e-5, own request KL1,29373108e-4.
Влияние обоих новых input bands проверено по настоящим histories.
Это проверка использования входов, не доказательство полезности для побед.

Следующий конечный pipeline запущен после закрытия training coordinator:
confirmation400 (четыре обновлённых варианта + rules,80 боёв каждому), offset4,
16×2; затем CUDA replay320 learned cases без обучения, native paired metrics,
ownership, post-release behavior v3 и weapon response audit.
Rules план не содержит модель. Данные confirmation не входят в train.
Этот pipeline стартует только при наличии complete receipt всех четырёх обновлений.

Все пять confirmation планов прошли проверку:20 семейств по4 сида каждый,
эпизоды/instances/seeds совпадают между вариантами; rules не содержит модель.
Начало capture подтверждено живым процессом, первые22/400 боёв закрыты.
На момент записи обучение закрыто, confirmation качество ещё не оценено.
Новые веса не повышены до опубликованной/игровой модели.

Артефакты в `workspace/artifacts`:

- `combat-equipped-balanced-training-pool-v1-20261011` — accepted native train.
- `combat-equipped-balanced-training-analysis-v1-20261011` — processing/spec/update/audit.
- `train-equipped-balanced-20261011.py` — coordinator четырёх обновлений.
- `capture-equipped-evaluation-20261011.ps1` — новая независимая confirmation.
- `analyze-equipped-evaluation-20261011.py` — строгая проверка и сводка результатов.

Сохранённые checkpoints, weights и отчёты остаются отдельными файлами;
исходные веса сверены по seals. Исходники Go/PowerShell нельзя менять во время
native capture или экспорта: привязанные fingerprints должны оставаться прежними.
