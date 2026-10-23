# Выбор поправки по последствиям до конца боя

23.10.2026. Протокол фиксируется после native исходов независимой проверки shared:10/24 против source9/24, constant11/24 и rules17/24. CUDA-проверка этой серии ещё выполняется. Кандидат не прошёл критерий по победам и потерям; большую проверку этих весов не запускать. Малый положительный screen6/8 сохраняется отдельно.

Проверяем гипотезу: метка «попал этим выстрелом» недостаточна для выбора полезного боевого действия. В уже закрытой диагностике есть поправки с большим immediate HP и худшей последующей наградой. Это не устанавливает причину провала shared; одиночный rollout также может быть шумным.

## Данные и границы

Использовать только26новых late one-frame контекстов и129принятых вариантов из combat-native-aim-late-dataset-v1-20261023, с проверенными reward prefixes из combat-one-frame-return-diagnostic-v1-20261023. Старые17constant-head контекстов не использовать для return labels: у них вмешательство могло повторяться после выбранного выстрела. Сохранить14train/7seed и12development/6seed. Никакие confirmation/validation данные не участвуют в fitting или нормализации. Development уже исследовательский; не называть его независимой боевой оценкой.

Полный return — сумма доступных reward score от изменённого шага до конца first-life capture. Sparse return — та же сумма только monster_damage/received_damage/self_damage_extra/friendly_damage/death/monster_kill/tick с существующими коэффициентамиv13. Не менять reward игры. Недоступный post-terminal хвост не восстанавливать и не выдавать capture за бесконечный return. До fitting проверить уникальность(seed,frame), соответствие категорий accepted labels, неизменность исходных файлов, совпадение baseline и reward config. Все исключения показывать.

## Два равных опыта на GPU

Одинаковая shared191→16→1 архитектура, тот же frozen S4 body/motor/hit predictor и candidate feature layout. Начать заново, не продолжать веса отрицательного24-case кандидата. Общий seed20261023, фиксированные500Adam шагов, networkLR0.003/weight_decay0.0001, betaLR0.05/weight_decay0, gradientclip0.5, score/beta bounds±20. Прежняя source TRAIN mean P0 задаёт исходный beta; случайная первая матрица одинакова между опытами. Не выбирать промежуточный checkpoint поdevelopment.

1. Контроль immediate: прежнее предпочтение максимального известного health_damage; при отсутствии улучшения baseline получает предпочтение. Набор теперь ровно26late contexts, поэтому прежний43-state fit не является matched контролем.
2. Sparse-return teacher: предпочесть известные действия с максимальным sparse return, если максимум выше baseline не менее чем на1reward unit; иначе предпочесть baseline. Порог1равен одной пятой существующей награды за убийство и фиксируется до нового fit, а не подбирается по development. Все категории в пределах1e−6 от максимума делят предпочтение поровну.

Оба target распределения =0.1исходных S4 probabilities +0.9предпочтения. Непроверенные действия не получают вымышленных return/HP меток. Source tensors/files, Adam/RNG и predictor остаются неизменными. Все neural forward/fitting/численные сравнения толькоCUDA. Новый результат — prototype, не готовый training resume.

## Метрики и допуск

Для обоих опытов показывать immediate полезную/вредную массу, sparse-return полезную/вредную массу (delta≥1 или≤−1), вклад только известных категорий в sparse/full delta, known mass, baseline probability и unmeasured argmax. Последний вклад не равен ожидаемому return всейpolicy: исходы неизвестных категорий отсутствуют. Показать per-context, train/development и распределение поseed.

Контроль каждого endpoint — постоянное25распределение из его среднего наTRAIN. Допуск sparse-return варианта к новому native screen: development полезная sparse масса≥source+0.05,≥своегоconstant+0.02 и≥matched immediate+0.02; вредная sparse масса≤source+0.05; вклад известных sparse delta≥constant+0.10 и≥matched immediate+0.10 reward unit. Отрицательные результаты всех критериев сохранить. Это фильтр гипотезы, не гарантия ожидаемой полезности или победности.

Еслидопускнепройден, не запускатьновыеигрыэтихвесовинеослаблятьпорог. Следующая отдельнаязадача — расширение независимыхTRAINсостояний и при необходимости one-frame вмешательства на более реалистичной behaviorpolicy; повторные одиночные rollouts не объявлять полноценнымMCTS. Еслипройден — отдельныйversioned export и frozen fresh paired source/immediate/return/constant/rules screen слимитом40native/32CUDA. Его seeds/порог зафиксировать до игр. Default автоматически не меняется.
