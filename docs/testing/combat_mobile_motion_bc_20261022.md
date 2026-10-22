# Coordinated ground motion/aim BC и разделение компонентов

Статус:200-step CUDA fit,independent audit,smoke и40-case component matrix закрыты. Full/movement candidates не продвигаются;aim component имеет лишь слабый положительный сигнал. Дополнительный CUDA world-input audit выявил расхождение teacher-forced и own-mode качества;нужна проверка калибровки mode до дорогого расширения.

## Данные и протокол

Источник: `combat-rules-target-fire-bc-v1-20261022`,development candidate с прежним weapon head. Учебный корпус/качество меток описаны в [mobile teacher протоколе](combat_mobile_teacher_20261022.md). `combat-rules-mobile-motion-labels-v1-20261022`:1881 train/1255 development labels,полные4277/2365 context rows,32/16 native demonstrations,новые seeds.

Новый `train_registered_teacher_motion_cuda.py` создаёт `combat-rules-mobile-motion-bc-v1-20261022`. Фиксированы200 steps,lr0.0003,seed20261022,final checkpoint без выбора по development. Loss:balanced moving/explicit-hold physical MSE + coordinated yaw/pitch latent Huber(beta0.1) +0.2 balanced coarse/fine CE +0.02 policy KL retention. Эти коэффициенты не подбирались по текущему smoke.

Обучаются только motion rows0:2,mode rows63:81 в base head/residual и существующий sampled-motion aim MLP. Encoder,base aim weights,shared spatial/weapon-state weights,target/fire/weapon/vertical rows,std,value заморожены. **Замороженный spatial branch получает movement mean на вход**,поэтому его эффективный aim может измениться даже без изменения его weights. Не считать этот опыт независимым вмешательством только в движение.

Учебный aim использует movement из того же исполненного teacher action;для conditioner clip±(1−1e−6) нужен лишь для конечного atanh,physical movement target остаётся исходным±1. Никаких future features:соседний displacement используется только при отборе labels. Context пересчитывается causal attention window32,без переноса между seed/life/gaps. Retention использует≤128 train teacher histories;movement один раз сэмплируется из замороженной исходной Gaussian NN на CUDA и сохраняется. Teacher commands не выдаются за own-policy Gaussian draws. Target/mode expectation в KL аналитическая,условная aim часть использует этот фиксированный draw.

Вся NN арифметика и checkpoint loading на CUDA,CPU fallback отсутствует. Optimizer loop занял10.75с;это небольшой supervised fit32 боёв,не доказательство достаточного RL бюджета. Development не участвует в loss или выборе endpoint.

## Offline результат

| Development | До | После |
|---|---:|---:|
| Balanced movement RMSE |0.464|0.371|
| Moving RMSE |0.523|0.417|
| Hold RMSE |0.396|0.319|
| Aim RMSE,°/ось |4.169|3.731|
| Fine aim RMSE,°/ось |3.488|2.566|
| Coarse aim RMSE,°/ось |17.228|20.167|
| Balanced mode agreement |80.27%|96.45%|
| Reference policy KL на фиксированных draws |0|3.907|

Fine и movement имитация улучшились,coarse ухудшился;KL заметный. Это не достаточное основание объявить более сильный бой.

`audit_registered_teacher_motion_cuda.py` независимо проверил JSON/checkpoint/finite values,точную ownership разрешённых parameter rows,value/value Adam/std/config/PPO consumption history. Actor Adam после BC явно пустой;дальше PPO только на свежих траекториях ученика. На2890 real source native rows/2906context rows encoder memory совпадает в пределах9.06e−6;raw2:29 (none aim,attack,vertical,weapon,target) сохранены точно. Changed tensors:head/residual weight+bias и6 tensors conditional MLP. При разных будущих observations решение других heads может меняться;аудит не обещает одинаковых live trajectories.

## Закрытый smoke

`combat-mobile-motion-bc-smoke-evaluation-v1-20261022`,offset616,порт45200:8 fresh условий/вариант,4ordinary/4recovery,24native/16CUDA,16 ×2. Rules уже допускают short combat retreat в isolated fixtures. Пять отчётов/hashes и CUDA replay закрыты,decision `candidate_not_advanced_component_ablation_only`,screen=false,promotion=false.

| Метрика | Source target/fire BC | Coordinated motion BC | Mobile rules |
|---|---:|---:|---:|
| Победа |5/8|4/8|6/8|
| Убито монстров |13|13|14|
| Нанесённый урон |2273|1972|2208|
| Полученный урон |383|620|300|
| Большой yaw ≥30° |3.44%|6.01%|—|
| Стояние при clear enemy |17.98%|19.32%|—|
| Смены цели/100frames |8.06|9.70|—|
| Attack без выбранной цели |2.24%|22.12%|—|
| Реальный MG damage contact |132/196(67.35%)|117/200(58.50%)|—|

Неплохая supervised ошибка не перенеслась в бой. Target/fire weights не изменялись,но новая траектория часто приходит в другие состояния,включая отсутствие наблюдаемой цели. Нельзя приписывать все изменения только одной голове или скрывать ухудшение полученного урона.

## Matched component matrix

`fork_registered_teacher_motion_cuda.py` без нового fitting переносит в source либо motion parameter rows0:2,либо mode63:81+conditional aim weights. CUDA exact checkpoint/JSON и тот же independent source-history audit прошли у обеих компонент. Все остальные weights source сохранены. Это parameter interventions;movement inputs всё равно влияют на frozen spatial outputs.

Один динамический пул16 ×2 сравнивает source/movement weights/aim weights/full candidate/rules на одних8 fresh условиях:40native/32CUDA,offset620,4ordinary/4recovery,8 paired comparisons. Никакого автоматического promotion. Report bundle теперь поддерживает отдельный `combat_component_ablation_stage_v1` с проверкой полного budget/четырёх CUDA replay/component identities;старый3-way stage сохранён. Три focused tests проверили оба режима и отказ при пропавшем replay/неверном budget.

Первая подготовка `combat-mobile-component-ablation-evaluation-v1-20261022` остановилась до pool creation:UDP45301 занят PnkBstrB. Сервис не останавливался,боёв не было. V2 spec изменяет только порт45400 и reason;frozen weights/scenes/seeds и historical scan сохранены. Реальный запуск — `combat-mobile-component-ablation-evaluation-v2-20261022`;результатов пока нет. Failed preparation сохранена для аудита.

После матрицы выбирать следующий опыт по реальным исходам/урону/лишним действиям,не по меньшему RMSE. Полный candidate не получает screen автоматически. Не обучать на текущих evaluation logs;новые DAgger/PPO состояния должны собираться в train split. Полностью обучаемое движение/бой,перенос на кампанию/coop и превосходство над полноценными правилами остаются целью.

## Закрытая component matrix

Все40 native/32CUDA и пять отчётов завершены;hashes verified. Компоненты получили один и тот же8-case набор,парные выводы внутри этой матрицы допустимы.

| Метрика | Source | Movement weights | Aim weights | Full | Mobile rules |
|---|---:|---:|---:|---:|---:|
| Победа |4/8|4/8|5/8|3/8|4/8|
| Убито монстров |11|9|12|11|10|
| Нанесённый урон |1813|1706|1991|1758|1866|
| Полученный урон |421|509|446|508|316|
| Смерти |3|4|2|3|3|
| Clear-enemy standing |23.21%|25.49%|17.45%|19.35%|—|
| Большой yaw ≥30° |4.56%|6.34%|9.55%|7.51%|—|
| Attack без selected target |11.82%|6.94%|21.57%|14.23%|—|
| Native MG damage contact |78/160(48.75%)|67/149(44.97%)|103/189(54.50%)|73/160(45.63%)|—|

Aim component/source:2 gained/1lost;aim/rules:2/1. Это небольшой development сигнал,не превосходство. По yaw и targetless attack стало хуже. Full повторно ухудшился;movement не даёт выигрыша побед/урона. Default сохранён. Первая decision допускает только дальнейшую проверку aim component;large confirmation/продвижение любого checkpoint не назначены.

## Сравнение teacher forcing с собственным решением

`audit_teacher_motion_world_cuda.py` без обучения пересчитывает те же teacher histories наCUDA для четырёх вариантов. Target фиксирован явной доступной teacher целью. Сравниваются teacher-forced movement/mode и собственное mean movement/argmax mode. Planar world INPUT basis учитывает yaw/pitch,но не заменяет PMove/collision/velocity model;это не реальная траектория и не полный stochastic rollout.

| Development1255 | Source | Full | Movement weights | Aim weights |
|---|---:|---:|---:|---:|
| World-input balanced RMSE |0.466|0.373|0.373|0.466|
| Teacher-forced aim RMSE,° |4.169|3.731|4.214|4.234|
| Own-mean/own-mode aim RMSE,° |3.987|4.299|3.987|4.324|
| Argmax fine mode |98.57%|95.54%|98.65%|95.14%|

World-input ошибка уменьшилась вместе с body-axis ошибкой;нельзя объявить отсутствие world-input loss доказанной причиной игровых провалов. Напротив,при собственном mode положительный forced-aim результат исчезает. В labels coarse66/1881 train и23/1255development (3.51%/1.83%),а aim component выбирает coarse argmax10.47%/4.86%. **Гипотеза:**balanced mode CE переоценивает редкий крупный поворот;не доказано,что одна калибровка устранит игровой провал.

Следующая дешёвая проверка перед большим screen:калибровать только coarse/fine logit intercept наTRAIN с естественными частотами,оставляя остальные tensors неизменными. Не подбирать intercept поdevelopment/native evaluation. Проверить proper NLL/own-mode error наGPU,затем fresh matched бои source/aim/calibrated aim/rules при полезном offline результате. Альтернатива при отсутствии сигнала — собственные train trajectories/DAgger/PPO от лучшего source,а не дальнейшее усиление имитации неподходящих motion labels. No calibration fit выполнен на момент этой записи.

Калибровка теперь закрыта наCUDA:изменены только bias65/66,train-only intercept−2.8421. Наdevelopment coarse probability8.98→2.00%,natural CE.1063→.0293,own-mode aim4.324→3.464°. Independent audit1026 source histories прошёл. Запущен fresh32native/24CUDA probe четырёх вариантов,offset624,16 ×2;подробности в [calibration протоколе](combat_aim_mode_calibration_20261022.md).
