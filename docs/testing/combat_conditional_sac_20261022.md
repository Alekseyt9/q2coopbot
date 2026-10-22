# Conditional SAC: первый опыт с повторным использованием боёв

## Цель и метод

После отрицательного опыта с native miss cost проверяется следующий пункт плана:другой способ использования опыта. SAC (Soft Actor-Critic) обучает стохастическую политику и две Q-функции по сохранённым переходам. Здесь реализован только непрерывный контур прицела;полная замена смешанной политики SAC ещё не выполнена. Основание алгоритма — [работа авторов](https://proceedings.mlr.press/v80/haarnoja18b.html),формулы twin Q/soft target/tanh Gaussian — [официальное описание](https://spinningup.openai.com/en/latest/algorithms/sac.html).

Общий старт: `combat-sequence-hitscan-pair-v2-20261022/control-source`,actor/std прежнего target/fire BC,одинаково обнулённый critic и пустые PPO Adam. Reward остаётся `combat_reward_v13` с нулевым machinegun_miss;commitment bonus и остальные coefficients сохранены. Одновременно новый reward не проверяется.

Общий корпус:закрытые32 TRAIN боя контрольной ветки,3450 accepted transitions. Старые validation/test данные в обучение не попадают. Это повторное использование одного корпуса двумя алгоритмами,не64 новых симуляций. Выигрыши на этих train seeds не используются для выбора checkpoint.

## Что обучается

Обе ветки меняют только существующую `sampled_motion_aim_branch` и два aim log_std:10212 параметров. Условие включает выбранную цель,aim mode и фактически sampled movement. Encoder/attention,все raw81 outputs,target/weapon/fire/vertical/movement distributions и movement std заморожены. Модель продолжает сама выбирать остальные команды своими прежними learned heads;правила не подставляют прицел.

SAC critics получают наблюдаемые features,causal summary того же замороженного encoder,а также выбранные frozen-policy категории и движение. Это уже выбранное условие перед двухмерным действием прицела. Следующие категории/движение для Bellman target выбираются из той же frozen policy. SAC coordinates — нормализованные `tanh(yaw/pitch latent)`;fine/coarse scale хранится в условии critic. Entropy учитывает корректный tanh Jacobian;движок и существующий Go sampler не меняются.

Этот ограниченный опыт не доказывает,что фиксировать остальные головы правильно для итогового агента. Он нужен для сравнения алгоритма на одной непрерывной подзадаче;конечная цель полного learned боя сохраняется.

## Replay и границы

`prepare_combat_replay_cuda.py` проверяет source/checkpoint/objective/native pool и закрытые CUDA exports. Все3450 переходов имеют доказуемую связь с контекстом;3419 bootstrap transitions используют точное следующее наблюдение того же seed/frame.31 terminal/truncated transition используют явно сохранённый bootstrap_zero. Пропуск successor никогда не превращается в выдуманный terminal,через reset/bootstrap не переходим. Это прежняя задача конечного provider segment,а не новая трактовка завершения всей карты.

Features/current context используют только текущие/прошлые наблюдения. Следующее наблюдение доступно только для TD target. Native скрытое состояние не добавляется в actor/critic inputs. Encoder history посчитана на CUDA с настоящим полным causal prefix;разница к записанной памяти9.06e−6,log probability max error1.55e−4. Оба значения внутри существующих runtime tolerances.

Binary replay `combat-conditional-sac-replay-v1-20261022/replay.pt` хранит общие context tensors и индексы переходов,а не многократные JSON копии каждого prefix. Read-back проверен на CUDA. CPU unit tests проверяют только границы/JSON логику;все NN/численные algorithm checks — CUDA.

## Фиксированный первый бюджет

| Ветка | Учебные бои | Actor updates | Critic updates | Обучаемый прицел |
|---|---:|---:|---:|---:|
| PPO conditional aim |те же32|10|40 value|10212 параметров|
| SAC conditional aim |те же32|256|512 twin Q,из них256 warmup|10212 параметров|

PPO сохраняет свой clip/target KL;SAC использует twin Q,Polyak retention0.995,batch256,alpha0.001,actor lr3e−5,Q lr3e−4. Оба без дополнительного sequence retention loss;проверочный retention spec у PPO используется только для диагностики. Разное число optimizer steps намеренно проверяет повторное использование тех же данных. Записывается также стоимость вычислений:одинаковый budget среды не означает одинаковое время/число градиентов.

Оба fit завершены на RTX5070. В отчётах собственная fit timing1.36s PPO и17.96s SAC;границы таймеров разные,поэтому это не end-to-end speedup benchmark. У SAC это только offline warm update;пригодность дальнейшего online SAC требует свежего опыта его собственными обновлёнными весами. Нельзя выдавать один offline fit за проверенную экономность полного цикла.

Independent `combat-conditional-aim-pair-audit-v1-20261022.json` проверил обе модели на3450 actual source histories:все raw logits совпадают точно,разрешены изменения только conditional aim tensors/std[2:4]. Проверены JSON/checkpoint readback,Adam shapes/steps/finiteness;SAC сохраняет Q/targets/optimizers отдельно в `sac-checkpoint.pt`,legacy inference checkpoint не выдаётся за выполненные PPO updates. Переход SAC→PPO требует явного algorithm/scope fork. Продолжение произвольно прерванного SAC optimiser step пока не заявляется.

## Заранее закреплённая оценка

Fresh smoke32 native/24 CUDA:initial source,PPO aim,SAC aim,rules,по8 одинаковых условий/4ordinary+4recovery,offset644,порт46020,16 ×2. Группы четырёх карт,включая SSG/recovery,проверяют перенос;это не исчерпывающая SAC оценка. Никакого fitting/выбора checkpoints по этим боям. Frozen spec:`combat-conditional-aim-algorithm-spec-v1-20261022.json`;до запуска завершён independent CUDA pair audit.

Главное — выполнение цели первой жизнью,затем урон/смерти/timeouts,actual MG outcomes и лишние команды. Подавление остальных heads или замена их правилами не считается успехом полного learned боя. Promotion=false;дальнейший сбор своим SAC actor зависит от результатов,положительного эффекта пока не заявляется.
