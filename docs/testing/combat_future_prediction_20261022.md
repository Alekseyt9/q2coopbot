# Малый предиктор наблюдаемого будущего: первый результат

Статус: supervised pilot и повтор с другим training seed закрыты на CUDA.
Actor и его рабочий контракт не менялись. Это ещё не JEPA/EMA latent model,
MPC или результат улучшения победности.

## Что реализовано

`prepare_combat_future_dataset.py` создаёт компактный binary NPZ из закрытого
`combat-reactive-advice-corpus-v3-20261022`: 32 TRAIN / 16 development,
base3/city1/jail1/ware2, обычные/recovery группы. Использованы реальные действия
learner; советы учителя, reward, native hidden poses и будущие команды во вход
не попадают. Новых боёв не записывали.

Вход — 1147 канонических текущих features и 6 текущих applied action channels.
Метки: self displacement, displacement максимум 8 текущих observed tracks,
присутствие того же track в следующем наблюдении и clear-shot при его наличии.
Движение выражено в локальном базисе текущего yaw: forward/left/up.
Для enemy world displacement из относительной позиции исключается движение ego.

Соседние endpoints подтверждены оригинальной трассой и ровно одним exclusive
native100ms command window. Изменения map/life/actor, gaps, recovery, мёртвый
endpoint и отсутствующий реальный successor не превращаются в нулевые метки.
Enemy join использует entity+track и выдерживает перестановку distance slots.

| Часть | Переходы | Motion labels self+enemy | Current track labels | Track исчез | PMove subset |
|---|---:|---:|---:|---:|---:|
| TRAIN | 3350 | 9955 | 6674 | 69 | 2487 |
| Development | 1926 | 5187 | 3309 | 48 | 1439 |

Два сжатых массива занимают 1,853,244 bytes, около 1.77 MiB.
V1 builder остановился до создания набора на null successor: исключение
этого ожидаемого случая исправлено и добавлено в тест. V2 — первая завершённая
выгрузка. V3 дополнительно связана с оригинальными source/sequence hashes,
closed processing report SHA и индивидуальными CUDA receipts. **Все числовые
массивы V2/V3 совпадают**, что проверено в `binding-audit.json`.

Четыре data tests проверяют ego compensation, reorder/recycled track,
reset/gap/death/dispatch/null successor и текущий локальный базис.
Native report source hashes связывают реальные trace, steps, capture и server
log; canonical feature vectors связываются с исходным `sequence_sha256`.

## Фиксированное обучение

`train_combat_future_cuda.py`: MLP 1153→128→128→43, **169771 parameters**.
Motion head предсказывает residual к constant velocity; координаты нормированы
на 64 units. Loss: masked SmoothL1 beta0.1 +0.1 track BCE +0.1 clear BCE.
Output layer инициализирован нулём: начальный motion forecast точно равен CV.

В каждой паре action-conditioned и action-blind модели имеют одинаковые initial
weights и ровно те же TRAIN draws. Blind обнуляет последние 6 action channels,
сохраняя архитектуру. По 2000 AdamW steps, batch256, lr0.001, weight_decay0.0001;
gradient clip1, TF32 off, deterministic CUDA. Development не выбирает checkpoint.

Пары:

* `combat-future-pair-v1-20261022`, seed20261022, batch seed20261023;
* `combat-future-pair-replica-v1-20261022`, seed20261024, batch seed20261025.

Обе завершены с fixed final checkpoint; fit+metrics заняли 33.52/26.33 s.
Повтор менял только training seeds на том же числовом наборе. Уточнение guard
для empty-mask batches не повлияло: CUDA audit подтвердил, что такие batches
не встречаются ни в одной паре. Первый trainer и builder сохранены рядом
с исходными receipts для воспроизведения прежних script hashes.

## Development: прогноз движения

RMSE ниже — средняя ошибка на ось в Quake units за 100ms, не длина 3D-вектора.
Разные training seeds не означают два независимых development набора.

| Метод | Enemy RMSE | Self RMSE | Self при непустом movement/up |
|---|---:|---:|---:|
| Zero displacement | 6.027 | 7.126 | 7.710 |
| Constant velocity | 5.883 | 7.093 | 7.816 |
| Action-blind seed22 | 5.200 | 6.802 | 7.386 |
| Action-conditioned seed22 | 5.152 | 4.339 | 4.639 |
| Action-blind seed24 | 5.264 | 6.758 | 7.360 |
| Action-conditioned seed24 | 5.168 | 4.041 | 4.242 |

Enemy improvement к CV: **12.43% / 12.16%**. Case-weighted enemy RMSE:
CV5.960, action5.210/5.227. Улучшение на 15/16 и 16/16 cases; среднее лучше
CV на каждой из четырёх карт в обоих повторах.

Action conditioning заметно помогает self motion, а enemy forecast почти
так же хорошо работает без action. На горизонте100ms это не доказывает
готовность модели правильно оценивать произвольные новые манёвры.
При unknown enemy velocity первый seed хуже fallback0: 4.641 против4.478;
второй лучше:4.345. Устойчивого улучшения этой редкой страты не установлено.

## Физика и видимость: ограничения результата

На общей поддержанной ground-подвыборке1439 development rows:

| Метод | Self RMSE |
|---|---:|
| CV | 7.692 |
| Action-blind | 7.312 / 7.247 |
| Action-conditioned | 4.565 / 4.244 |
| Существующий Go PMove predictor | **0.060** |

PMove subset исключает ambiguous support и сохраняет собственные ограничения
dry-ground модели; это не результат для прыжков, воды и любого контакта.
Нейросетью заменять этот проверенный физический прогноз оснований нет.
Наблюдаемая зависимость от action полезна как sanity check, а не достижение
выше доступной физики движка.

`policy.Observe` исключает врагов с false/unknown ClearShot. Поэтому условный
clear-shot head на retained track имеет только positive labels: 6605 TRAIN,
3261 development. Его почти нулевой Brier — вырожденная задача, **не прогноз
потери видимости**. Runtime observation contract ради labels не расширялся.

Track disappearance — отдельная непростая задача, но improvement не воспроизвёлся:
Brier0.014425/0.018246 против persistence0.014506. Отсутствие track может
означать смерть, потерю наблюдения или смену доступного набора; не трактуем
его как доказанную причину. Visibility/risk MPC на этом основании не принимается.

## Решение и следующий опыт

`replication-audit-v2.json` закрыл hashes, одинаковые данные/бюджет,
CUDA checkpoints/draws, initial CV equivalence и два независимых training seeds.
Решение: `forecast_signal_replicated_actor_utility_unproven`.
Motion gate первого этапа пройден; actor/default/promotion сохранены,
влияние на реальные бои не измерялось.

Следующий этап: прочитать причинное attention representation **замороженного
исходного actor** на тех же current histories и обучить небольшой future decoder
на этих latent+current-action входах. Сначала сравнить с raw-feature predictor
и CV на тех же held-out rows, без изменения actor и без новых игровых прогонов.
Это определит, отсутствует ли полезный forecast signal в representation или
уже содержится в нём. Effective decoder input/parameter counts указывать явно:
сжатие входов нельзя выдавать за идеально совпадающую ёмкость.

Дальнейшее auxiliary обучение самого actor потребует отдельной matched
PPO пары, одинакового свежего TRAIN опыта/бюджета и новых native source/rules
боёв. Дегенеративный clear-shot head в полезный auxiliary signal не включать;
известную self physics сохранять как геометрический прогноз. Latent forecast
loss, горизонты3/5, EMA/anti-collapse и планирование остаются отдельными,
пока не проверенными изменениями.

Продолжение выполнено: [frozen actor readout и normalization control](combat_future_latent_readout_20261022.md).
На двух seeds latent readouts остались около CV, тогда как matched raw
comparator лучше. Следующий этап — отдельная matched PPO auxiliary пара
и новые игровые оценки; actor пока не изменён.
