# Проверка native ground prediction, 2026-10-12

Поведение guard и веса не менялись. Добавлены decoder PM_TYPE/PM_TIME/PMFlags, признак известного состояния и diagnostic logging. Старые actor features остаются неизменными. Predictor считает полную3D скорость для PM_Friction, потом ground acceleration, native18-unit step/slide, observed translated/rotated brush models. PM timers, не-ground/не-normal, прыжок и неподдерживаемая поверхность исключаются. Это диагностический model, не разрешение движения; невидимые двери, динамические actors, изменённая серверная физика и PM_SnapPosition ещё не покрыты полностью.

## Живые проверки

Первая версия (static only):24 native/16 CUDA,165.95с. Из1525 next-frame пар813 с командой движения;65 расхождений позиции больше1. На47 контактных кадрах maximum0.162. Ошибка у jail1 door frame194 достигала30: brush model57 игнорировался. Сохранена точная small regression fixture; с observed mover предсказание отличается от native next position меньше0.25, без mover снова больше20. Эти данные не использованы для NN обучения.

Observed-mover версия:8 rules native за39.24с. Все738 измеренных пар неподвижны, поэтому zero error не является проверкой движения. Затем16 learned native/16 CUDA cases закрыты за133.69с, 2626 PPO verification rows. Обучения/продвижения нет. Всего24 native боя версии2, восемь условий по четырём картам; diagnostic datasets не сравнивать как оценку выигрышей старой/новой модели, seeds отличаются от первой версии.

В learned версии2 измерены1909 пар, из них1512 с командой движения: position error p95=0.15544, maximum28.406;29 пар больше1. На87 кадрах контакта корпуса position maximum0.17780, ни одного выше0.25; velocity maximum102.14 показывает, что хорошая позиция не доказывает полную динамическую согласованность. Есть native подъём на16 при предсказании flat пути и боковые блокировки без observed brush contact. Причина этих оставшихся случаев не установлена; candidates — неучтённые actor/brush collisions или applicability/timing. Не включать guard, опираясь только на p95.

## Проверки и следующий шаг

Targeted decoder tests покрывают full/delta/reset/truncation PM fields плюс velocity/gravity. Go physics tests покрывают trace/slide/step, translated/rotated observed mover, unknown model rejection и recorded door frame. Первый запуск real-map bot tests использовал неправильный asset root; исправлен на assets/baseq2, после этого тесты прошли. Четыре Python telemetry tests исключают duplicate/gap, damage, airborne, pre-release и следующий life. Они не исполняют NN; численные проверки моделей в обоих learned smoke только CUDA.

Следующий этап: сохранить и классифицировать оставшиеся29 frames через native collision telemetry (blocker entity/model, touchents, step). Затем ограничить applicability или добавить нужные collision bodies и проверить исходный command целиком, динамические hazards и floor support. Только после native проверки подключить guard и провести matched короткое сравнение со старыми весами.

Artifacts: combat-pmove-ground-diagnostic-smoke-v1-20261012; combat-pmove-movers-native-v2-20261012; combat-pmove-movers-learned-native-v2-20261012; два combat-pmove-movers-learned-processing-*-v2-20261012. Все raw traces активного цикла сохранены.
