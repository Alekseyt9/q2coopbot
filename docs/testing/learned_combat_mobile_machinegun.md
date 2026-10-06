# Подвижные монстры, Machinegun: исправленный цикл

2026-10-06. После просмотра демки обнаружены два ограничения прежнего Mixed fixture: монстры были startsolid на плоскости пола z24 и могли переключаться друг на друга после союзного урона. Старые architecture tables сохраняются как исторические результаты с неподвижными противниками.

В текущем harness стандартный Parasite и Mixed Gunner появляются при z24.125. Отдельный native повтор с Blaster подтвердил движение обоих; Temporal10 погиб без убийств. Учебное снаряжение нового цикла — Machinegun с100 bullets, stock monster HP и skill1.

Нативный game module поддерживает `g_test_monster_no_infighting=1` только для single-client cheat combat fixture с barrier. `T_Damage` подавляет monster→monster контакт перед knockback, pain, health damage и retargeting; player→monster и monster→player остаются штатными. Корпуса монстров продолжают перекрывать траектории. По умолчанию опция0. Startup acknowledgement и подавленные контакты записываются в server log; готовность опции включена в batch manifest.

Скрипт `scripts/run_temporal_mobile_training.ps1` запускает четыре worker с независимыми seeds30000–30015, x2, horizon300 после release100. Начальные веса — Temporal attention после update10, используемые только как warm start. Четыре новых PPO updates, свежий Adam, CUDA; optimizer/resume наследуются только внутри нового цикла. Python modules заморожены в output/python-sources. Старые stationary retention losses отключены (`retention_weight=0`, `bank_weight=0`); anchor/bank сохраняются лишь для provenance и диагностик, без вклада в loss. Resume проверяет неизменность этих весов loss.

Перед каждым CUDA update batch обязан подтвердить: native provenance, одну source/native revision, acknowledgement no-infighting, отсутствие monster→monster health damage, горизонтальное движение обоих типов во всех четырёх эпизодах. Проверка движения может консервативно остановить цикл, если противник погиб до наблюдаемого перемещения; такой случай требует анализа, а не ослабления проверки.

После четырёх updates начальный и итоговый checkpoint оцениваются на одинаковых отдельных seeds30400–30403, deterministic, с тем же оружием и исправленным fixture. Четыре evaluation seeds — диагностический объём, недостаточный для общего вывода об архитектурах. Выбор оружия сеть пока не обучает: Machinegun выдан и фиксирован сценарием.

Артефакты: `workspace/artifacts/combat-mobile-machinegun-v1r2-20261006`; coordinator log и err лежат рядом. Незавершённая pre-game попытка `combat-mobile-machinegun-v1-20261006` сохранена отдельно, в этот цикл не входит.

Проверки перед запуском: native game rebuilt через CMake target game; CUDA architecture resume integration test прошёл; PowerShell parser прошёл. Native мобильность/no-infighting и CUDA update подтверждаются только соответствующими receipt/reports завершённых этапов.

## Результат первого блока

Цикл завершён:24 valid captures (16 training +8 evaluation),4 CUDA updates,39 accepted actor steps,1567 eligible training rows. Native provenance един для всех batch; горизонтальное движение обоих типов и отсутствие monster→monster damage подтверждены во всех24 эпизодах. Native logs фиксируют подавленные союзные контакты. Независимый `audit.json` проверил rollout consumption, hashes, trainer snapshot, resume chain, retention0/0, Adam actor clock39/value160 и GPU parity/KL. `accepted:true` в audit относится к целостности опыта, не к качеству политики.

| Seed | Урон до | Урон после | Убийства до/после | Исход до/после |
|---|---:|---:|---|---|
|30400|104|128|0 /0|гибель /гибель|
|30401|40|40|0 /0|гибель /гибель|
|30402|96|56|0 /0|гибель /гибель|
|30403|48|96|0 /0|гибель /гибель|

Победы0/4→0/4, убийства0→0, смерти4→4, суммарный first-life outgoing health damage288→320. Первый короткий блок не научил убивать эту группу. Выдача Machinegun решает снаряжение, но не заменяет обучение прицеливанию/манёвру против преследования. Это новая мобильная baseline; live promotion нет. Следующий обучающий блок должен оставаться на исправленном fixture, с GPU и свежими seeds; тренировочное одиночное упражнение можно выделить отдельно, чтобы получать kill transitions, не используя эти evaluation seeds для fit.

Демка итоговой оценки seed30400: `evaluation-after/worker-0-episode-0/bot-smooth.dm2`. Повторы synchronous snapshots удалены с проверкой точного per-tick state; оригинальная запись сохранена. Она показана в portable Yamagi windowed1920×1080, x1, по кругу; F5 повторяет. Демонстрируется фактическая гибель итоговой политики, а не выбранный удачный training episode.
