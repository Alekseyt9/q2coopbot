# Продолжение PPO HP60: вертикальное удержание прицела

06.10.2026. Предыдущий HP60 checkpoint сохраняется как родитель: SHA256 weights `e85ff19b8000f4bb173b6ab81221f772769a7399efaf940c67780851bc405763`. Новый цикл: `workspace/artifacts/combat-ppo-tracking-hp60-v2-20261006`.

До запуска зафиксированы четыре fresh on-policy batch по четыре инстанса, x2, training seeds 15300–15315, отдельный seed каждого эпизода. 300 post-barrier игровых кадров, release frame 100, стартовый Blaster gunframe 9, post-frame seed reset, fixture HP60. Reward v2, PPO config, архитектура, optimizer/RNG/history продолжаются без изменения. Eval: исходный и фиксированный четвёртый checkpoint, обычные 175 HP, четыре одинаковых held-out seed 15400–15403; eval не используется для обучения или выбора промежуточного checkpoint. Full engine repeatability не доказана.

`scripts/diagnose_combat.py` v2 дополняет прежний yaw вертикальным отклонением и углом между направлением взгляда и направлением от наблюдаемого глаза к наблюдаемому origin ближайшего clear-shot монстра. Используются **applied** углы, eye offset из Go `Snapshot.EyePoint` (+22 стоя, −2 пригнувшись). Метрика origin не является точностью выстрелов: observation v3 не содержит enemy bbox/upper-body AimPoint; muzzle offset, полёт снаряда и упреждение не учитываются. Отдельно считаются кадры атаки с abs(pitch) ≥88° и реальные `protocol_pitch_limit` interventions.

Повторная offline диагностика предыдущего HP60 eval сохранена отдельным `diagnostics-aim-v2.json`; исходная v1 не перезаписывается. До → после: origin pitch error >15° — 1108/1188 → 911/1180; 3D origin angle >15° — 1124/1188 → 978/1180; возле pitch limit — 701 → 442. Существенная ошибка сохраняется, хотя эти показатели снизились. Для новой диагностики прошли семь тестов: знак pitch, высота глаза/crouch, applied delta, yaw wrap, 3D angle, вырожденная цель, life/frame/ownership границы.

## Завершённый цикл

24 capture прошли native provenance/dispatch/seed checks; SHA reports/steps и first-life kill counts сохранены в `capture-audit.json`. Training first-life kills: **0 из 16**. В PPO вошли 1153/1032/739/954 = **3878** свежих переходов; actor steps 10/10/10/10, KL 0.009419/0.006610/0.009867/0.009682. CPU быстрее CUDA во всех четырёх benchmark. Накоплено 29 updates/288 actor steps. Фиксированные финальные weights SHA256 `97651633277075a74f3ac041d2ce244a03321397fb800a0fb022a8eff5eb5c6d`.

| Eval seed | Исходящий урон до → после | Полученный урон до → после | Kills до → после |
| --- | ---: | ---: | ---: |
| 15400 | 20 → 20 | 9 → 75 | 0 → 0 |
| 15401 | 20 → 20 | 30 → 85 | 0 → 0 |
| 15402 | 20 → 20 | 30 → 75 | 0 → 0 |
| 15403 | 20 → 20 | 11 → 80 | 0 → 0 |

Суммарно outgoing 80→80, incoming 80→315, first-life kills/deaths 0→0. Эти native показатели описывают первую жизнь всего capture, включая интервалы передачи управления rules после guard. Во всех четырёх after эпизодах есть `control_handoff` с `unsupported_motion_guard`; их нельзя объявлять непрерывным learned-only боем. PPO экспортирует только проверенный начальный provider segment, diagnostics считает все provider frames первой жизни, включая возвраты управления.

Provider observations 1204→1034; visible attack origin samples 1183→968. Pitch error >15°: 921/1183 (**77.9%**)→810/968 (**83.7%**); 3D origin angle >15°: 981/1183 (**82.9%**)→942/968 (**97.3%**). Pitch возле ±89°: 477/1183→373/968; реальные pitch guard interventions 439→363. Падение абсолютных счётчиков нельзя назвать улучшением: число provider frames уменьшилось. Requested stationary frames 916/1204→523/1034 тоже сравниваются на разной длине управления.

Результат отрицательный: продолжение прежнего PPO не решило уход прицела и добивание. Финальный checkpoint сохраняется как эксперимент, без выбора лучшей промежуточной модели и без live promotion. Следующее изменение должно адресовать наблюдение/обучение удержания прицела и проверить handoff, а не автоматически увеличивать бюджет той же настройки. GRU/attention остаются планом, в этом опыте не реализованы. Семь focused Python tests прошли; исходники Go/native не изменялись. Harness порты 33100–33103 освобождены, пользовательский live marker сохранён.
