# Mixed curriculum: Temporal attention Update28 → 29

Дата:07.10.2026. Пул16, независимый seed каждого боя, x2; native server и Go harness. Учебная партия80 боёв/20 семейств, train offset12. Validation offset4: одни и те же80 условий для двух версий; эти validation seeds не использовались при обновлении. Геометрия знакомая: это диагностическая парная оценка, не location holdout и не полное прохождение кампании.

CUDA-only update:7509 PPO transitions,85 sequences,10 actor steps,279 total actor steps,29 completed updates. Архитектура combat_causal_attention_v1. Capture и evaluation полностью проверены; completion receipt weights/checkpoint/report SHA подтверждён. После прерывания экспорта данные заново проверены в fresh resume root, бои не повторялись.

| Метрика | Update28 | Update29 |
|---|---:|---:|
| Победы |39/80|39/80|
| Смерти первой жизни |39|37|
| Убийства |42|42|
| Полученный health damage |4520|4386|

Снижение суммарного урона:134 (2.96%). Старые4 семейства: победы8/16→9/16; кампанийные16:31/64→30/64. Победы в целом не выросли. На4 seeds/семейство устойчивый эффект не доказан.

| Семейство | Победы до→после | Смерти до→после | Убийства до→после | Урон до→после |
|---|---:|---:|---:|---:|
|parasite-blaster-generated|4→4|0→0|4→4|251→267|
|parasite-gunner-blaster-generated|0→0|4→3|0→0|360→285|
|parasite-machinegun-recoil|3→4|1→0|3→4|182→128|
|parasite-gunner-machinegun-recoil|1→1|3→3|4→4|310→354|
|campaign-base1-site-01-blaster|0→1|4→3|0→1|380→305|
|campaign-base1-site-01-machinegun|2→1|2→3|2→1|174→288|
|campaign-base1-site-02-blaster|3→3|1→1|3→3|111→111|
|campaign-base1-site-02-machinegun|4→4|0→0|4→4|69→55|
|campaign-base1-site-03-blaster|3→1|1→3|3→1|262→372|
|campaign-base1-site-03-machinegun|0→0|4→4|0→0|360→360|
|campaign-base1-site-04-blaster|3→4|1→0|3→4|196→168|
|campaign-base1-site-04-machinegun|0→0|3→3|0→0|314→268|
|campaign-base2-site-01-blaster|4→4|0→0|4→4|55→25|
|campaign-base2-site-01-machinegun|2→2|2→1|2→2|248→152|
|campaign-base2-site-02-blaster|0→0|4→4|0→0|360→360|
|campaign-base2-site-02-machinegun|0→0|3→3|0→0|280→280|
|campaign-base2-site-03-blaster|1→1|3→3|1→1|280→280|
|campaign-base2-site-03-machinegun|4→4|0→0|4→4|0→0|
|campaign-base2-site-04-blaster|4→4|0→0|4→4|48→48|
|campaign-base2-site-04-machinegun|1→1|3→3|1→1|280→280|

Полная победа означает достижение цели живым в первой жизни. В группах количество убийств может превышать победы: бот способен убить одного монстра и проиграть второму. Таймаут без смерти не засчитывается как победа. Capture accepted означает пригодность данных, а не победу.

Изменения по победам: Parasite/Machinegun3→4; base1/site01/Blaster0→1; base1/site04/Blaster3→4; base1/site01/Machinegun2→1; base1/site03/Blaster3→1. Остальные семейства без изменения числа побед. Parasite/Gunner Blaster остаётся0/4, base1/site03/Machinegun0/4, base2/site02 оба оружия0/4.

Дальше: сначала разобрать поведение в нулевых и ухудшившихся семействах; проверить полезные aim/fire transitions и расстояние/геометрию; адаптировать curriculum для сложных групп и углов; проверить удержание старых навыков на новых seeds; затем независимая location holdout и transfer на полные base1/base2. Простое увеличение числа updates пока не обосновано результатом.

Новая модель сохраняется экспериментальной: superior to rules не доказано, автоматического переключения live/default нет. Веса/checkpoints остаются локально в игнорируемой workspace/artifacts, .gitignore не изменён.

[Итог experiment](../../workspace/artifacts/combat-mixed-registry-resume-v3-20261007/report.json), [capture pool](../../workspace/artifacts/combat-mixed-registry-resume-v3-20261007/epoch-1-pool/report.json), [evaluation pool](../../workspace/artifacts/combat-mixed-registry-resume-v3-20261007/evaluation-pool/report.json), [CUDA report](../../workspace/artifacts/combat-mixed-registry-resume-v3-20261007/temporal28/epoch-1/update/report.json).

Воспроизводимость продолжения capture описана в [instance pool](combat_instance_pool_20261007.md). Продуктовые Go tests cmd/... и internal/... прошли; PowerShell parser, native build, fresh on-policy re-export и live receipts прошли. Полный общий go test ./... сохраняет известные ограничения diagnostic artifact packages.
