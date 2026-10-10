# Где остаются ошибки боевой модели

Источник: закрытая400-бойная development оценка
`workspace/artifacts/native-waste-parent-rules-eval-v1-20261010/quality-report.json`.
Parent — сильный v8 update5, candidate — v9 update6 из reset A/B.
Это не результаты текущего warm A/B.

В parent/candidate приведена сумма двух policy RNG arms:8 запусков,
но только4 независимые сгенерированные сцены на семейство. Rules имеет
4 запуска тех же сцен. Малое число сцен не доказывает статистическую
устойчивость различий. Reward totals не используются для выбора победителя.

| Семейство | Parent побед /8 | Candidate побед /8 | Rules побед /4 |
|---|---:|---:|---:|
| parasite-gunner-blaster-generated |0|0|0|
| parasite-gunner-machinegun-recoil |3|3|0|
| campaign-base1-site-01-blaster |1|4|4|
| campaign-base1-site-01-machinegun |7|6|4|
| campaign-base1-site-03-machinegun |7|6|0|
| campaign-base1-site-04-blaster |7|5|4|
| campaign-base2-site-04-blaster |8|8|2|
| campaign-base2-site-04-machinegun |8|8|2|

В остальных12 семействах все варианты выиграли все свои запуски.
Итого parent137/160, candidate136/160, rules64/80.
Улучшение candidate на base1 site01 Blaster сопровождается потерями
в других сценах; общее преимущество над parent не установлено.

После текущего warm A/B отдельно проверить эти семейства. Приоритет
дальнейшего curriculum — mixed Parasite/Gunner с обоими вооружениями,
base1 site01 Blaster и сохранение site04 Blaster. Это вывод о том,
где наблюдались неудачи, а не доказательство их причины. Для разделения
недостаточной точности, неподвижности, опасного отступления и timeout
нужны actual native shots/damage и временные трассы конкретных проигрышей.

Новые training сцены брать из train split и новых seed offsets;
не переносить эти validation траектории в обучение. Легкие семейства
сохранить для проверки деградации прежних навыков. Архитектуру и награду
одновременно с изменением состава curriculum не менять. Итогового
test acceptance или live promotion эта оценка не подтверждает.

## Первый life сильного parent в mixed сценах

Read-only разбор8 старых captures (RNG arm a,4 сцены на оружие) сохранен
в `native-waste-parent-rules-eval-v1-20261010/mixed-first-life-review.json`.
Он не поставляет обучающие labels или наблюдения политики.
Blaster:3 смерти,1 timeout с50 HP,0 kills во всех4 сценах.
Machinegun:2 полные победы и2 смерти без kills. В проигрыше730028
урон88 Parasite/162 Gunner, входящий73 от Gunner/9 от Parasite;
в730031 урон72/72, входящий80 от Gunner, смерть за35 наблюдаемых frames.
В этих двух проигрышах средняя горизонтальная ошибка команд к ближайшему
clear enemy2.52° и4.73°, ground-move-under-1unit proxy0.
Это не точность native shots: метрика не учитывает pitch, recoil, lead
и не гарантирует выбор конкретной цели. Данные показывают необходимость
проверить завершение убийства и опасность Gunner наряду с прицеливанием;
они не доказывают единственную причину неудач.
