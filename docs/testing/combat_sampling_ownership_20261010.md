# Владение управлением в stochastic пилоте

Разобраны56 sealed offset-pilot captures. Reporter
`scripts/report_combat_control_ownership.py` сопоставляет каждую
исполненную native команду steps.jsonl с selection из bot.jsonl по полной
observation identity. Учитывается первая наблюдаемая жизнь и живое
состояние перед командой. Несопоставленные/неоднозначные selections
отклоняются. Подготовительные callbacks, которых нет среди выполненных
dataset steps, не включаются. Нейросеть не исполняется.

| Вариант | Provider кадров | Rules кадров | Rules кадров с clear target | Из них attack отправлен |
| --- | ---: | ---: | ---: | ---: |
| m1 deterministic | 838 | 205 | 0 | 0 |
| m1 stochastic-a | 460 | 29 | 9 | 9 |
| m1 stochastic-b | 503 | 324 | 9 | 9 |
| parent3 deterministic | 1314 | 0 | 0 | 0 |
| parent3 stochastic-a | 545 | 44 | 0 | 0 |
| parent3 stochastic-b | 590 | 12 | 0 | 0 |

У m1-a equip fallback13 кадров, у m1-b14. Оружие во всех этих кадрах —
`models/weapons/v_shotg/tris.md2`. Это фактически Shotgun, а не
неизвестное промежуточное имя оружия. Видимая цель и rules attack есть
в победном Blaster-start случае seed1650026 у a и seed1650025 у b.
Это не доказывает причинный вклад правил в победу или принадлежность
конкретного delayed damage выстрелу. Но эти бои нельзя представлять как
полностью learned управление каждым боевым кадром.

Причина в текущем combatCommand: synchronous Blaster/Machinegun fixture
разрешает сетевое управление только исходными пилотными типами оружия;
Shotgun допускается для multiweapon fixture или campaignEvaluation.
Наблюдаемый инвентарь и weapon head позволяют выбрать Shotgun, после
чего gate с причиной pilot_equip_not_ready передаёт команду правилам.
Указанное расхождение требует исправления после завершения текущего
source-bound широкого сбора. Исходники Go/PS сейчас не меняются.

Остальные rules кадры относятся к system2_noncombat и
combat_visibility_timeout. Они отделены от кадров с наблюдаемой clear
целью. Нулевая clear цель не доказывает отсутствие живых монстров вне PVS.
В частности, доля rules по всем кадрам не является самостоятельной
оценкой помощи правил во время видимого боя.

Медиана первого запрошенного attack у m1: deterministic16 кадров,
stochastic-a2.5, stochastic-b4. Но движение остаётся ограниченным:
provider-requested movement stopped fractions0.772/0.737/0.724.
Основные зарегистрированные причины — static_hull_blocked и
unsupported_motion_guard. Это наблюдаемые решения guard, не доказательство
единственной причины проигрыша или ошибок BSP. У parent3 stationary
fractions0.894/0.789/0.782; PPO/sampling ещё не устранили стояние.

Снимок хранения текущего wide:134 sealed/compressed streams,
325004282 логических байта и113827840 физических; SHA всех проверены.
У36 завершённых участников client/exporter по одному inode на тип.
Это ограниченный снимок, не оценка всего дерева или окончательная экономия.

Широкий560 сбор остаётся running без изменения условий. Дополнительный
analysis driver удерживает OS handle процесса21984; после terminal и
всех core seals выполнит такой же ownership разбор всех вариантов и
конечный physical-storage audit. Качественные выводы широкой оценки
пока не подтверждены. Наличие rules attack при видимой цели должно быть
явно раскрыто и устранено до признания полностью самостоятельного боя;
одних побед выше правил для этого недостаточно.

Далее: расширить допустимое прямое learned управление на поддерживаемый
Shotgun для этих fixtures, проверить собственные атаки/смены оружия и
отсутствие equip fallback в новом живом прогоне; затем повторить парную
оценку. Сеть должна сохранять выбор действий, безопасность и native
dispatch остаются проверяемыми ограничениями. Независимый test не
используется для настройки этого исправления.

Пилот: `workspace/artifacts/action-sampling-offset-smoke-v1-20261010/control-ownership.json`.
Storage snapshot: `workspace/artifacts/action-sampling-offset-wide-v1-20261010/storage-snapshot-01.json`.
Очередь анализа: `scripts/finalize_combat_sampling_analysis.py`.

## Подготовленное исправление

`queue_combat_pickup_ownership_fix.py` подготовил проверяемый patch двух
файлов: combat_policy.go и multiweapon_fixture_test.go. `git apply --check`
прошёл; SHA исходников и patch сохранены. Сейчас исходники не изменены.
Driver удерживает process handle текущего560 evaluation и применит patch
только после terminal процесса, complete quality и девяти diagnostics seals.
Перед применением проверит исходные SHA, чтобы не переписать параллельную
правку пользователя. Git commit/push не выполняются.

В synchronous parasite_blaster/parasite_machinegun/multiweapon fixtures
управление поддерживаемыми Machinegun и Shotgun остаётся у provider.
Ограничения вне этих fixtures сохраняются. Проверки допуска и barrel
guards используют фиксированный probeProvider и не выполняют нейросеть;
пропуски тестов не принимаются. Затем автоматически запускается свежий
56-case native validation с прежними frozen weights и engine conditions,
раздельными policy RNG offsets, pool16 ×2 и поэтапным LZX.

Завершение исправления требует полного provenance/quality/diagnostics
и ownership audit: ни equip fallback, ни rules-with-clear-target у learned
вариантов. Проверка ещё не выполнена; успех нельзя выводить из одного patch.
На данный момент evaluation и оба ожидающих процесса подтверждены живыми.

Корень: `workspace/artifacts/pickup-ownership-fix-v1-20261010`.
Patch: `pickup-ownership.patch`; source hashes: `preparation.json`.

Для последующих sampling запусков ownership audit включён в общий driver,
его SHA и число visible-target rules frames сохраняются в progress.
Допуск к широкому сравнению теперь требует sealed smoke ownership report,
его связь с quality/protocol и отсутствие equip fallback/visible-target
rules frames у всех learned вариантов. Старый уже выполняющийся560
когортный сбор не переписывается и остаётся историческим сравнением.
Проверка реальным contaminated smoke прошла: driver отклонил его с
`Visible-target rules fallback must be resolved before wider comparison`
до compilation/dispatch. Native pool не создан, нейросеть не исполнялась.
Evidence: `workspace/artifacts/ownership-gate-negative-v1-20261010/gate-check.json`.
