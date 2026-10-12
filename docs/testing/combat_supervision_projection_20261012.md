# Быстрое чтение трассы для харнеса: прототип

Во время conditional weapon confirmation пул сохранял 16 рабочих слотов, но ресурсный монитор часто показывал 8–10 одновременно работающих native servers при CPU100%. Слоты продолжали брать новые задачи, завершения подтверждались, зависания не обнаружены. В коротком замере CPU основные потребители — PowerShell coordinator и дочерние процессы харнеса. Это наблюдение текущего запуска, не доказательство единственной причины задержек.

В `run_learned_combat_baseline.ps1` после capture каждый JSONL row целиком преобразуется через `ConvertFrom-Json`, включая массивы памяти нейросети. Supervisor cursor также преобразует целые завершённые строки. Проверка capture/goal/death использует ограниченный набор полей. [Прототип](../../scripts/combat_supervision_projection.cs) сначала проверяет полный JSON через System.Text.Json, затем создаёт компактную supervisor projection. Этот C# helper относится к PowerShell-харнесу; политику, игровой протокол и решения Go companion он не изменяет. Нейросеть не исполняется.

Сохраняются root health/map/connection/spawncount/observation_frame/self_entity/sent_command; combat provider/applied command/command timestamp; observation version/identity/health; applied version; selection owner/mode/candidate command/interventions/latency. Полная исходная трасса остаётся источником обучения, аудитов и наблюдений. Projection предназначена только для supervisor и capture metadata; её нельзя передавать вместо исходного JSONL в экспортёр.

## Проверенная часть

`workspace/artifacts/check-combat-supervision-projection-20261012.ps1` проверил шесть завершённых train captures: по выигранному и проигранному бою control1/conditional2/conditional4. Всего1269 rows, положительные native goal receipts в3 случаях и death receipts в3 случаях. Старый и новый разбор дали одинаковые:

- counts captures/mismatches/provider/shadow, interventions и latency samples;
- active seconds, root identity/health и sent commands каждого кадра;
- goal и first-life death receipts на каждом кадре, включая положительные случаи;
- SHA исходных traces после чтения.

Невалидный JSON в отброшенном поле также отклонён: parser проверяет всю строку перед projection. Receipt первой версии `combat-supervision-projection-check-v2-20261012.json`, исходный первый трёхбойный receipt и C# source сохранены отдельно. Первоначальный tracked C# source побайтно совпадал с этой версией; дальнейшее усиление описано ниже.

В локальном замере шести traces полное чтение/PowerShell conversion заняло6,50с, projection/PowerShell conversion —1,24с. Компиляция helper не входит в эти времена; она должна происходить один раз на процесс. Замер выполнен под параллельной полной нагрузкой текущего пула и не доказывает ускорение всей игровой оценки или обучения. Поведение других архитектур/rules и свежая live supervisor приёмка здесь не проверены.

## Следующее внедрение

Текущие `.ps1` и native исходники заморожены до закрытия confirmation и её CUDA/audit receipts. Прототип пока не подключён к рабочему харнесу. После закрытия этого эксперимента:

1. Сохранить helper в source provenance, подключить его один раз на процесс и заменить только supervisor/capture metadata чтение; исходные traces, exports, reward, policy и рецепты сохранять.
2. Проверить partial UTF-8/partial JSON, несколько append порций, первый death перед respawn и задержанный native damage log; строгие errors/version/dispatch проверки не ослаблять.
3. Проверить правила и learned control на нескольких map/loadout fixtures, закрыв goal/death/seed/reset/dispatch/native source receipts; neural verification только CUDA.
4. Измерить весь fresh smoke/pool, включая helper startup и число одновременно живых серверов. Только после этого использовать обновлённый харнес для следующих собственных train корпусов.

Изменение metadata reader не снимает ограничения движения и оружия и не подтверждает superiority модели. Отдельный execution audit текущей confirmation должен показать фактические reasons подавления предложенных действий.

## Проверка append и совместимости ошибок

Вторая версия tracked helper сохраняет case-insensitive field lookup PowerShell и проверяет весь JSON tree на пустые имена и имена, различающиеся только регистром. Так projection не скрывает ошибки, которые старый `ConvertFrom-Json` обнаружил бы в отброшенном поле. Поддержка этих имён не применяется к значениям protocol/version: они сохраняются без изменений.

Изолированная копия существующего `Read-CombatTraceRows` использует новый converter, оставляя алгоритм bytes/offset/pending прежним. `combat-projection-append-check-v2-20261012/complete.json` подтверждает:

- частичные UTF-8 bytes и JSON не выдаются до завершённой строки; повторное чтение не дублирует кадр;
- чтение порциями7 bytes сохраняет порядок death→respawn и удерживает первую death observation до прихода native damage;
- respawn не становится квитанцией первой смерти, truncate уже прочитанной трассы отклоняется;
- неверный JSON, case-colliding и пустые имена отклоняются даже в отброшенном subtree;
- известные поля с другим регистром сохраняют прежнюю PowerShell семантику.

Первая попытка запуска изолированного теста остановилась на синтаксисе объявления cloned function до чтения данных; исходный partial test directory сохранён. Вторая попытка с корректным function body закрылась успешно. Working `.ps1` не менялись.

После усиления повторена проверка всех шести прежних real train captures/1269 rows: metadata/commands и положительные goal/death receipts вновь совпали, source SHA совпадает с append receipt. `combat-supervision-projection-check-v3-20261012.json` фиксирует новый локальный замер3,70→1,56с. Его нельзя напрямую сравнивать с предыдущими6,50→1,24с: системная нагрузка изменилась, добавлена проверка дерева. Это ускорение стадии чтения в этом замере; live pipeline/startup overhead пока не проверены. Helper остаётся прототипом до fresh native приёмки после текущей frozen evaluation.
