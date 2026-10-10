# Blaster launch safety: crouch и движение

Причина двух событий в `natural-companion-leader-smoke-v3-20261010`,
base2 learned, actor2 → leader1:

- Shot14: spawn server_frame93, попадание в barrel324 frame94,
  barrel explosion frame96 убивает ведущего (100 live health damage).
- Shot25: spawn frame135, попадание в ведущего frame136 (10 damage).

В обоих предложениях Up=-400. Guard считал прежнюю standing EyePoint,
но PM_CheckDuck применяется перед стрельбой. Реальный muzzle становится
ниже24 units; stock Blaster имеет ещё24 forward,8 right,viewheight-8.
Guard не покрывал смену стойки и перемещение origin до исполнения команды.

Добавлен общий boltLaunchPaths для Blaster: command crouch, обе высоты
при отпускании crouch под возможным потолком, native muzzle offset,
наблюдаемая own velocity с нижней границей320units/s, не менее100ms
на исполнение и4units quantization margin.

Barrel guard сохраняет прежнюю проверку и добавляет padded muzzle paths,
включая короткий путь от игрока к muzzle. Новая проверка не сокращается
старой трассировкой из прежнего EyePoint: она могла отсечь другую траекторию.
Это консервативная защита; unseen/custom barrels не покрываются.

Новый guardBoltTeammate проверяет body sphere и перемещение обоих игроков
за время команды, свежую/недавно исчезнувшую позицию, а также существующую
оценку пересечения с летящим снарядом по наблюдаемой скорости. Защита
применяется и к rules, и к learned командам. Меняется только attack bit;
aim, weapon и движение остаются предложенными контроллером.
Она не доказывает безопасность неизвестных игроков, произвольной задержки,
поворотов после запуска или всей жизни снаряда за пределами моделируемого пути.

Geometry regressions прошли: оба записанных crouch commands запрещены,
движение сохранено, удалённый напарник вне траектории разрешает fire,
posture/muzzle positions проверены. Прежние barrel chain, projectile friend,
hitscan и campaign leader проверки прошли. Artifact:
`coop-hitscan-guard-fix-v1-20261010/bolt-launch-regressions.jsonl`.
CPU neural numerical tests и обучение не запускались.

## Native повтор

Artifact `natural-companion-bolt-smoke-v4-20261010`: base2 rules/learned,
seed184110,1200frames,timescale2,skill1, та же опубликованная attention64,
Stochastic;2 сервера ×2 клиента, stock monsters/barrels, без выдачи оружия.
source_unchanged=true; infrastructure_ok=true у обеих сцен.

| Companion | Friendly damage | Reverse friendly damage | Смерти | Убийства | Monster health damage | Provider frames |
|---|---:|---:|---:|---:|---:|---:|
| rules | 0 | 0 | 1 | 2 | 60 | 0 |
| learned | 0 | 0 | 1 | 1 | 20 | 338 |

Learned native client отправил151 attack commands; огонь не полностью
заблокирован. Оба уровня не завершены. В прежней learned сцене было110
friendly damage,0 смертей companion и40 monster damage; новая сцена
не доказывает общего выигрыша качества. Один seed и real-time scheduling
не дают детерминированного сравнения trajectories.

Следующее: несколько seeds на обеих картах, диагностика потери ведущего,
проверка дополнительной задержки fire/движения и сохранения damage rate.
Веса не заменены; человеческий сервер1976 и бот3556 не перезапускались.
