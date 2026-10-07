# Masked weapon head: Go, CUDA likelihood и native switch

07.10.2026. Реализован versioned `combat_masked_weapon_v1`: actor20 = прежние8 outputs +12 weapon logits. Go Sample содержит weapon category, Action.Weapon получает соответствующее имя. Mask применяется до softmax; Review отклоняет недоступные категории и включает stable masked logprob в общий likelihood. Legacy actor8 и его RNG stream сохраняются; weapon head требует V6 и не допускает entity attention. Temporal/GRU residual shape соответствует ширине actor, совместное Temporal+GRU по-прежнему не разрешается.

Python `combat_weapon_head.py` содержит тот же joint likelihood и entropy. Recurrent CUDA trainer подключён к новой голове и V6 masks; legacy samples не превращаются в weapon samples. Retention с legacy8 anchor сохраняет только прежние stochastic heads; teacher20 дополнительно требует masked weapon KL. Старый feedforward trainer явно отклоняет weapon head, чтобы не интерпретировать12 weapon logits как vertical категории.

## Проверки

- Go: masked stochastic sampling, deterministic choice, stable extreme logits, legacy joint-likelihood comparison, rejected unavailable samples и stale/unknown inventory.
- GPU:3 tests passed. Доступный выбранный weapon logit получает gradient; masked logits и keep-only cases имеют нулевой weapon gradient; entropy/gradients конечны. Legacy anchor не навязывает новую weapon distribution.
- Existing recurrent3 и PPO14 tests прошли. `go test ./cmd/... ./internal/...` прошёл.
- `go test ./...` повторён с real-model/Python parity: product packages прошли; общий exit1 из-за ранее существовавших duplicate main в generated demo/aim helper directories. Лог: `workspace/artifacts/combat-weapon-head-v1-20261007/go-test-all.log`.

## Инициализация и fresh live replay

`initialize_combat_weapon_head.py` расширяет V6 actor last layer и Temporal residual (либо GRU output) на12 нулевых строк. Keep bias5 задаёт начальное предпочтение сохранять оружие при ненулевой exploration. Остальные8 logits и critic сохраняются. Это weights-only initialization, не миграция Adam и не обучение.

Артефакты: `workspace/artifacts/combat-weapon-head-v1-20261007/`. Source V6 SHA256 `6b0160609b30109de29f1264d95cd0fe3dbd67dfa6e9e5fc47e88af5a307479c`; initialized weights `e97ae237e211975c653566e3db29a1158335996b989398a961bd0ead73919f35`.

4 server/client instances x2, seeds44100–44103, Mixed/Machinegun fixture, release100/max300, reward recoil-v5. Все4 captures/native dispatch приняты. Go PPO exporter проверил377 first-life rows и4 terminals. Samples:371 keep,4 Blaster,2 Machinegun. CUDA replay RTX5070: legacy8 actor logits и critic до/после расширения отличаются на0; joint Go/Python logprob max error0.000046253, value0.000018224, context errors<0.000005. Optimizer steps0.

В этих4 эпизодах бот погиб. Выбор оружия ещё не обучен, seeds новые и не paired с Curriculum24, поэтому это не сравнение боевого качества. Native use requests действительно получены сервером, но first-life UDP observations оставались Machinegun: политика держала спуск. Штатный Weapon_Generic ждёт окончания firing перед сменой. Запрос и actual equip нельзя считать одним событием.

## Отдельная native switch проверка

Diagnostic probe сохраняет движение исходной сети, задаёт attack bias−10 при нулевых attack output weights/residual и Blaster bias10. Это явно изменённый небоевой probe, не кандидат обучения. 4 instances x2, distinct seeds44200–44203. Все4 captures приняты;685 PPO rows прошли Go replay/native proof.

`audit_combat_weapon_switch.py` связывает native `sv_test_client_cmd ... use Blaster` с alive first-life UDP-equipped weapon, проверяет map/spawncount. Во всех4 случаях request frame98, first equipped Blaster frame103, HP100, delay5 frames. Не включает respawn observations. Это доказывает прохождение request→native→UDP equip при release fire для MG→Blaster, не качество learned choice и не исполнение всех12 категорий.

## Следующее

Перенести optimizer state Curriculum24: first-layer input columns, actor output rows и Temporal residual rows с нулевыми новыми moments; сохранить Adam steps, updates24, total_actor_steps230, RNG и consumed rollouts. Согласовать V6 anchor/retention bank. Затем расширить fixture доступными Shotgun/другими оружиями и их ammo, разрешить соответствующие проверенные guards, обучать совместное release/fire/switch только CUDA и провести fresh paired evaluation. До этого weapon learning не принято; live/default не переключены.
