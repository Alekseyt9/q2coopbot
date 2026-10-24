# Более ранние альтернативы движения перед гранатой

## Итог на 22:12 МСК

Native/GPU сбор закрыт6/6:пятьdiagnosticвариантов+freshordinaryreference. Separate acceptance:5admitted,0excluded,zeroordinaryfull exact;prefix/state/features/common Gaussian+fire draws/неизменённые root categories и все3windowreceiptsпрошли. Report SHA `432b8063576eaabb4ef0ae0392053f89a8b313f20f2694960e3a1e23109ad724` в `workspace/artifacts/combat-grenade-early-window-acceptance-v1-20261024/report.json`.

Дополнительный `strict-prefix-proof.json` сравнил поведение без entity bijection:нормализуетсятолькоspawncount;версиямодели/diagnosticreceiptисключенытолькокакметаданные. Полное совпадениеzero/ordinary и prefix всех4сдвигов подтвердились при сохранении entity IDs.

| Вариант | Смерть network frame | Цель | Убийства | Нанесено HP |
|---|---:|---|---:|---:|
| Control | 137 | нет | 0 | 60 |
| Forward | 139 | нет | 0 | 70 |
| Backward | 138 | нет | 0 | 60 |
| Right | 137 | нет | 0 | 60 |
| Left | 137 | нет | 0 | 60 |
| Ordinary reference | 137 | нет | 0 | 60 |

Это не новые обученные модели и не6независимыхоценочныхбоёв. Forward/backward лишь немного продлили жизнь; все проиграли. Нельзя превращать лучший discounted return этой группы в метку «выигрышный манёвр» или общее преимущество. Ни fitting,ниpromotionнебыло.

### Исправление отдельного метадатного аудитора

Первый acceptance остановился до output creation на `KeyError aim_candidate`:старыйвыборконтекстовгарантировалактивнуюпоправкуприцела,а на132текущегоконтекстаеёнет. Теперь timing требует одинаковой наличности head и одинакового timing при его наличии; movement source берётся из исходного Gaussian latent при неактивном head. Исходный Gaussian/sample receipt не меняется. Два focused metadata tests прошли, старый источник и failure JSON сохранены в preflight. Ни одной игры не повторяли, capture/результаты не меняли, условия admission не ослаблялись.

### Новая конкретная гипотеза: выбор оружия

Исходный контекст — blaster-условие. В наблюдаемом инвентаре на100..132естьShotgunи20Shells,inventory_age=0. Маска разрешаетkeep/Blaster/Shotgun,однаковыйбластерприэтомсохраняетсявовсехпроверенныхкадрах. Отдельный CUDA replay текущего actor со всеми weapon residual/exploration transforms показал keep≈99.79..99.83%,Shotgun≈0.155..0.185%. Это не недоступность оружия и не stale inventory; наблюдаемая policy почти не исследует переключение.

Отчёт `workspace/artifacts/combat-grenade-city1-weapon-distribution-v1-20261024/report.json`,SHA `41130c4c53a51d5a8cb543a48dcdfe9fea9a408290eb3aae2a1085e680b28699`. Скрипт `audit_combat_context_weapons_cuda.py`,neuralexecutionтолькоCUDA,nofit/newgames. Общий`ppo_training_ready=false`этойсложнойpolicyнеявляетсяотказомданныхдляанализа:источникCUDA-verified,неdiagnostic,modelSHAсверен;genericPPOдопускэтимнеизменён.

Текущий`weapon_exploration`T2вкодеприменяетсятолькопринаблюдаемомэкипированномGrenades(index11),неприBlaster. Следующий отдельный опыт должен сохранять этот прежний grenades режим и менять только исследование приBlasterсдоступнойальтернативой. Нужны zero-controlCUDA/маски/неизменные другие outputs/неизменные optimizer tensors и новый paired native screen до fitting/promotion. Пока не доказано, что переключение наShotgunспасётэтотбой;такаяпроверкаещёнезапускалась.

## Постановка до прогона

24.10.2026. Предыдущая проба отменённого запроса city1/frame134 подтвердила исполнимость движения, но не улучшила выживание:control/request погибли на137. Следующий опыт заранее переносит вмешательство на132..134,а не повторяет поздний override.

## Совместимость и ограничения

`validateDiagnosticMovementProbe` в Go и Python разрешает известные equipped-prefix семействаv10/v11/body-ablated/v12typed/neutral/ablated. Пределы mean_shift, frame/duration, stochastic joint head и запрет несовместимых diagnostic/geometry/execution режимов сохранены. Неизвестные и прежние несовместимые схемы отвергаются. Нормальная policy и команды без metadata не меняются. Continuation RNG fork здесь не используется и его admission пока остаётся прежним.

`go test ./internal/policy -run '^TestDiagnosticMovement' -count=1` прошёл:metadata/scalar/window/draw tests, включая новые feature families и negative admissions. Go NN forward не выполнялся. Основные веса не менялись.

Существующий `prepare_combat_movement_window_cuda.py` обработал настоящий source endpoint и его собственный TRAIN корпус на GPU. Actor/joint neural parameters exact,zero control exact,обычная scoring/training admission отвергает diagnostic header. Source LP max error0.00003624в пределах прежнего gate0.003;analytic Gaussian shift и window masks проверены на CUDA. Это техническая совместимость, не качество боя.

## Фиксированный опыт

Source neutral round02 weights `5dcf0d640d28394817bc105a58b379e4e6723eee747867d16a0cfbd87535ea0c`. City1 seed1468045009,тотжеfixture,actorизсобственногоround03TRAIN. Root132имеетhealth89,grounded,прошлое наблюдаемое падениеHPна2. Состояние намеренно выбрано из уже разобранной TRAIN неудачи; это не observation-only случайная выборка и не независимая оценка.

Окно132..134(3networkframes),пять mean_shift вариантов:control[0,0],forward[1,0],backward[-1,0],right[0,1],left[0,-1]. Шестой вариант — fresh ordinary reference. Параметры сети/std/обычный порядок случайных draws сохраняются; safety guard остаётся прежним. Это вмешательство в movement distribution: связанный с движением прицел/fire могут изменяться по текущей policy, а не чистый override только integer Forward/Side.

Бюджет6native/6CUDA,существующий harness/общаяочередь≤16,×2. Сиды повторяются намеренно ради одного и того же мира;новое обучение отсутствует. До интервенции должны совпасть prefix/state/features,а Gaussian/fire draws должны совпасть в окне. Zero control должен полностью совпасть с ordinary reference. Сначала technical admission,после него — описательные native последствия. Один контекст не доказывает превосходство над rules,не становится обычным TRAIN и не меняет default.

Протокол `docs/combat_grenade_early_movement_window_protocol_20261024.json`,SHA `34194f34f3c11fb06d1f8c0e46d50a4cc1e72fd69dd93366928e13426381733c`. Context adapter и GPU preparation: `workspace/artifacts/combat-grenade-early-window-preflight-v1-20261024`. Native output: `workspace/artifacts/combat-grenade-early-window-native-v1-20261024`. Runner `scripts/run_combat_movement_probe.py --fresh-reference`,затемотдельный native common-state/draw acceptance audit. На момент записи запущен native runner,результаты ещё не приняты.
