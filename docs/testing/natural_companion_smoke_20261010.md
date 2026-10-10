# Natural companion: первый двухклиентный прогон

Существующий `scripts/run_learned_campaign_comparison.ps1` расширен опцией
`WithCampaignTeammate`: второй native клиент идёт в campaign режиме с
обычными правилами, испытуемый клиент работает в companion режиме.
Learned клиент использует штатный `combat.live_companion`, без teleport,
выдачи оружия, invulnerability, scripted aim/fire или lockstep overrides.
Наблюдения игроков и native damage сохранены. Добавлена опция Stochastic;
прежний режим сравнения по умолчанию остаётся deterministic и solo.
Число worker slots допускает до16; каждый слот содержит один сервер
и двух клиентов. Источники не изменялись во время capture.

Artifact: `workspace/artifacts/natural-companion-smoke-v1-20261010`.
Четыре сцены: base1/base2 × rules/learned, по500 game frames,
timescale2, skill1, seeds182010/182110, ports29793..29796.
Модель — опубликованная attention64, не новый кандидат retention.

Все4 клиента испытуемой стороны завершились штатно, infrastructure_ok=true,
source_unchanged=true. Напарник наблюдался495..496 из501 уникальных кадров.
Friendly health damage в обе стороны0; уровни не завершены.
У learned стороны provider_frames=0 на обеих картах: это проверка подключения
и наблюдений, **не проверка работы learned боя** и не показатель качества.

Выявлена причина отсутствия боя: в Planner campaign активен только при
отсутствии напарника (`p.Campaign && s.Teammate == nil`). Ведущий campaign
бот при появлении companion переходит на cover_teammate; обе стороны
остаются рядом. У base2 ведущий стрелял, но не провёл companion к бою.

Следующий шаг: явная роль campaign leader. Она должна сохранять полные
наблюдения напарника и friendly-fire guards, но продолжать campaign route,
включая pickups, exit preparation, dependency/button handling и combat
spacing. Нельзя просто скрыть Teammate в snapshot или изменить только
основное условие Planner: дополнительные ограничения есть в campaign.go,
campaign_dependency.go, button.go, exit_preparation.go, resource_memory.go
и combat_intent.go. Обычный companion и прежнее campaign поведение должны
сохраниться по умолчанию. После изменения повторить двухклиентный smoke
и подтвердить ненулевые provider frames/native attacks до оценки качества.

Существующий сервер человеческой сессии PID1976 и бот PID3556 не заменены.
Все4 сервера и8 клиентов этого smoke закрыты собственной cleanup процедурой.

## Явный ведущий и повторный прогон

Добавлен `run.campaign_leader=true`, допустимый только в campaign режиме.
По умолчанию поведение прежнее. Общий campaignActive используется для
маршрута, подготовки ресурсов, кнопок, зависимостей и spacing. Наблюдения
напарника сохраняются; CampaignDecision явно отмечает роль leader для
combat intent. Ведущий включается только у второго клиента данного harness.

Промежуточный v2 smoke завершился штатно, но боя модели ещё не было:
другая строка Planner безусловно заменяла campaign goal позицией Teammate.
Исправлено условие замены; регрессия теперь проверяет и точку назначения,
и сохранность teammate observations, и владение задачей кнопки.
Campaign/button/friendly-fire/live-config проверки прошли с полным
существующим набором BSP/AAS (`leader-route-regressions.jsonl`).
Ранний расширенный запуск без нужных AAS и с несовместимым combat intent
сохранён как неуспешный; asset root исправлен, прежний intent с напарником
сохранён для неведущих ролей.

Итоговый artifact: `natural-companion-leader-smoke-v3-20261010`.
Четыре сцены,1200 frames, timescale2, skill1, seeds184010/184110,
ports29801..29804; source_unchanged=true, infrastructure_ok=true у всех.

| Карта | Companion | Выход на следующую карту | Смерти companion | Provider frames | Убийства companion |
|---|---|---|---:|---:|---:|
| base1 | rules | да | 1 | 0 | 1 |
| base1 | learned | да | 0 | 360 | 2 |
| base2 | rules | нет | 1 | 0 | 1 |
| base2 | learned | нет | 0 | 473 | 0 |

Friendly health damage0 в обеих base1 сценах и base2 rules. В base2 learned
native ledger зарегистрировал110 единиц урона companion напарнику:
100 от barrel explosion (mod26, frame96, lethal) и10 от Blaster
(mod1, frame136, shot25). Frame98 explosion повреждает уже мёртвое тело,
live_health_damage=0 и в110 не включён;
обратный friendly damage0. Поэтому cooperative safety acceptance не пройдена:
геометрическая MG проверка не покрывает обнаруженные проблемы barrel/Blaster.
Следующий приоритет — разобрать эти damage events по наблюдениям
и моментам запуска/попадания, прежде чем расширять оценку. Learned
нанёс40 единиц monster health damage на каждой карте по native ledger.
Это подтверждает включение модели и реальный бой с напарником, но один
seed на карту не доказывает превосходство. Убийство может быть добиванием
после ведущего; суммарный урон команды не приписывается companion.
Base2 не завершён в данном бюджете; требует отдельной диагностики маршрута,
смерти/исчезновения ведущего и восстановления сопровождения.
Новые веса не обучались и не публиковались; human session не заменена.
