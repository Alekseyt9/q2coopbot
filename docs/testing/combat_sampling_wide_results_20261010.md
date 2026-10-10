# Широкое сравнение режимов исполнения

Все560 боёв завершены без job errors,560 individual receipts проверены,
core quality report завершён.20 registered development семейств ×4
validation engine conditions24..27 ×7 вариантов. Веса двух моделей
одинаковы между режимами, меняется deterministic флаг и явный offset
policy RNG. Engine conditions/геометрия и монстры парные.

| Вариант | Победы /80 | Смерти |
| --- | ---: | ---: |
| m1 deterministic | 61 | 17 |
| m1 stochastic-a | 64 | 15 |
| m1 stochastic-b | 64 | 11 |
| parent3 deterministic | 61 | 14 |
| parent3 stochastic-a | 67 | 10 |
| parent3 stochastic-b | 60 | 13 |
| Правила | 68 | 8 |

m1 получает три дополнительные победы в обоих stochastic arms;
parent3-a шесть, parent3-b теряет одну относительно своего deterministic.
Нельзя считать лучший RNG arm доказательством устойчивого преимущества
sampling для каждого условия. Правила всё ещё выигрывают больше любого
кандидата; допуск к независимому test не выполнен. Test остаётся
зарезервированным, веса не продвигаются.

Результаты относятся к знакомой development выборке и этой версии runtime.
Повторные executions предыдущих cohorts дают небольшие различия даже
deterministic/rules исходов; это не доказательство полной детерминированности
нативного мира. Парные сравнения используют текущий cohort.

Quality SHA256:
`2a2d776450d66364e4178eb1122028310f9ed820ad6589eea9be41d6530f0c7f`.
Корень: `workspace/artifacts/action-sampling-offset-wide-v1-20261010`.
Все девять diagnostics и дополнительный executed-frame ownership/storage
разбор завершены. Diagnostics SHA256:
`4fc17b26f8f5e0d40dfde72159d350d87effee3f8a15b56ea847e6afde47fc19`.
Ownership SHA256:
`61c1e18fd542a9e1f74a6c2af9732c4576ce767beab9456c41e837fce9b157f2`.

Wide audit также нашёл Shotgun fallback: m1-a123 clear-target rules
кадра в четырёх победных случаях, m1-b72 в пяти, parent3-a60 в трёх,
parent3-b66 в четырёх. Это не причинное распределение побед между
правилами и сетью, но оно исключает заявление полностью самостоятельного
управления этими боями. Deterministic arms таких кадров не имеют.

После закрытия wide process patch допуска поддерживаемого подобранного
Shotgun/Machinegun применён; четыре targeted stub-provider checks passed
без skips, без нейросетевого offline Go/CPU исполнения. Новый56-case
validation завершён с прежними весами и условиями: m1 stochastic4/8 в
обоих arms вместо5/8, parent3-a5/8 и b4/8, deterministic обоих0/8,
правила8/8. Ни equip fallback, ни clear-target rules frames у learned
вариантов не зарегистрированы. Этот56 smoke не заменяет чистую широкую
оценку нового runtime. Подробности: combat_pickup_ownership_fix_20261010.md.

Физический audit1601 compressed streams:3255810668 логических байт,
1044443136 физических, все hashes сохранены. Все560 client paths имеют
один inode, все560 exporter paths тоже один. Это ограниченная accounting
выборка, не полный размер дерева.

После исправления сначала подтвердить чистое управление и эффект runtime
на живых боях, затем продолжать on-policy PPO из проверенных GPU checkpoints.
Увеличение сети без этой проверки не устраняет найденный gate между
инвентарём, выбором оружия и владельцем команды.
