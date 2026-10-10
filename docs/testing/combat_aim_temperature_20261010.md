# Categorical aim confidence: temperature pilot

Continuous noise comparison560 fully sealed: best quarter-noise mean67/80
versus rules68/80; independent-test gate negative. Half noise lost three wins
in each RNG arm. Quarter gained/lost3/3 inarmA and5/4 inarmB. It does not
establish robust improvement. Reference ownership clean,9 diagnostic/source/
storage receipts accepted; no test dispatch or promotion.

## Transformation

`prepare_combat_aim_temperature.py` prepares three inference profiles from
one stochastic parent: baseline, mode4 (coarse/fine logits×4), targetmode4
(target and coarse/fine logits×4). This is softmax temperature0.25 for the
selected categorical heads. Choices remain stochastic; logits and means are
computed by the same learned network. No nearest-target or hand-coded aim
override, new runtime feature, Go source change or native oracle is introduced.

Every additive logit contribution is scaled: actor final rows20..28 and/or
63..80, GRU actor output or temporal-attention actor residual rows when
present, spatial-aim final rows2/3. Spatial fine/coarse correction rows,
movement/aim means, fire/vertical/weapon logits, std and critic stay unchanged.
Identical inputs retain argmax rankings; altered sampling can change later
observations and outputs. The procedure is inference calibration, not a PPO
update or compatible checkpoint/Adam continuation.

CUDA32-real-feature-row audit on current Attention64 parent passed all three
profiles: selected logit scaling maximum error0.0, all other raw outputs and
critic exact. RTX5070, no CPU/Go numerical NN replay. Artifacts:
`aim-temperature-preparation-v1-20261010/cuda-audit.json`.

Additional real CUDA audit passed for MLP64(m0),Attention128(m2),GRU128(m3),
using the sealed machinegun-shared after weights. Existing deterministic
metadata explicitly copied as stochastic before calibration, all other fields
preserved and original weights SHA bound. These three architecture types and
current Attention64 demonstrate transformation compatibility on32 rows, not
all-seed native quality or training acceptance. Root:
`aim-temperature-architecture-audit-v1-20261010/report.json`.

## Native pilot112

`run_combat_aim_temperature_evaluation.py`, root
`aim-temperature-pilot-v1-20261010`, actual Python PID6872 confirmed running.
Three profiles ×two RNG arms ×16 common development conditions plus rules16:
112 battles,16 refill slots,×2. Families: parasite+gunner Blaster/Machinegun,
campaign base2 site02 Blaster/Machinegun, validation24..27. No test split.

Driver requires sealed previous560 quality/9 diagnostics/ownership/storage,
current source bindings, accepted CUDA audit,2GiB headroom. All7 plans checked
by actual registry planner, generated fixtures and engine seeds identical;
policy offsets20261011/20261012. SHA-bound terminal captures transparently
LZX-compressed. Current pilot110/112 captures completed without errors at
last verified snapshot; quality pending.

After capture: member proofs, actual sampling config audit, quality,
9 diagnostics,control ownership,outcome/storage. Paired baseline→mode4,
baseline→targetmode4,mode4→targetmode4 reporter holds actual process handle6872.
No fresh server job is dispatched by reporter. Pilot quality cannot establish
full20-family or independent-test superiority; next wide evaluation depends
on sealed pilot and available storage. Test remains reserved, no promotion.

## Core pilot112

Все112 native captures завершены без ошибок,core quality SHA256
`6cce455f78e976c33049d45d530a79148bff1293b35365ff32c475485b5d8256`.

| Profile | Wins A/B /16 | Deaths A/B | Mean wins /16 |
|---|---:|---:|---:|
|baseline|9 /6|6 /9|7.5|
|mode4|8 /6|7 /8|7.0|
|targetmode4|7 /6|8 /9|6.5|
|rules|8|7|8.0|

На этих четырёх development families увеличение categorical уверенности
не добавило побед. Pilot не оправдывает расширение такого режима на остальные
архитектуры или независимый test. Closure diagnostics/ownership/storage ещё
обрабатывается actual process6872. Преобразование корректно на GPU, но это
отдельно от поведения: learned logits не становятся правильнее от scaling.
До заключения о дальнейших train изменениях дождаться paired transitions/
aim/outcome результатов. Не считать базовые9/16 доказательством победы над
rules8/16: вторая baseline ветка6/16.

Pilot112 полностью закрыт:9 diagnostics,clean ownership,distinct actual
sampling configs,outcome/storage SHA gates проверены повторно; process6872
авторитетно завершился. Paired baseline→mode4:0/1 gained/lost wins вarmA,
1/1 вarmB; baseline→targetmode4:0/2 и1/1. Mode switch fraction по declared
provider frames baseline319/1243 и236/862,mode4243/1257 и214/1093.
Уменьшение переключений не повысило победы. Это отрицательная проверка этой
конкретной inference calibration; не вывод о ненужности target/mode heads.
Wide temperature evaluation не запускается.

Следующий шаг — [два последовательных CUDA PPO раунда](combat_ppo_series_20261010.md)
исходного parent с новым own-policy опытом, без изменения награды/голов.
