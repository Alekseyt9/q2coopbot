# CUDA сравнение shared scorers: fixed500

23.10.2026. Фиксируется до fitting. Источник S4 и sealed datasetv217states:12train/5development, split и native labels не меняются. Flat200 результат сохранён отрицательным. На этой фазе нет новых игр и нет runtime совместимых новых PPO файлов.

## Общие данные и цель

НаCUDA один раз получить source1359inputs, logits и25frozen hints. Исходные Gaussian/discrete actions из native context; не генерировать новые actor actions. Native utilities/known masks служат targets/метрикам, не входам. Teacher тот же:0,1source probabilities+0,9preference; baseline если не хуже всех измеренных альтернатив, иначе равномерно по лучшим. Равный вес состояния; development не участвует в gradient или checkpoint selection.

Для всех вариантов ровно500train updates, clip grad norm0,5, без early stopping. Оценить fixed endpoint. Source actor/value/std/fire/predictor и исходные файлы/чекпоинты остаются неизменными. Все NN constructors/forward/fits/checks CUDA; GoNN offline не используется. Seed инициализации20261023 под fork_rng.

## Три варианта

**Flat500:** source1359→64ReLU→25, все88665параметров head; Adam LR0,0003,weight decay0,0001. Начало source S4, без продолжения200-step fit.

**Calibrator:** два параметра alpha,beta. `logit_k=alpha*(sigmoid(hint_k)-sigmoid(hint_0))+beta*is_baseline`. Alpha init0,project[0,20]; beta initlog(24P0/(1−P0)),P0среднее source поtrain,project[−20,20]. Adam LR0,05,безweight decay. Это контекст через frozen hint; не greedy runtime ranker.

**Shared191:** один MLP191→16ReLU→1 для всех25направлений и отдельныйbeta baseline.191inputs =185candidate causal inputs, нормированные source predictor mean/std[:185] иclamp[−20,20],2offsets/5,3probabilities(candidate,baseline,candidate−baseline),1baseline flag. Candidate causal185 отличается финальным aim через прежний source-latent transform/scale; все остальные action/context поля прежние. Network outputclamp[−20,20], затемbeta*baseline. Last layer initzero, beta какcalibrator,network first layer fixedseed. Adam networkLR0,003,weight decay0,0001; betaLR0,05,weight decay0; beta project[−20,20]. Около3089параметров (точныйcountизмодуля). Начальная distribution constantnearidentity с source train meanP0; это не точная nonuniform source head.

Такие bound сохраняют logit span≤60уshared/calibrator и положительную поддержку вfloat32. Это архитектура/параметризация, не action masks. Разные learning rates зафиксированы до fit; опыт сравнивает практические варианты, а не изолирует единственную причину изменения.

## Оценка и решение

Для каждого fixed endpoint constantcontrol =mean25probabilities наtrain, без использованияdevelopment. Показать per-context и aggregate useful/harmful known mass, coverage unknown действий, conditional known HP. Непроверенные действия не считать промахами/попаданиями. Gate прежний: development usefulmass≥source+0,05 и≥своегоconstant+0,02; harmfulmass≤source+0,05.

Development уже использован в итерации гипотез; это не независимая оценка победности. Если gates прошли несколько вариантов, порядок для последующей runtime реализации: больше usefulmass−constant, затем меньше harmfulmass, затем имяvariant. Только после runtime интеграции и density проверки — свежий native source/trained/constant/rules screen. Если ни один не прошёл, native quality прогоны не запускать ради нужного результата.

Prototype outputs имеют отдельный kind и native_runtime_compatible=false; это не готовые игровые weights. Flat reference также сохраняется как prototype head, чтобы не путать fitting и проверку игры. Старые default/models не изменяются.
