# Fixed500 повтор на поздней когорте

23.10.2026. Фиксируется после baseline selection26состояний, до их альтернативных labels и до fitting. Три17-state прототипа закрыты сgate=false и не заменяются задним числом.

Child dataset =неизменные17parent contexts/splits +принятые поздние one-frame пары свежей когорты. Новыеlowerseed train/upperseed development были зафиксированы доальтернатив. Каждыйновыйконтекст:baseline+4кандидата, минимум3принятых варианта послеproof. Неуспешныеprefix/timing/контексты сохраняются какисключения, безскрытых повторов. Минимальныйдопуск —6новыхконтекстов на4новыхseed вкаждомsplit; минимум2независимыхTRAIN улучшения. Недостатокпокрытиянеобъявляетсяготовымfit.

Алгоритмы, layout191, teacher0,1source+0,9nativepreference, normalization/bounds/initializationseed и500steps каждогоизflat/calibrator/shared полностью как combat_native_shared_scorer_protocol_20261023.md. Начало source S4, никакогопродолжения17-state fits. FlatLR0,0003; calibrator0,05; sharednetwork0,003/beta0,05;clip0,5; прежниеweightdecay. Sourcebody/value/std/fire/predictor и исходныеAdam/RNG/files неизменны. ВсеNN толькоCUDA. Нетearlystop/выбораcheckpoint поdevelopment.

Для каждогоendpoint constant=mean25probabilities наTRAIN. Показать aggregate development и отдельноfresh/parent development, а такжеper-context. Основнойgate неизменен: usefulmass≥source+0,05 и≥своегоconstant+0,02;harmfulmass≤source+0,05. Дополнительныйfresh guard: usefulmass≥freshsource+0,02,≥freshconstant иharmfulmass≤freshsource+0,05. Этотguardфиксируется доfreshlabels, не заменяет основнойgate. Unknown directions остаются неизвестными.

Еслинесколькоgatesпройдены — большеaggregate usefulmass−constant, затемменьшеharmfulmass, затемимяварианта. Толькопослеruntime/densityинтеграции нужна свежаяnative source/trained/constant/rulesоценка. Prototype неготовыйPPOmodel;default автоматическинеизменять. Есливсенеудачны — неослаблятьthreshold, разбиратьnativeutilitysupport, single-spreadlabels иforecaster/geometry, анепросто повторятьfits.
