# Восстановимые кэши завершённых оценок

22.10.2026 прирост свободного места около3.08GiB тремя операциями:0.51,1.45 и1.12GiB. Удалены76 **производных объединённых** JSONL-кэшей закрытых evaluation/screen,около7.03GiB логического размера. Исходные per-case CUDA streams, native captures, модели/checkpoints и отчёты сохранены. Текущие TRAIN/development corpora не затрагивались. Измеряется разница свободного места до/после; во время последней операции шла другая запись, поэтому возможен небольшой дрейф.

Перед удалением для каждого cache проверено: его bytes точно равны конкатенации сохранённых per-case files в записанном порядке, а SHA256/размер совпадают с исходным merged report. Повторно проверены hashes всех частей непосредственно перед удалением. Это устраняет дубликаты данных, а не первичные записи. После удаления одного cache выполнен настоящий restore roundtrip с точным SHA256, затем этот дубликат снова убран.

Manifests/receipts:

- `workspace/artifacts/combat-cold-merged-cache-trim-v1-20261022/manifest.json`, `receipt.json`, `restore-test.json`.
- `workspace/artifacts/combat-cold-merged-cache-trim-v2-20261022/manifest.json`, `receipt.json`.
- `workspace/artifacts/combat-cold-merged-cache-trim-v3-20261022/manifest.json`, `receipt.json`: ещё24файла/2.57GiB logical; закрытые joint-motor screen, SAC behavior-KL/feedback и sequence-hitscan evaluations.

Некоторые старые отчёты по-прежнему ссылаются на оригинальные cache paths. Для повторного NN replay/проверки, которой нужны merged files, сначала восстановить нужный manifest:

```powershell
F:/src/strat/.venv-gpu/Scripts/python.exe scripts/restore_combat_merged_cache.py --manifest workspace/artifacts/combat-cold-merged-cache-trim-v1-20261022/manifest.json
```

Для второй/третьей группы заменить v1 наv2/v3 в пути manifest. Можно создать отдельный manifest с нужным подмножеством `entries`, чтобы восстановить один cache. Существующие файлы не перезаписываются: их SHA должен совпадать. При несовпадении частей плохой результат не публикуется; остаётся явно названный restore-partial для диагностики.

Восстановление возвращает bytes и paths, но потребляет место. Manifest описывает первоначальный локальный workspace; после переноса каталога потребуется корректировка путей. Удаление per-case parts после этой процедуры лишит гарантии точного восстановления, поэтому их нельзя считать лишними дубликатами.
