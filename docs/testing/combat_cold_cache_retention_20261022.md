# Восстановимые кэши завершённых оценок

22.10.2026 прирост свободного места около12.74GiB пятью операциями:0.51,1.45,1.12,4.84 и4.82GiB. Удалены100 **производных объединённых** JSONL-кэшей закрытых evaluation/screen/quality,около29.21GiB логического размера. Исходные per-case CUDA streams, native captures, модели/checkpoints и отчёты сохранены. Текущие TRAIN/development corpora не затрагивались. Измеряется разница свободного места до/после; во время некоторых операций шла другая запись, поэтому возможен небольшой дрейф.

Перед удалением для каждого cache проверено: его bytes точно равны конкатенации сохранённых per-case files в записанном порядке, а SHA256/размер совпадают с исходным merged report. Повторно проверены hashes всех частей непосредственно перед удалением. Это устраняет дубликаты данных, а не первичные записи. После удаления одного cache выполнен настоящий restore roundtrip с точным SHA256, затем этот дубликат снова убран.

Manifests/receipts:

- `workspace/artifacts/combat-cold-merged-cache-trim-v1-20261022/manifest.json`, `receipt.json`, `restore-test.json`.
- `workspace/artifacts/combat-cold-merged-cache-trim-v2-20261022/manifest.json`, `receipt.json`.
- `workspace/artifacts/combat-cold-merged-cache-trim-v3-20261022/manifest.json`, `receipt.json`: ещё24файла/2.57GiB logical; закрытые joint-motor screen, SAC behavior-KL/feedback и sequence-hitscan evaluations.
- `workspace/artifacts/combat-cold-merged-cache-trim-v4-20261022/manifest.json`, `receipt.json`: ещё12файлов/11.11GiB logical; закрытый `combat-conditional-weapon-quality-recovered-v1-20261012` с112условиями на вариант. Полная конкатенация всех сохранённых per-case частей совпала с каждым merged файлом; hashes повторно проверены перед native `Remove-Item -LiteralPath`, без рекурсивного удаления. Прирост свободного места4.84GiB, после операции≈25.37GiB.
- `workspace/artifacts/combat-cold-merged-cache-trim-v5-20261022/manifest.json`, `receipt.json`: ещё12файлов/11.07GiB logical; закрытые `combat-grenades-recovery-confirmation-quality-v1-20261011` и `combat-weapon-state-quality-analysis-v1-20261011`. Exact concat/SHA/parts/closed seals перепроверены,границы каждого пути проверены,удаление только native `Remove-Item -LiteralPath`. Прирост4.82GiB,послеоперации24.35GiB.

Некоторые старые отчёты по-прежнему ссылаются на оригинальные cache paths. Для повторного NN replay/проверки, которой нужны merged files, сначала восстановить нужный manifest:

```powershell
F:/src/strat/.venv-gpu/Scripts/python.exe scripts/restore_combat_merged_cache.py --manifest workspace/artifacts/combat-cold-merged-cache-trim-v1-20261022/manifest.json
```

Для остальных групп заменить v1 наv2/v3/v4/v5 в пути manifest. Можно создать отдельный manifest с нужным подмножеством `entries`, чтобы восстановить один cache. Существующие файлы не перезаписываются: их SHA должен совпадать. При несовпадении частей плохой результат не публикуется; остаётся явно названный restore-partial для диагностики.

Восстановление возвращает bytes и paths, но потребляет место. Manifest описывает первоначальный локальный workspace; после переноса каталога потребуется корректировка путей. Удаление per-case parts после этой процедуры лишит гарантии точного восстановления, поэтому их нельзя считать лишними дубликатами.

Позднее выполнена отдельная [очистка100GiB](disk_cleanup_100g_20261022.md) с истечением хранения старых raw traces/datasets. Все3440per-case parts,перечисленные вэтих пятиtrim manifests,исключены из неё иих наличие проверено. Гарантия восстановления именно merged caches сохраняется;прочиестарыеnative logs могли быть удалены,что отмечено вretention records.
