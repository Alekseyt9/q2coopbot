# Temporal attention: Mixed demo, 2026-10-06

## Исправленный прогон

После просмотра обнаружен fixture bug: оба монстра в старой записи неподвижны. Их hull при `z=24` касается solid floor plane; native collision считает это startsolid. Fixed-release barrier переносит startup после первой секунды, и stock droptofloor пропускается. BSP diagnostic для Parasite: brush451 solid, при z24.125 пересечение исчезает; Gunner: brushes418/419 при z24.

В canonical `scripts/run_solo_tactical_retreat.ps1` стандартный Parasite и Mixed Gunner появляются при z24.125. Frozen comparison snapshot не менялся. Новый отдельный прогон `run-mobile` выполнен на timescale1 с тем же Temporal10 и seed28100. Оба монстра двигаются: наблюдались42 различные позиции Parasite и34 Gunner после release. Бот погиб на кадре351, убийств0, first-life health damage100. Предыдущие2 kills/27HP относятся к неподвижному fixture и не характеризуют полноценный бой.

В Yamagi теперь показывается `run-mobile/bot-smooth.dm2`. Оригинальная новая запись сохранена рядом. Нормализация повторов проверяет точное per-tick равенство world/player state. Playback: windowed1920×1080, timescale1, fixedtime0, timedemo0. Измеренный цикл предыдущей нормализованной записи:42.14s при391 игровых кадрах39.1s; сюда входят signon и загрузка. Ускорение записи x2 не требуется компенсировать x0.5 при просмотре: demo frames кодируют стандартный game tick100ms. Новый capture дополнительно выполнен на x1.

## Предыдущая запись и диагностика

Записан отдельный демонстрационный прогон Temporal attention после update10 опыта `combat-architecture-v2-20261006`. Seed28100, Mixed Parasite + Gunner, skill1, Blaster, synchronous harness, timescale2, release100, horizon300. Policy: `workspace/artifacts/temporal-mixed-demo-20261006/policy.json`.

Native first-life diagnostic: два убийства, 73 входящего health damage, бот жив с27 HP в конце. Legacy tactical harness вернул `accepted:false`: его проверка Shotgun/ranged-retreat не соответствует этому Blaster combat fixture. Демонстрация не добавлена к сравнительным evaluation results.

Оригинал: `workspace/artifacts/temporal-mixed-demo-20261006/run/bot.dm2`. В нём1095 frame messages для391 уникального world tick:704 повтора возникли во время synchronous hold. Yamagi `demomap` читает по одному demo chunk за tick, поэтому повторы растягивали время и вызывали остановки между движениями.

Для просмотра создан `run/bot-smooth.dm2`:704 повторных frames удалены, сопутствующие configstrings/messages сохранены. Standalone helper `normalize_demo.go` проверил через Go decoder точное равенство всех391 конечных per-tick player/entity states исходной записи и результата. Повторов0, пропусков0, корректный EOF. Оригинал сохранён. Это обработка данной записи; основной recorder ещё требует аналогичного исправления для synchronous captures.

Yamagi8.70 запущен с отдельным portable profile: windowed1920×1080, OpenGL3.2, timescale1, async render, vsync, лимит120FPS. Фактический client area1920×1080 подтверждён DPI-aware Win32 measurement и engine log; на снимке FPS counter59.95. Настройки и assets находятся в `workspace/artifacts/temporal-mixed-demo-20261006/playback`, пользовательский профиль не изменён. PAK подключены hard links.

Запись проигрывается по кругу; F5 запускает её заново. Для повторного запуска из каталога playback:

```powershell
.\quake2.exe -portable +set r_mode -1 +set r_customwidth 1920 +set r_customheight 1080 +set vid_fullscreen 0 +exec temporal-demo.cfg
```

Снимок и измерение: `playback-1920.png`, `playback-window.json` в корне артефактов демонстрации.
