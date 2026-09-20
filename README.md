# Q2 CoopBot

Экспериментальная версия бота для кооперативного прохождения оригинального
Quake II под [Yamagi Quake II](https://github.com/yquake2/yquake2).

Проект основан на исходниках [Q2 Gladiator Bot Botlib Reconstruction](https://github.com/themuffinator/Q2-Gladiator-Bot).
Сохраняются исходный интерфейс botlib v0.96, формат AAS и большая часть
оригинального game-модуля Gladiator. В этот репозиторий добавлены изменения для
первого coop-MVP.

## Что добавлено

- серверная команда **sv coopbot <name> <skin> <charfile> <charname>**;
- передача значения **coop** в botlib;
- поиск монстров за пределами клиентских слотов;
- игроки не рассматриваются как враги в coop;
- базовая совместимость с обычными Quake II coop-картами;
- сборка под Windows x86 и x64.

Это не законченный автономный напарник для всей кампании. Бот умеет
ориентироваться по AAS, искать и атаковать монстров, но пока не понимает
полноценно цели уровня: кнопки, двери, лифты, сюжетные триггеры и переходы
между картами требуют дальнейшей разработки.

## Требования

- оригинальные файлы Quake II из легальной установки;
- Yamagi Quake II для Windows;
- CMake и Ninja;
- MinGW-w64 или другой совместимый C-компилятор.

Готовый официальный Windows-пакет Yamagi обычно запускается как **i386**, поэтому
для него используется x86-сборка CoopBot. x64 DLL также собираются этим
проектом и предназначены для x64-сборки движка.

## Сборка

Примеры для MinGW-w64 из PowerShell:

~~~powershell
cmake -S . -B build-x86 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++

cmake --build build-x86 --target gladiator game --parallel 4
~~~

Для x64 нужно выполнить те же команды с x64-компилятором и каталогом
build-x64.

Результаты:

~~~text
build-x86/src/game/gamex86.dll
build-x86/libgladiator.dll

build-x64/src/game/gamex86_64.dll
build-x64/libgladiator_x64.dll
~~~

## Установка в Yamagi

Создайте каталог мода:

~~~text
<Quake II>\YamagiQ2\coopbot\
~~~

Для текущего 32-битного Windows Yamagi скопируйте:

~~~text
gamex86.dll       -> coopbot\game.dll
libgladiator.dll  -> coopbot\gladiator.dll
~~~

Также в каталоге мода должны находиться данные Gladiator:

~~~text
coopbot\pak7.pak
coopbot\bots.cfg
coopbot\Gladiator.gsl
coopbot\default\
~~~

Эти данные берутся из оригинального дистрибутива Gladiator или из сохранённой
копии в archive/gladiator-bot/. Файлы Quake II pak0.pak, pak1.pak и pak2.pak
в репозиторий не входят.

Если MinGW собрал DLL с динамической зависимостью на libgcc, положите
libgcc_s_dw2-1.dll рядом с q2ded.exe или quake2.exe.

## Запуск coop с ботом

Запустите Yamagi с модом:

~~~text
quake2.exe -portable +set game coopbot
~~~

В консоли игры выполните:

~~~text
map base1
exec coopbot.cfg
~~~

Файл coopbot.cfg содержит:

~~~text
sv coopbot "RangerBot" "male/grunt" "bots/player_c.c" "player"
~~~

Для ручного добавления другого бота используйте ту же команду с другим именем,
skin и character-файлом. На сервере можно проверить результат командой status.

## AAS для новых карт

Gladiator требует навигационный файл .aas. Для карты, которой нет в готовых
данных, извлеките её BSP и запустите BSPC:

~~~text
bspc.exe -bsp2aas path\to\map.bsp -output path\to\coopbot
~~~

Результат должен находиться здесь:

~~~text
coopbot\maps\map.aas
~~~

Для base1 в рабочей установке уже подготовлен coopbot\maps\base1.aas.

## Происхождение и благодарности

Основная база проекта — [Q2-Gladiator-Bot](https://github.com/themuffinator/Q2-Gladiator-Bot),
репозиторий реконструкции Gladiator botlib. Исходники game-модуля происходят
из опубликованного исходного кода Gladiator Bot; реконструированный botlib
воспроизводит интерфейс и поведение оригинальной библиотеки насколько это
возможно по доступным материалам.

Отдельная благодарность Mr Elusive / Jan Paul van Waveren за оригинальный
Gladiator Bot и архитектуру AAS.

Q2 CoopBot не связан с id Software или Yamagi Software.

## Статус

Проект экспериментальный. Рабочая проверка выполнена на Yamagi Q2 8.70a,
оригинальной карте base1 и 32-битной Windows-сборке. Изменения наследуют
ограничения и предупреждения исходного проекта Gladiator.
