# Пеленг — причинное сопровождение CAD по нескольким камерам

Программа оценивает положение и ориентацию заданной метрической CAD-модели. Основной поток выдаёт координаты кадра сразу после его обработки. Отдельная оранжевая траектория поддерживает настраиваемый фиксированный лаг.

## Структура

| Путь | Назначение |
| --- | --- |
| `online/perception/` | Сегментация, PicoPose, GoTrack, RoMa, TAPNext и рендеринг |
| `online/estimation/` | Геометрические измерения, состояние и причинные поправки |
| `online/evaluation/` | Оценка метрик, сравнение и графики |
| `online/pipeline.py` | Объединение измерений подключённых моделей |
| `online/tracker.py` | Управление наблюдениями камер |
| `online/run.py` | Покадровый запуск |
| `core/` | Камеры, системы координат и чтение данных |
| `web/` | Веб-плеер и настройки сглаживания |
| `tools/` | Запуск, загрузка ресурсов и генерация проверочных сцен |
| `tests/` | Проверки соглашений координат и совместимости |
| `experiments/` | Исторические экспериментальные реализации |
| `docs/` | Инструкции и устройство проекта |

Прежние команды `python -m online.run`, `online.evaluate`, `online.compare` продолжают работать. Прежние пути импорта перенаправляются на соответствующие пакеты.

## Запуск на Linux

```bash
python tools/run_online.py --runtime configs/linux-runtime.json   --data /path/to/data --out /path/to/results   --sam models/sam2/sam2.1_hiera_tiny
python -m online.evaluate --out /path/to/results --reference /path/to/reference.csv
```

Файл runtime задаёт каталоги дополнительных библиотек и кэша на Linux. Каталоги данных, весов и результатов хранятся отдельно от исходников.

Полная инструкция находится в [docs/online.md](docs/online.md). Новые оценщики описаны в [docs/modernization.md](docs/modernization.md), проверенные метрики — в [docs/results.md](docs/results.md). История исходного решения сохранена в [docs/legacy-overview.md](docs/legacy-overview.md).

Параметры опубликованного Way2 v6, CSV координат и ссылка на точный дополнительный checkpoint находятся в [submission/way2/README.md](submission/way2/README.md). Для Way2 приводятся метрики по независимым проверочным кадрам.
