# 3dtracking-absolute — Хакатон Пеленг 2.0

3D-координаты движущегося объекта (грузовика) по видео с нескольких калиброванных камер и восстановление его траектории.

**Решение — [`experiments/sam2_model`](experiments/sam2_model/README.md)**:
- маски SAM 2.1 в каждой камере;
- подгонка 3D-модели грузовика `CargoTruck.fbx` по всем камерам сразу (две части — кабина и кузов, устойчиво к перекрытию деревьями и рельефом);
- глобальное сглаживание траектории.

Выход — координаты кабины на каждом кадре.

| Way1 (1001 кадр, 5 камер) | RMSE | MAE | медиана | покрытие |
|---|---|---|---|---|
| e04 v3 | 4.14 м | 2.51 м | 1.61 м | 100% |
| кадры 300–700 (2–4 камеры) | — | 0.98 м | 0.90 м | 100% |

## Запуск
```
pip install -r requirements.txt
bash tools/get_yolo.sh
mkdir -p models/sam2 && cd models/sam2 && curl -LO https://huggingface.co/vietanhdev/segment-anything-2.1-onnx-models/resolve/main/sam2.1_hiera_small_20260221.zip && unzip sam2.1_hiera_small_20260221.zip && cd ../..
bash experiments/sam2_model/run.sh
```
- Данные кладутся в `data/`: `object_transform_*.csv` и `*CineCameraActor{1..5}.mp4`. Другую папку можно задать через `PELENG_DATA`.
- Результат: `out/e04/trajectory.csv` (`Frame;Loc_X;Loc_Y;Loc_Z`, см).
- Если рядом с данными лежит файл истины, в `out/e04/report/` появятся метрики и видеоотчёт.

## Структура
| папка | назначение |
|---|---|
| `experiments/sam2_model/` | решение (SAM 2 + 3D-модель + сглаживание) |
| `core/` | данные, камеры, метрики, видеоотчёт |
| `experiments/bgsub_ekf/`, `experiments/yolo_ekf/`, `experiments/e03_yolo_ekf.py` | YOLO11n + вычитание фона + 3D-трекер: грубая траектория, по которой SAM 2 получает подсказки, где искать грузовик |
| `tools/get_yolo.sh` | загрузка модели YOLO11n |

## Соглашения
- Единицы — сантиметры (Unreal Engine). Система координат левая: X вперёд, Y вправо, Z вверх.
- fx = W·f/23.76, fy = H·f/13.365, главная точка в центре кадра, без дисторсии.
- Положение и ориентация камер берутся из CSV на каждый кадр, поэтому поворотные камеры поддерживаются.
