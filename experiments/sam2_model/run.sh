#!/usr/bin/env bash
# Полный прогон e04 (SAM 2 + 3D-модель). Запуск из корня репозитория:
#   bash experiments/sam2_model/run.sh            -> версия v3 (основная)
#   FINAL=1 bash experiments/sam2_model/run.sh    -> + глубина по изменению размера на кадрах с одной камерой
# Данные: data/ (или PELENG_DATA=...), модели: models/yolo/yolo11n.onnx, models/sam2/sam2.1_hiera_small.{encoder,decoder}.onnx
set -e
S=$(cd "$(dirname "$0")" && pwd); W=${WORK:-out/e04}; mkdir -p "$W"; cd "$W"
export DET_DIR=${DET_DIR:-cache} A_MAX=500
[ -f "$DET_DIR/det_cam5.pkl" ] || python3 "$S/stage0_detect.py"                      # 0. YOLO + вычитание фона (подсказки)
python3 "$S/stage1.py"                                                                 # 1. грубая траектория -> рамки-подсказки
STRIDE=${STRIDE:-4} python3 "$S/stage2_sam.py"                                         # 2. маски SAM 2 (каждый 4-й кадр)
python3 "$S/fit_all.py" 3.1 fits_a.pkl 0:1001:${STRIDE:-4}                             # 3. подгонка 3D-модели по маскам всех камер
python3 "$S/gsmooth.py" fits_a.pkl T_a.npy                                             # 4. глобальное сглаживание
python3 "$S/stage2b_sam.py" T_a.npy 0:1001:${STRIDE:-4} masks2.pkl                     # 5. SAM 2 по проекции модели (недостающие камеры)
MASKS=masks.pkl,masks2.pkl INIT=T_a.npy python3 "$S/fit_all.py" 3.1 fits_b.pkl 0:1001:${STRIDE:-4}
MASKS=masks.pkl,masks2.pkl python3 "$S/gsmooth.py" fits_b.pkl T_v3.npy                 # 6. итог v3
OUT=T_v3.npy
if [ -n "$FINAL" ]; then
  MASKS=masks.pkl,masks2.pkl python3 "$S/monofit.py" fits_b.pkl T_v3.npy mono.pkl      # 7. одна камера: глубина по размеру
  MASKS=masks.pkl,masks2.pkl python3 "$S/gsmooth.py" fits_b.pkl T_final.npy mono.pkl
  OUT=T_final.npy
fi
python3 "$S/render_report.py" "$OUT" report video                                     # метрики + видеоотчёт (вид сверху)
python3 - "$OUT" <<'PY'
import sys, numpy as np
X = np.load(sys.argv[1]); open('trajectory.csv', 'w').write('Frame;Loc_X;Loc_Y;Loc_Z\n' + ''.join(f'{i};{x:.2f};{y:.2f};{z:.2f}\n' for i, (x, y, z) in enumerate(X[:, :3])))
PY
