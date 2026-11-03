import sys, numpy as np, json
import os; ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, ROOT)
from core.dataset import load_scene
from core.validation import Validator, Prediction
X = np.load(sys.argv[1]); OUTD = sys.argv[2]; video = len(sys.argv) > 3 and sys.argv[3] == 'video'
scene = load_scene(os.environ.get('PELENG_DATA', os.path.join(ROOT, 'data'))); N = len(X)
pred = Prediction(name='e04_sam2_model', xyz={t: X[t, :3] for t in range(N)}, n_cams={t: 0 for t in range(N)},
                  meta={'detector': 'SAM 2.1 (маски: кабина+кузов) + подгонка 3D-модели CargoTruck + глобальное сглаживание'})
try: V = Validator(scene, experiment_name='e04_sam2_model', out_dir=OUTD, top_view=True)   # вид сверху (ветка ekf-multicam)
except TypeError: V = Validator(scene, experiment_name='e04_sam2_model', out_dir=OUTD)
s = V.run(pred, render_video=video)
print(json.dumps(s, ensure_ascii=False, default=str)[:800])
