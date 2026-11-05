# Way2 v6 refined

The release uses causal processing with k=0 and emits one position per frame.
The Way1 page uses its separately validated production result.

The coordinate export is `Way2-coordinates.csv`. Frame numbering starts at zero.
`Loc_X`, `Loc_Y`, and `Loc_Z` use the original absolute coordinate system in centimetres.
`trajectory.csv` on the demo additionally contains the estimated speed.

## Checkpoint

The auxiliary YOLO26l segmentation checkpoint was fine-tuned on Way1.
The complete Way2 evaluation uses that checkpoint and the pretrained primary YOLO26l model.
SAM2 and GoTrack use their published weights.

Download the auxiliary checkpoint into the local models directory.

```bash
mkdir -p models/way2
curl --fail --location https://plane.tail6ffd57.ts.net:8443/way2/precision-yolo26l-seg.pt -o models/way2/precision-yolo26l-seg.pt
printf '%s  %s\n' 'c7c242a00276778bcab6fb6fd4bedaf5172c55acc4c44245b0d7aa14a14a3c66' 'models/way2/precision-yolo26l-seg.pt' | sha256sum --check -
```

The exact checksum is also recorded in `release.json`.

## Run

Install the dependencies and upstream GoTrack source described in `docs/online.md`.
Place the pretrained primary YOLO26l checkpoint and SAM2 ONNX encoder and decoder under `models`.
The input directory contains the Way2 videos, camera CSV exports and sensor dimensions.

```bash
python -m online.run \
  --data /path/to/way2 --out /path/to/result \
  --global-seg models/cab-ray/yolo26l-seg.pt \
  --sam-video facebook/sam2.1-hiera-base-plus \
  --sam-video-fallback facebook/sam2.1-hiera-large \
  --cad-points --cad-regions \
  --appearance-target assets/cargotruck/truck-appearance.json \
  --gotrack-source vendor/gotrack \
  --gotrack-checkpoint vendor/gotrack/gotrack_checkpoint.pt \
  --fill-gaps \
  --precision-recovery-model models/way2/precision-yolo26l-seg.pt \
  --precision-recovery-sam models/sam2/sam2.1_hiera_tiny \
  --verify-weak-tracks --video
```

The supplied CAD describes the demonstration target. The target interface supports a metric
CAD and a designated reference part. Training and transfer to different object appearances
require separate validation.

## Evaluation

Way2 has no supplied complete ground-truth trajectory. The release report records deviations
from 217 independent image-based reference positions and reports the comparison on matched
frames separately. These sample metrics apply to the specified frame sets.
The full replay reports coverage and adjacent position changes on all 1933 frames.

The fine-tuning recipe used six epochs at image size 768, batch size 4, the first ten layers
frozen, AdamW, learning rate 0.00015, one warmup epoch with zero initial bias learning rate,
FP32, seed 0 and internal validation separated by temporal blocks. The labels were generated
with `tools/detection_labels.py` and augmented with smoke and contrast variations.
