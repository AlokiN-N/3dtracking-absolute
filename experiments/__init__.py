"""Эксперименты-варианты поверх общего слоя `core`."""

from experiments.e00_oracle import OracleTriangulation
from experiments.e01_bgsub_ekf import BgSubEKF
from experiments.e02_mast_ekf import MaSTEKF
from experiments.e03_yolo_ekf import YoloEKF

EXPERIMENTS = {
    OracleTriangulation.name: OracleTriangulation,
    BgSubEKF.name: BgSubEKF,
    MaSTEKF.name: MaSTEKF,
    YoloEKF.name: YoloEKF,
}
