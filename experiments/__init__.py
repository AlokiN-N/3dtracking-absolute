"""Эксперименты-варианты поверх общего слоя `core`."""

from experiments.e00_oracle import OracleTriangulation

EXPERIMENTS = {
    OracleTriangulation.name: OracleTriangulation,
}
