"""Базовый интерфейс эксперимента.

Эксперимент получает сцену и обязан вернуть только 3D-траекторию.
Всё остальное (метрики, отрисовка, видео) делает общий валидатор —
так разные подходы сравниваются в одинаковых условиях.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.dataset import Scene
from core.validation import Prediction


class Experiment(ABC):
    name: str = "experiment"

    @abstractmethod
    def run(self, scene: Scene) -> Prediction:
        """Возвращает предсказание по сцене."""
        raise NotImplementedError
