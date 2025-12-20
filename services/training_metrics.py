"""
Training Metrics Parser and Collector

Parses ai-toolkit output lines to extract training metrics (loss, lr, speed, etc.)
and collects them for report generation.
"""
import re
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass
class TrainingMetric:
    """Single metric point from training output"""
    timestamp: float
    step: int
    total_steps: int
    progress_pct: float
    loss: float
    learning_rate: float
    speed: float  # it/s


@dataclass
class MetricsCollector:
    """Collects metrics during training"""
    task_name: str
    model_type: str
    config: dict
    metrics: List[TrainingMetric] = field(default_factory=list)
    start_time: float = 0

    def add_metric(self, metric: TrainingMetric):
        """Add a metric point to the collection"""
        self.metrics.append(metric)

    def get_loss_series(self) -> tuple:
        """Returns (steps, losses) for plotting"""
        steps = [m.step for m in self.metrics]
        losses = [m.loss for m in self.metrics]
        return steps, losses

    def get_lr_series(self) -> tuple:
        """Returns (steps, learning_rates) for plotting"""
        steps = [m.step for m in self.metrics]
        lrs = [m.learning_rate for m in self.metrics]
        return steps, lrs


class MetricsParser:
    """Parse ai-toolkit output lines to extract training metrics"""

    # Regex for ai-toolkit tqdm output
    # Example: k0f_allstar_2_zimage_turbo_512:  18%|█▊ | 544/3000 [06:24<29:50,  1.37it/s, lr: 3.0e-04 loss: 3.016e-01]
    PATTERN = re.compile(
        r'(\d+)/(\d+)\s+'           # step/total (group 1, 2)
        r'\[[\d:]+<[\d:]+,\s*'      # [elapsed<remaining,
        r'([\d.]+)it/s,\s*'         # speed (group 3)
        r'lr:\s*([\d.e+-]+)\s+'     # learning rate (group 4)
        r'loss:\s*([\d.e+-]+)\]'    # loss (group 5)
    )

    @classmethod
    def parse_line(cls, line: str) -> Optional[TrainingMetric]:
        """
        Parse a single output line.

        Args:
            line: Output line from ai-toolkit training process

        Returns:
            TrainingMetric if line contains metrics, None otherwise
        """
        match = cls.PATTERN.search(line)
        if not match:
            return None

        step, total, speed, lr, loss = match.groups()

        return TrainingMetric(
            timestamp=datetime.now().timestamp(),
            step=int(step),
            total_steps=int(total),
            progress_pct=(int(step) / int(total)) * 100,
            loss=float(loss),
            learning_rate=float(lr),
            speed=float(speed)
        )
