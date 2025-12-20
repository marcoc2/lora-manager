"""
Training Report Generator

Generates training reports with loss graphs and configuration details.
Uses Pydantic models for structured JSON output.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless generation
import matplotlib.pyplot as plt

from services.training_metrics import MetricsCollector


class TrainingStats(BaseModel):
    """Training statistics summary"""
    final_loss: float = 0.0
    min_loss: float = 0.0
    max_loss: float = 0.0
    avg_loss: float = 0.0
    steps_completed: int = 0
    total_steps: int = 0
    training_duration_seconds: float = 0.0
    training_duration_formatted: str = "N/A"
    avg_speed_its: float = 0.0


class LossDataPoint(BaseModel):
    """Single loss data point for graphing"""
    step: int
    loss: float
    learning_rate: float
    timestamp: float


class TrainingReport(BaseModel):
    """Complete training report with Pydantic validation"""
    version: str = "1.0"
    task_name: str
    model_type: str
    date: str = Field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Statistics
    stats: TrainingStats = Field(default_factory=TrainingStats)

    # Loss history for potential re-plotting
    loss_history: List[LossDataPoint] = Field(default_factory=list)

    # Full training configuration
    training_config: Dict[str, Any] = Field(default_factory=dict)

    # Paths to generated artifacts
    loss_graph_path: Optional[str] = None


class TrainingReportGenerator:
    """Generate training report with graphs and JSON summary"""

    def generate(self, collector: MetricsCollector, output_dir: Path) -> Path:
        """
        Generate report and save to output_dir.

        Args:
            collector: MetricsCollector with training data
            output_dir: Directory to save the report (same as preview images)

        Returns:
            Path to the generated JSON report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build report data
        report = self._build_report(collector, output_dir)

        # Generate loss graph
        loss_graph_path = self._generate_loss_graph(collector, output_dir)
        if loss_graph_path:
            report.loss_graph_path = str(loss_graph_path.name)

        # Save JSON report
        report_path = output_dir / "training_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report.model_dump_json(indent=2))

        return report_path

    def _build_report(self, collector: MetricsCollector, output_dir: Path) -> TrainingReport:
        """Build the TrainingReport from collected metrics"""
        metrics = collector.metrics
        losses = [m.loss for m in metrics]

        # Calculate statistics
        stats = TrainingStats()
        if losses:
            stats.final_loss = losses[-1]
            stats.min_loss = min(losses)
            stats.max_loss = max(losses)
            stats.avg_loss = sum(losses) / len(losses)

        if metrics:
            stats.steps_completed = metrics[-1].step
            stats.total_steps = metrics[-1].total_steps

            # Duration
            if len(metrics) > 1:
                duration = metrics[-1].timestamp - collector.start_time
                stats.training_duration_seconds = duration
                stats.training_duration_formatted = self._format_duration(duration)

            # Speed
            speeds = [m.speed for m in metrics]
            stats.avg_speed_its = sum(speeds) / len(speeds) if speeds else 0

        # Build loss history
        loss_history = [
            LossDataPoint(
                step=m.step,
                loss=m.loss,
                learning_rate=m.learning_rate,
                timestamp=m.timestamp
            )
            for m in metrics
        ]

        return TrainingReport(
            task_name=collector.task_name,
            model_type=collector.model_type,
            stats=stats,
            loss_history=loss_history,
            training_config=collector.config
        )

    def _generate_loss_graph(self, collector: MetricsCollector, output_dir: Path) -> Optional[Path]:
        """Create loss vs steps graph"""
        steps, losses = collector.get_loss_series()

        if not steps:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Raw data (semi-transparent)
        ax.plot(steps, losses, 'b-', linewidth=1, alpha=0.4, label='Loss')

        # Moving average for smoother visualization
        if len(losses) > 10:
            window = min(50, len(losses) // 10)
            smooth_losses = self._moving_average(losses, window)
            # Adjust x-axis for moving average (starts at window-1)
            ma_steps = steps[window-1:]
            ax.plot(ma_steps, smooth_losses, 'r-', linewidth=2, label=f'MA({window})')

        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training Loss - {collector.task_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set y-axis to start from 0 or slightly below min for better visualization
        if losses:
            min_loss = min(losses)
            max_loss = max(losses)
            margin = (max_loss - min_loss) * 0.1
            ax.set_ylim(max(0, min_loss - margin), max_loss + margin)

        # Save
        graph_path = output_dir / "loss_graph.png"
        fig.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return graph_path

    @staticmethod
    def _moving_average(data: list, window: int) -> list:
        """Calculate moving average"""
        return [sum(data[i:i+window])/window for i in range(len(data)-window+1)]

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m {secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
