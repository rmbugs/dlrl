import pytest
import subprocess
import numpy as np
from datetime import datetime
import os
import json
import logging
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
out_dir = 'benchmark_results'
os.makedirs(out_dir, exist_ok=True)

@pytest.fixture
def dry_run(request):
    return request.config.getoption("--dry-run")

def parse_output(output):
    lines = output.split('\n')
    metrics = {
        'train_loss': None,
        'test_acc': None,
        'training_time': None,
    }

    for line in lines:
        if "Test Acc" in line:
            acc = float(line.split()[-1])
            metrics['test_acc'] = acc
        elif "Train Loss" in line:
            loss = float(line.split()[-1])
            metrics['train_loss'] = loss
        elif "Total training time:" in line:  # Parse training time
            time_str = line.split()[-2]  # Get the number before "seconds"
            metrics['training_time'] = float(time_str)

    return metrics

class TestPruningBenchmarks:
    @pytest.fixture(params=['MNIST', 'CIFAR10'])
    def dataset(self, request):
        return request.param

    @pytest.fixture(params=['random', 'loss', 'loss-grad', 'el2n'])
    def strategy(self, request):
        return request.param

    @pytest.fixture(params=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
    def ratio(self, request):
        return request.param

    @pytest.fixture
    def pruning_epochs_by_dataset(self, dataset):
        return [1, 2, 3, 4, 5] if dataset == 'MNIST' else [4, 8, 12, 16, 20]

    @pytest.fixture(params=range(5))  # Index into the pruning epoch list
    def pruning_epoch_idx(self, request):
        return request.param

    def save_result(self, metrics, extra_info):
        """Save experiment result to JSON file"""
        result = {**metrics, **extra_info, 'timestamp': datetime.now().isoformat()}

        # Load existing results
        result_file = 'results.json'
        existing_results = []
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            with open(result_file, 'r') as f:
                try:
                    existing_results = json.load(f)
                except json.JSONDecodeError:
                    existing_results = []

        # Append new result
        existing_results.append(result)

        # Save updated results
        with open(result_file, 'w') as f:
            json.dump(existing_results, f, indent=2)

    def run_command(self, cmd):
        """Run command and log output in real-time"""
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True,
                                 bufsize=1)  # Line buffered

        # Store all output lines
        output_lines = []
        for line in process.stderr:
            line = line.strip()
            logger.info(line)
            output_lines.append(line)

        process.communicate()  # Ensure process is complete
        return parse_output('\n'.join(output_lines))

    def test_ratio_based_pruning(self, dataset, strategy, ratio, dry_run):
        """Test pruning with varying ratios"""
        # Skip strategy variations when not pruning
        if ratio == 0 and strategy != 'random':  # Use 'random' as the default for no-pruning case
            pytest.skip("No need to test different strategies when not pruning")

        cmd = [
            "python", "dlrl.py",
            "--dataset", dataset,
            "--pruning-ratio", f"{ratio:.1f}"
        ]

        # Only add pruning strategy if we're actually pruning
        if ratio > 0:
            cmd.extend(["--pruning-strategy", strategy])

        if dry_run:
            pytest.skip("Dry run mode - skipping actual benchmark")

        metrics = self.run_command(cmd)

        # Save results
        self.save_result(metrics, {
            'dataset': dataset,
            'strategy': strategy if ratio > 0 else 'none',
            'ratio': ratio,
            'type': 'ratio_based'
        })

        # Basic assertion to ensure the experiment runs successfully
        assert metrics['test_acc'] is not None
        assert metrics['train_loss'] is not None

    def test_epoch_based_pruning(self, dataset, strategy, pruning_epochs_by_dataset, pruning_epoch_idx, dry_run):
        """Test epoch-based pruning with dataset-specific pruning epochs"""
        if strategy not in ['loss', 'loss-grad', 'el2n']:
            pytest.skip("Epoch-based pruning only applies to loss, el2n and loss-grad strategies")

        cmd = [
            "python", "dlrl.py",
            "--dataset", dataset,
            "--pruning-ratio", "0.5",
            "--pruning-strategy", strategy,
            "--pruning-epoch", str(pruning_epochs_by_dataset[pruning_epoch_idx])
        ]

        if dry_run:
            pytest.skip("Dry run mode - skipping actual benchmark")

        metrics = self.run_command(cmd)

        # Save results
        self.save_result(metrics, {
            'dataset': dataset,
            'strategy': strategy,
            'pruning_epoch': pruning_epochs_by_dataset[pruning_epoch_idx],
            'type': 'epoch_based'
        })

        # Basic assertion to ensure the experiment runs successfully
        assert metrics['test_acc'] is not None
        assert metrics['train_loss'] is not None
