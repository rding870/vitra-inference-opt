"""
Repeatable end-to-end latency benchmark for VLA inference.

Times model.predict_action, from raw image and instruction to a numpy array on
the CPU, with the profiler off. Each result file records everything needed to
judge whether two runs are comparable: the model revision, inputs, precision,
sampling settings, and hardware.

Example:
    python scripts/benchmark_inference.py \
        --config_path VITRA-VLA/VITRA-VLA-3B \
        --image_path ./examples/0002.jpg \
        --output ./benchmark_results/baseline.json
"""
import argparse
import collections
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pynvml
import torch
from huggingface_hub import HfApi
from PIL import Image, ImageOps

from inference_human_prediction import get_state
from vitra.datasets.dataset_utils import ActionFeature, StateFeature
from vitra.datasets.human_dataset import pad_action, pad_state_human
from vitra.models import load_model
from vitra.utils.config_utils import load_config
from vitra.utils.data_utils import load_normalizer, resize_short_side_to_target


class GpuMonitor:
    """Samples GPU utilization and used memory on a background thread."""

    def __init__(self, device_index, interval_s=0.1):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval_s = interval_s
        self.utilization = []
        self.memory_used_mib = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval_s):
            self.utilization.append(pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
            self.memory_used_mib.append(pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 2**20)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        pynvml.nvmlShutdown()


def prepare_inputs(args, normalizer, chunk_size):
    """Build one fixed set of model inputs, matching inference_human_prediction.py."""
    hand_data = np.load(Path(args.image_path).with_suffix('.npy'), allow_pickle=True).item()

    image = Image.open(args.image_path)
    ori_w, ori_h = image.size
    image = ImageOps.exif_transpose(image)
    image_resized = resize_short_side_to_target(image, target=224)

    state_right, beta_right, fov_x, _ = get_state(hand_data, hand_side='right')
    fov_x = fov_x * np.pi / 180
    f_ori = ori_w / np.tan(fov_x / 2) / 2
    fov_y = 2 * np.arctan(ori_h / (2 * f_ori))

    state = np.concatenate(
        [np.zeros_like(state_right), np.zeros_like(beta_right), state_right, beta_right], axis=0
    )
    state_mask = np.array([False, True], dtype=bool)
    action_mask = np.tile(np.array([[False, True]], dtype=bool), (chunk_size, 1))

    unified_state, unified_state_mask = pad_state_human(
        state=normalizer.normalize_state(state.copy()),
        state_mask=state_mask,
        action_dim=normalizer.action_mean.shape[0],
        state_dim=normalizer.state_mean.shape[0],
        unified_state_dim=StateFeature.ALL_FEATURES[1],
    )
    _, unified_action_mask = pad_action(
        actions=None,
        action_mask=action_mask.copy(),
        action_dim=normalizer.action_mean.shape[0],
        unified_action_dim=ActionFeature.ALL_FEATURES[1],
    )

    return {
        'image': np.array(image_resized),
        'current_state': unified_state.unsqueeze(0),
        'current_state_mask': unified_state_mask.unsqueeze(0),
        'action_mask_torch': unified_action_mask.unsqueeze(0),
        'fov': torch.from_numpy(np.array([fov_x, fov_y], dtype=np.float32)).unsqueeze(0),
    }


def model_revision(model_load_path):
    """Return the Hub commit of the checkpoint repo, or None for a local checkpoint."""
    if os.path.exists(model_load_path):
        return None
    return HfApi().model_info(model_load_path).sha


def summarize(values):
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'p50': float(np.percentile(values, 50)),
        'p95': float(np.percentile(values, 95)),
        'p99': float(np.percentile(values, 99)),
        'max': float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser(description='End-to-end latency benchmark for VLA inference.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to model configuration JSON file')
    parser.add_argument('--image_path', type=str, required=True, help='Input image; needs a matching precomputed .npy hand state')
    parser.add_argument('--instruction', type=str, default='Left hand: None. Right hand: Pick up the picture of Michael Jackson.', help='Text instruction')
    parser.add_argument('--sample_times', type=int, default=4, help='Action samples per prediction')
    parser.add_argument('--num_ddim_steps', type=int, default=10, help='DDIM sampling steps')
    parser.add_argument('--cfg_scale', type=float, default=5.0, help='Classifier-free guidance scale')
    parser.add_argument('--warmup', type=int, default=10, help='Initial predictions to exclude from the statistics')
    parser.add_argument('--iterations', type=int, default=200, help='Predictions to measure')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_bf16', action='store_true', help='Run the vision encoder and Gemma under bf16 autocast (the DiT action head stays fp32)')
    parser.add_argument('--bf16_weights', action='store_true', help='Also store their linear weights in bf16, so autocast does not re-cast them every call (requires --use_bf16)')
    parser.add_argument('--no_cuda_graphs', action='store_true', help='Run the DiT sampling steps eagerly instead of through a CUDA graph')
    parser.add_argument('--compile_dit', action='store_true', help='torch.compile the DiT sampling step before CUDA graph capture')
    parser.add_argument('--output', type=str, required=True, help='Path to save the JSON results')
    args = parser.parse_args()
    if args.bf16_weights and not args.use_bf16:
        parser.error('--bf16_weights requires --use_bf16')

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.manual_seed(args.seed)

    configs = load_config(args.config_path)
    load_start = time.perf_counter()
    model = load_model(configs).cuda()
    model.eval()
    model.use_bf16 = args.use_bf16
    model.set_bf16_weights(args.bf16_weights)
    model.act_model.compile_step = args.compile_dit
    if args.no_cuda_graphs:
        # predict_action only builds the graphed step when this is None, so this keeps sampling eager
        model.act_model.graphed_cfg_step = model.act_model.net.forward_with_cfg
    load_s = time.perf_counter() - load_start
    normalizer = load_normalizer(configs)

    inputs = prepare_inputs(args, normalizer, configs.get('fwd_pred_next_n', 16))

    def predict():
        start = time.perf_counter()
        # predict_action copies its result to the CPU, so GPU work is finished when it returns
        model.predict_action(
            instruction=args.instruction,
            num_ddim_steps=args.num_ddim_steps,
            cfg_scale=args.cfg_scale,
            sample_times=args.sample_times,
            **inputs,
        )
        return (time.perf_counter() - start) * 1000

    warmup_ms = [predict() for _ in range(args.warmup)]

    torch.cuda.reset_peak_memory_stats()
    with GpuMonitor(torch.cuda.current_device()) as monitor:
        measure_start = time.perf_counter()
        latencies_ms = [predict() for _ in range(args.iterations)]
        measure_s = time.perf_counter() - measure_start

    latency = summarize(latencies_ms)
    results = {
        'setup': {
            'model': configs.get('model_load_path'),
            'model_revision': model_revision(configs['model_load_path']),
            'git_commit': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip(),
            'git_dirty': bool(subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip()),
            'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(),
            'param_dtypes': dict(collections.Counter(str(p.dtype) for p in model.parameters())),
            'use_bf16': model.use_bf16,
            'bf16_weights': args.bf16_weights,
            'cuda_graphs': not args.no_cuda_graphs,
            'compile_dit': args.compile_dit,
            'grad_enabled': torch.is_grad_enabled(),
            'params_requiring_grad': sum(p.requires_grad for p in model.parameters()),
            'batch_size': 1,
            'image_path': args.image_path,
            'image_shape': list(inputs['image'].shape),
            'instruction': args.instruction,
            'instruction_tokens': len(model.tokenizer(args.instruction).input_ids),
            'sample_times': args.sample_times,
            'num_ddim_steps': args.num_ddim_steps,
            'cfg_scale': args.cfg_scale,
            'seed': args.seed,
            'warmup': args.warmup,
            'iterations': args.iterations,
        },
        'load_s': load_s,
        'first_call_ms': warmup_ms[0] if warmup_ms else None,
        'warmup_ms': warmup_ms,
        'latency_ms': latency,
        'throughput': {
            'predictions_per_s': args.iterations / measure_s,
            'samples_per_s': args.iterations * args.sample_times / measure_s,
        },
        'gpu': {
            'utilization_pct': summarize(monitor.utilization),
            'nvml_memory_used_mib_max': max(monitor.memory_used_mib),
            'torch_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'torch_peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
        },
        'latencies_ms': latencies_ms,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))

    print(f"\nLatency (ms): p50={latency['p50']:.1f} p95={latency['p95']:.1f} p99={latency['p99']:.1f}")
    print(f"Throughput: {results['throughput']['predictions_per_s']:.2f} predictions/s")
    print(f"GPU utilization: {results['gpu']['utilization_pct']['mean']:.0f}% mean")
    print(f"Peak GPU memory: {results['gpu']['torch_peak_allocated_mib']:.0f} MiB allocated")
    print(f"Saved results to {output}")


if __name__ == '__main__':
    main()
