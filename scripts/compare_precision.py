"""
Measure how far bf16 predictions drift from fp32 on the same inputs and noise.

Loads the model once and switches settings between calls:
  - precision: fp32, or bf16 autocast (model.use_bf16, covering the vision encoder
    and Gemma; the DiT action head stays fp32)
  - DiT sampling: eager, or CUDA graphs
Every call reseeds the RNG, so paired calls draw identical diffusion noise.

Each comparison reports two levels of fidelity:
  Numerical, on raw tensors:
  - cognition token passed from Gemma to the DiT: cosine similarity, relative L2, max abs diff
  - normalized action output for the active hand: relative L2, RMSE, max abs diff
  Trajectory, on absolute hand trajectories reconstructed from the predicted
  relative actions (as inference_human_prediction.py does):
  - wrist translation error (mm), mean over all steps and at the last step
  - wrist rotation error (degrees, geodesic)
  - finger joint error (degrees, geodesic, mean over the 15 MANO joints)

Two fp32-only comparisons give the scale for judging bf16:
  - fp32 vs fp32, same seed: run-to-run GPU nondeterminism (should be ~0)
  - fp32 vs fp32, different seed: the model's own sampling spread

Example:
    python scripts/compare_precision.py \
        --config_path VITRA-VLA/VITRA-VLA-3B \
        --output ./precision_results/bf16_vs_fp32.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation as R

from inference_human_prediction import get_state
from vitra.datasets.dataset_utils import ActionFeature, StateFeature
from vitra.datasets.human_dataset import pad_action, pad_state_human
from vitra.models import load_model
from vitra.models.action_model.cuda_graph import CUDAGraphedStep
from vitra.utils.config_utils import load_config
from vitra.utils.data_utils import load_normalizer, recon_traj, resize_short_side_to_target

EXAMPLES = [
    ('./examples/0001.jpg', 'left', 'Left hand: Put the trash into the garbage. Right hand: None.'),
    ('./examples/0002.jpg', 'right', 'Left hand: None. Right hand: Pick up the picture of Michael Jackson.'),
]

# name: (use_bf16, use_cuda_graphs)
CONFIGS = {
    'fp32_eager': (False, False),
    'fp32_graphs': (False, True),
    'bf16_eager': (True, False),
    'bf16_graphs': (True, True),
}

# name: (config_a, config_b, seed offset for b)
COMPARISONS = {
    'fp32 rerun, same seed (nondeterminism floor)': ('fp32_eager', 'fp32_eager', 0),
    'fp32 graphs vs fp32 eager': ('fp32_graphs', 'fp32_eager', 0),
    'bf16 eager vs fp32 eager': ('bf16_eager', 'fp32_eager', 0),
    'bf16 graphs vs fp32 eager': ('bf16_graphs', 'fp32_eager', 0),
    'fp32, different seed (sampling spread)': ('fp32_eager', 'fp32_eager', 1000),
}


def prepare_inputs(image_path, hand, normalizer, chunk_size):
    """Build model inputs for one hand, matching inference_human_prediction.py."""
    hand_data = np.load(Path(image_path).with_suffix('.npy'), allow_pickle=True).item()

    image = Image.open(image_path)
    ori_w, ori_h = image.size
    image = ImageOps.exif_transpose(image)
    image_resized = resize_short_side_to_target(image, target=224)

    hand_state, beta, fov_x, _ = get_state(hand_data, hand_side=hand)
    fov_x = fov_x * np.pi / 180
    f_ori = ori_w / np.tan(fov_x / 2) / 2
    fov_y = 2 * np.arctan(ori_h / (2 * f_ori))

    empty = [np.zeros_like(hand_state), np.zeros_like(beta)]
    active = [hand_state, beta]
    state = np.concatenate(active + empty if hand == 'left' else empty + active, axis=0)
    hand_mask = [hand == 'left', hand == 'right']
    state_mask = np.array(hand_mask, dtype=bool)
    action_mask = np.tile(np.array([hand_mask], dtype=bool), (chunk_size, 1))

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

    inputs = {
        'image': np.array(image_resized),
        'current_state': unified_state.unsqueeze(0),
        'current_state_mask': unified_state_mask.unsqueeze(0),
        'action_mask_torch': unified_action_mask.unsqueeze(0),
        'fov': torch.from_numpy(np.array([fov_x, fov_y], dtype=np.float32)).unsqueeze(0),
    }
    return inputs, hand_state


def to_trajectories(norm_action, normalizer, hand, hand_state):
    """Normalized model output [S, T, 192] -> absolute trajectories [S, T+1, 51]."""
    unnorm = normalizer.unnormalize_action(norm_action[:, :, :102])
    start = 0 if hand == 'left' else 51
    return np.stack([recon_traj(state=hand_state, rel_action=a[:, start:start + 51]) for a in unnorm])


def geodesic_deg(euler_a, euler_b):
    """Angle between rotations given as xyz Euler angles, in degrees."""
    rel = R.from_euler('xyz', euler_a.reshape(-1, 3)).inv() * R.from_euler('xyz', euler_b.reshape(-1, 3))
    return np.degrees(rel.magnitude()).reshape(euler_a.shape[:-1] + (-1,))


def trajectory_errors(traj_a, traj_b):
    """Errors between paired trajectories [S, T+1, 51], ignoring the shared initial state."""
    a, b = traj_a[:, 1:], traj_b[:, 1:]
    transl_mm = np.linalg.norm(a[..., :3] - b[..., :3], axis=-1) * 1000
    wrist_deg = geodesic_deg(a[..., 3:6], b[..., 3:6])[..., 0]
    joints_deg = geodesic_deg(a[..., 6:], b[..., 6:]).mean(axis=-1)
    return {
        'wrist_transl_mm_mean': transl_mm.mean(),
        'wrist_transl_mm_last_step': transl_mm[:, -1].mean(),
        'wrist_transl_mm_max': transl_mm.max(),
        'wrist_rot_deg_mean': wrist_deg.mean(),
        'wrist_rot_deg_max': wrist_deg.max(),
        'joint_rot_deg_mean': joints_deg.mean(),
        'joint_rot_deg_max': joints_deg.max(),
    }


def tensor_errors(a, b, prefix):
    """Numerical differences between raw tensors a and reference b."""
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    diff = a - b
    return {
        f'{prefix}_rel_l2': np.linalg.norm(diff) / np.linalg.norm(b),
        f'{prefix}_rmse': np.sqrt(np.mean(diff ** 2)),
        f'{prefix}_max_abs_diff_max': np.abs(diff).max(),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare bf16 and fp32 VLA predictions.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to model configuration JSON file')
    parser.add_argument('--num_seeds', type=int, default=8, help='Seeds per example; each gives sample_times samples')
    parser.add_argument('--sample_times', type=int, default=4, help='Action samples per prediction')
    parser.add_argument('--num_ddim_steps', type=int, default=10, help='DDIM sampling steps')
    parser.add_argument('--cfg_scale', type=float, default=5.0, help='Classifier-free guidance scale')
    parser.add_argument('--output', type=str, required=True, help='Path to save the JSON results')
    args = parser.parse_args()

    configs = load_config(args.config_path)
    model = load_model(configs).cuda()
    model.eval()
    normalizer = load_normalizer(configs)
    chunk_size = configs.get('fwd_pred_next_n', 16)

    # Keep the cognition token from each call, to see how much drift starts before the DiT
    cognition = {}
    extract = model.extract_cognition_token

    def extract_and_keep(*a, **kw):
        out = extract(*a, **kw)
        cognition['last'] = out.detach().float().cpu()
        return out

    model.extract_cognition_token = extract_and_keep

    graphed_step = CUDAGraphedStep(model.act_model.net.forward_with_cfg)

    def set_config(name):
        use_bf16, use_graphs = CONFIGS[name]
        model.use_bf16 = use_bf16
        model.act_model.graphed_cfg_step = graphed_step if use_graphs else model.act_model.net.forward_with_cfg

    def predict(inputs, instruction, seed):
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = model.predict_action(
            instruction=instruction,
            num_ddim_steps=args.num_ddim_steps,
            cfg_scale=args.cfg_scale,
            sample_times=args.sample_times,
            **inputs,
        )
        torch.cuda.synchronize()
        return out, (time.perf_counter() - start) * 1000, cognition['last']

    results = {'examples': {}}
    latencies = {name: [] for name in CONFIGS}
    for image_path, hand, instruction in EXAMPLES:
        inputs, hand_state = prepare_inputs(image_path, hand, normalizer, chunk_size)

        # outputs[(config, seed)] = (trajectories, cognition token, normalized actions of the active hand)
        hand_dims = slice(0, 51) if hand == 'left' else slice(51, 102)
        outputs = {}

        def record(key, out, cog):
            outputs[key] = (to_trajectories(out, normalizer, hand, hand_state), cog, out[:, :, hand_dims])

        for name in CONFIGS:
            set_config(name)
            predict(inputs, instruction, seed=12345)  # warm up: graph capture, kernel selection
            for seed in range(args.num_seeds):
                for offset in {c[2] for c in COMPARISONS.values()}:
                    out, ms, cog = predict(inputs, instruction, seed + offset)
                    record((name, seed + offset), out, cog)
                    latencies[name].append(ms)
            # Second fp32 eager pass with the same seeds, for the nondeterminism floor
            if name == 'fp32_eager':
                for seed in range(args.num_seeds):
                    out, _, cog = predict(inputs, instruction, seed)
                    record(('fp32_eager_rerun', seed), out, cog)

        example = {}
        for label, (cfg_a, cfg_b, offset) in COMPARISONS.items():
            if cfg_a == cfg_b and offset == 0:
                cfg_a = 'fp32_eager_rerun'
            per_seed = []
            cos = []
            for seed in range(args.num_seeds):
                traj_a, cog_a, act_a = outputs[(cfg_a, seed)]
                traj_b, cog_b, act_b = outputs[(cfg_b, seed + offset)]
                per_seed.append({
                    **tensor_errors(cog_a, cog_b, 'cognition_token'),
                    **tensor_errors(act_a, act_b, 'norm_action'),
                    **trajectory_errors(traj_a, traj_b),
                })
                cos.append(torch.nn.functional.cosine_similarity(cog_a.flatten(), cog_b.flatten(), dim=0).item())
            metrics = {k: float(np.max([m[k] for m in per_seed]) if k.endswith('_max') else np.mean([m[k] for m in per_seed]))
                       for k in per_seed[0]}
            metrics['cognition_token_cosine'] = float(np.mean(cos))
            example[label] = metrics
        results['examples'][f'{image_path} ({hand} hand)'] = example

    results['latency_ms_mean'] = {name: float(np.mean(v)) for name, v in latencies.items()}
    results['setup'] = {
        'model': configs.get('model_load_path'),
        'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(),
        'num_seeds': args.num_seeds,
        'sample_times': args.sample_times,
        'num_ddim_steps': args.num_ddim_steps,
        'cfg_scale': args.cfg_scale,
        'bf16_scope': 'autocast for vision encoder and Gemma; DiT action head in fp32',
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))

    tables = {
        'Numerical fidelity': [
            ('cog cos', 'cognition_token_cosine', '.6f'),
            ('cog relL2', 'cognition_token_rel_l2', '.2e'),
            ('act relL2', 'norm_action_rel_l2', '.2e'),
            ('act RMSE', 'norm_action_rmse', '.2e'),
            ('act maxabs', 'norm_action_max_abs_diff_max', '.2e'),
        ],
        'Trajectory fidelity': [
            ('transl mm', 'wrist_transl_mm_mean', '.3f'),
            ('last mm', 'wrist_transl_mm_last_step', '.3f'),
            ('max mm', 'wrist_transl_mm_max', '.3f'),
            ('wrist deg', 'wrist_rot_deg_mean', '.3f'),
            ('joint deg', 'joint_rot_deg_mean', '.3f'),
        ],
    }
    for example_name, example in results['examples'].items():
        for title, cols in tables.items():
            print(f"\n{example_name}: {title}")
            print(f"  {'comparison':46s} " + ' '.join(f"{h:>10s}" for h, _, _ in cols))
            for label, m in example.items():
                print(f"  {label:46s} " + ' '.join(f"{m[k]:>10{fmt}}" for _, k, fmt in cols))
    print('\nMean latency per call (ms, includes RNG reseed; use benchmark_inference.py for timing):')
    for name, ms in results['latency_ms_mean'].items():
        print(f"  {name:12s} {ms:.1f}")
    print(f"Saved results to {output}")


if __name__ == '__main__':
    main()
