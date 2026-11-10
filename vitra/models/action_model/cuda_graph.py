import torch


class CUDAGraphedStep:
    """Run fn(x, t, **kwargs) through a CUDA graph, recapturing when shapes or non-tensor args change."""

    def __init__(self, fn, num_warmup=3):
        self.fn = fn
        self.num_warmup = num_warmup
        self.key = None

    def __call__(self, x, t, **kwargs):
        tensors = {'x': x, 't': t, **{k: v for k, v in kwargs.items() if torch.is_tensor(v)}}
        others = {k: v for k, v in kwargs.items() if not torch.is_tensor(v)}  # cfg_scale, None state
        key = (tuple((k, v.shape, v.dtype) for k, v in tensors.items()), tuple(sorted(others.items())))
        if key != self.key:
            self._capture(tensors, others)
            self.key = key

        for k, v in tensors.items():
            self.static_in[k].copy_(v)
        self.graph.replay()
        # The next replay overwrites static_out in place, so hand back a copy
        return self.static_out.clone()

    def _run_static(self):
        kw = {k: v for k, v in self.static_in.items() if k not in ('x', 't')}
        return self.fn(self.static_in['x'], self.static_in['t'], **kw, **self.others)

    def _capture(self, tensors, others):
        self.static_in = {k: v.clone() for k, v in tensors.items()}
        self.others = others

        # Warm up on a side stream, so capture records steady-state kernels and allocations
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.num_warmup):
                self._run_static()
        torch.cuda.current_stream().wait_stream(side)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = self._run_static()
