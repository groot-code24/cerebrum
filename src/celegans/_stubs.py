"""Pure-NumPy/stdlib fallback stubs for optional heavy dependencies.

When torch, torch_geometric, snntorch, gymnasium, or pydantic_settings are not
installed, these stubs provide compatible interfaces so the project runs in
offline / CI environments.  Production use should install the real packages.
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# torch stub
# ---------------------------------------------------------------------------

class _Tensor:
    """Minimal numpy-backed Tensor mirroring the torch.Tensor API used here."""

    def __init__(self, data: Any, dtype: Any = None) -> None:
        if isinstance(data, _Tensor):
            arr = data._arr
        elif isinstance(data, np.ndarray):
            arr = data.astype(np.float32)
        else:
            arr = np.array(data, dtype=np.float32)
        if dtype is not None:
            if dtype == _long:
                arr = arr.astype(np.int64)
            elif dtype == _bool_type:
                arr = arr.astype(bool)
            else:
                arr = arr.astype(np.float32)
        self._arr = arr

    # Shape / dtype
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._arr.shape

    @property
    def dtype(self) -> Any:
        return self._arr.dtype

    @property
    def ndim(self) -> int:
        return self._arr.ndim

    def dim(self) -> int:
        return self._arr.ndim

    def size(self, dim: Optional[int] = None) -> Any:
        if dim is None:
            return self.shape
        return self.shape[dim]

    # Conversion
    def numpy(self) -> np.ndarray:
        return self._arr

    def item(self) -> Any:
        return self._arr.item()

    def tolist(self) -> Any:
        return self._arr.tolist()

    def detach(self) -> "_Tensor":
        return _Tensor(self._arr.copy())

    def cpu(self) -> "_Tensor":
        return _Tensor(self._arr.copy())

    def to(self, *args: Any, **kwargs: Any) -> "_Tensor":
        return self

    def float(self) -> "_Tensor":
        return _Tensor(self._arr.astype(np.float32))

    def long(self) -> "_Tensor":
        return _Tensor(self._arr.astype(np.int64))

    def bool(self) -> "_Tensor":
        return _Tensor(self._arr.astype(bool))

    def clone(self) -> "_Tensor":
        return _Tensor(self._arr.copy())

    def view(self, *shape: int) -> "_Tensor":
        return _Tensor(self._arr.reshape(shape))

    def reshape(self, *shape: Any) -> "_Tensor":
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return _Tensor(self._arr.reshape(shape))

    def squeeze(self, dim: Optional[int] = None) -> "_Tensor":
        if dim is None:
            return _Tensor(self._arr.squeeze())
        return _Tensor(np.squeeze(self._arr, axis=dim))

    def unsqueeze(self, dim: int) -> "_Tensor":
        return _Tensor(np.expand_dims(self._arr, axis=dim))

    def permute(self, *dims: int) -> "_Tensor":
        return _Tensor(self._arr.transpose(dims))

    def transpose(self, d0: int, d1: int) -> "_Tensor":
        axes = list(range(self._arr.ndim))
        axes[d0], axes[d1] = axes[d1], axes[d0]
        return _Tensor(self._arr.transpose(axes))

    def contiguous(self) -> "_Tensor":
        return _Tensor(np.ascontiguousarray(self._arr))

    def flatten(self) -> "_Tensor":
        return _Tensor(self._arr.flatten())

    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> "_Tensor":
        return _Tensor(self._arr.sum(axis=dim, keepdims=keepdim))

    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> "_Tensor":
        return _Tensor(self._arr.mean(axis=dim, keepdims=keepdim))

    def max(self, dim: Optional[int] = None) -> "_Tensor":
        if dim is None:
            return _Tensor(np.array(self._arr.max()))
        return _Tensor(self._arr.max(axis=dim))

    def min(self, dim: Optional[int] = None) -> "_Tensor":
        if dim is None:
            return _Tensor(np.array(self._arr.min()))
        return _Tensor(self._arr.min(axis=dim))

    def abs(self) -> "_Tensor":
        return _Tensor(np.abs(self._arr))

    def clamp(self, min: Optional[float] = None, max: Optional[float] = None) -> "_Tensor":
        return _Tensor(np.clip(self._arr, min, max))

    def unique(self) -> "_Tensor":
        return _Tensor(np.unique(self._arr))

    def any(self) -> "_Tensor":
        return _Tensor(np.array(self._arr.any()))

    def all(self) -> "_Tensor":
        return _Tensor(np.array(self._arr.all()))

    def stack(self, *args: Any) -> "_Tensor":
        return _Tensor(np.stack([self._arr] + [a._arr for a in args]))

    # Indexing
    def __getitem__(self, idx: Any) -> "_Tensor":
        if isinstance(idx, _Tensor):
            idx = idx._arr
        elif isinstance(idx, tuple):
            idx = tuple(i._arr if isinstance(i, _Tensor) else i for i in idx)
        result = self._arr[idx]
        return _Tensor(result)

    def __setitem__(self, idx: Any, val: Any) -> None:
        if isinstance(idx, _Tensor):
            idx = idx._arr
        if isinstance(val, _Tensor):
            val = val._arr
        self._arr[idx] = val

    def __len__(self) -> int:
        return len(self._arr)

    def __iter__(self) -> Iterator["_Tensor"]:
        for row in self._arr:
            yield _Tensor(row)

    # Arithmetic
    def _coerce(self, other: Any) -> np.ndarray:
        if isinstance(other, _Tensor):
            return other._arr
        return np.array(other, dtype=np.float32)

    def __add__(self, other: Any) -> "_Tensor":
        return _Tensor(self._arr + self._coerce(other))

    def __radd__(self, other: Any) -> "_Tensor":
        return _Tensor(self._coerce(other) + self._arr)

    def __sub__(self, other: Any) -> "_Tensor":
        return _Tensor(self._arr - self._coerce(other))

    def __rsub__(self, other: Any) -> "_Tensor":
        return _Tensor(self._coerce(other) - self._arr)

    def __mul__(self, other: Any) -> "_Tensor":
        return _Tensor(self._arr * self._coerce(other))

    def __rmul__(self, other: Any) -> "_Tensor":
        return _Tensor(self._coerce(other) * self._arr)

    def __truediv__(self, other: Any) -> "_Tensor":
        return _Tensor(self._arr / self._coerce(other))

    def __neg__(self) -> "_Tensor":
        return _Tensor(-self._arr)

    def __matmul__(self, other: "_Tensor") -> "_Tensor":
        return _Tensor(self._arr @ other._arr)

    # Comparisons
    def __eq__(self, other: Any) -> "_Tensor":  # type: ignore[override]
        return _Tensor(self._arr == self._coerce(other))

    def __ne__(self, other: Any) -> "_Tensor":  # type: ignore[override]
        return _Tensor(self._arr != self._coerce(other))

    def __ge__(self, other: Any) -> "_Tensor":
        return _Tensor(self._arr >= self._coerce(other))

    def __le__(self, other: Any) -> "_Tensor":
        return _Tensor(self._arr <= self._coerce(other))

    def __gt__(self, other: Any) -> "_Tensor":
        return _Tensor(self._arr > self._coerce(other))

    def __lt__(self, other: Any) -> "_Tensor":
        return _Tensor(self._arr < self._coerce(other))

    def __and__(self, other: "_Tensor") -> "_Tensor":
        return _Tensor(self._arr & other._arr)

    def __or__(self, other: "_Tensor") -> "_Tensor":
        return _Tensor(self._arr | other._arr)

    def __repr__(self) -> str:
        return f"Tensor({self._arr})"


# Dtype sentinels
_float32 = np.float32
_float = np.float32
_long = np.int64
_bool_type = bool
_int64 = np.int64


class _TorchModule:
    """Minimal nn.Module stub."""

    def __init__(self) -> None:
        self._parameters: Dict[str, Any] = {}
        self._modules: Dict[str, Any] = {}
        self._buffers: Dict[str, Any] = {}
        self.training = True

    def register_buffer(self, name: str, tensor: Any) -> None:
        self._buffers[name] = tensor
        setattr(self, name, tensor)

    def register_parameter(self, name: str, param: Any) -> None:
        self._parameters[name] = param

    def named_parameters(self, *a: Any, **kw: Any) -> Iterator[Tuple[str, Any]]:
        yield from self._parameters.items()

    def parameters(self, *a: Any, **kw: Any) -> Iterator[Any]:
        yield from self._parameters.values()

    def train(self, mode: bool = True) -> "_TorchModule":
        self.training = mode
        for m in self._modules.values():
            if hasattr(m, "train"):
                m.train(mode)
        return self

    def eval(self) -> "_TorchModule":
        return self.train(False)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, _TorchModule):
            if hasattr(self, "_modules"):
                self._modules[name] = value
        super().__setattr__(name, value)

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, d: Dict[str, Any], strict: bool = True) -> None:
        pass


class _Linear(_TorchModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = _Tensor(np.random.randn(out_features, in_features).astype(np.float32) * scale)
        self.bias_vec = _Tensor(np.zeros(out_features, dtype=np.float32)) if bias else None
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: _Tensor) -> _Tensor:
        out = _Tensor(x._arr @ self.weight._arr.T)
        if self.bias_vec is not None:
            out = out + self.bias_vec
        return out


class _LayerNorm(_TorchModule):
    def __init__(self, normalized_shape: Any, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: _Tensor) -> _Tensor:
        arr = x._arr
        mean = arr.mean(axis=-1, keepdims=True)
        var = arr.var(axis=-1, keepdims=True)
        return _Tensor((arr - mean) / np.sqrt(var + self.eps))


class _ReLU(_TorchModule):
    def forward(self, x: _Tensor) -> _Tensor:
        return _Tensor(np.maximum(0, x._arr))


class _Dropout(_TorchModule):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: _Tensor) -> _Tensor:
        if not self.training or self.p == 0:
            return x
        mask = np.random.binomial(1, 1 - self.p, x._arr.shape).astype(np.float32)
        return _Tensor(x._arr * mask / (1 - self.p))


class _Identity(_TorchModule):
    def forward(self, x: _Tensor) -> _Tensor:
        return x


class _ModuleList(_TorchModule):
    def __init__(self, modules: Optional[List[_TorchModule]] = None) -> None:
        super().__init__()
        self._list: List[_TorchModule] = modules or []
        for i, m in enumerate(self._list):
            self._modules[str(i)] = m

    def append(self, module: _TorchModule) -> None:
        self._list.append(module)
        self._modules[str(len(self._list) - 1)] = module

    def __iter__(self) -> Iterator[_TorchModule]:
        return iter(self._list)

    def __len__(self) -> int:
        return len(self._list)


class _SAGEConv(_TorchModule):
    """GraphSAGE-style mean aggregation convolution (numpy backend)."""

    def __init__(self, in_channels: int, out_channels: int, aggr: str = "mean") -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.lin_self = _Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = _Linear(in_channels, out_channels, bias=False)
        self.bias = _Tensor(np.zeros(out_channels, dtype=np.float32))

    def forward(self, x: _Tensor, edge_index: _Tensor) -> _Tensor:
        n = x._arr.shape[0]
        src = edge_index._arr[0]
        tgt = edge_index._arr[1]

        # Aggregate neighbour features
        agg = np.zeros((n, self.in_channels), dtype=np.float32)
        counts = np.zeros(n, dtype=np.float32)
        np.add.at(agg, tgt, x._arr[src])
        np.add.at(counts, tgt, 1)
        if self.aggr == "mean":
            mask = counts > 0
            agg[mask] /= counts[mask, None]

        self_feat = self.lin_self.forward(x)
        neigh_feat = self.lin_neigh.forward(_Tensor(agg))
        return _Tensor(self_feat._arr + neigh_feat._arr + self.bias._arr)


class _GATConv(_TorchModule):
    """Stub Graph Attention Network conv (uses SAGEConv internally)."""

    def __init__(self, in_ch: int, out_ch: int, heads: int = 1,
                 concat: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = heads
        self.out_ch = out_ch
        self.concat = concat
        actual_out = out_ch * heads if concat else out_ch
        self._conv = _SAGEConv(in_ch, actual_out)

    def forward(self, x: _Tensor, edge_index: _Tensor,
                return_attention_weights: bool = False) -> Any:
        out = self._conv.forward(x, edge_index)
        if return_attention_weights:
            n_edges = edge_index._arr.shape[1]
            dummy_attn = _Tensor(np.ones(n_edges, dtype=np.float32) / self.heads)
            return out, (edge_index, dummy_attn)
        return out


# ---------------------------------------------------------------------------
# torch module namespace
# ---------------------------------------------------------------------------

class _TorchNNNamespace:
    Module = _TorchModule
    Linear = _Linear
    LayerNorm = _LayerNorm
    ReLU = _ReLU
    Dropout = _Dropout
    Identity = _Identity
    ModuleList = _ModuleList
    MSELoss = lambda: (lambda a, b: _Tensor(np.array(((a._arr - b._arr)**2).mean())))


class _TorchGeomNNNamespace:
    SAGEConv = _SAGEConv
    GATConv = _GATConv


@dataclass
class _PyGData:
    """torch_geometric.data.Data stub."""
    x: Optional[_Tensor] = None
    edge_index: Optional[_Tensor] = None
    edge_attr: Optional[_Tensor] = None

    @property
    def num_nodes(self) -> int:
        if self.x is not None:
            return self.x._arr.shape[0]
        return 0

    @property
    def num_edges(self) -> int:
        if self.edge_index is not None:
            return self.edge_index._arr.shape[1]
        return 0


class _TorchGeomDataNamespace:
    Data = _PyGData


class _TorchGeomNamespace:
    data = _TorchGeomDataNamespace()
    nn = _TorchGeomNNNamespace()


# ---------------------------------------------------------------------------
# snnTorch Leaky stub
# ---------------------------------------------------------------------------

class _Leaky(_TorchModule):
    """snnTorch Leaky stub using Euler integration."""

    def __init__(self, beta: float = 0.9, threshold: float = 1.0,
                 reset_mechanism: str = "subtract",
                 init_hidden: bool = False) -> None:
        super().__init__()
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.reset_mechanism = reset_mechanism

    def forward(self, input_current: _Tensor, mem: _Tensor) -> Tuple[_Tensor, _Tensor]:
        new_mem_arr = self.beta * mem._arr + input_current._arr
        spk_arr = (new_mem_arr >= self.threshold).astype(np.float32)
        if self.reset_mechanism == "subtract":
            new_mem_arr = new_mem_arr - spk_arr * self.threshold
        else:
            new_mem_arr = new_mem_arr * (1 - spk_arr)
        return _Tensor(spk_arr), _Tensor(new_mem_arr)


class _SnnTorchNamespace:
    Leaky = _Leaky


# ---------------------------------------------------------------------------
# gymnasium stub
# ---------------------------------------------------------------------------

class _Box:
    def __init__(self, low: Any, high: Any, shape: Tuple[int, ...],
                 dtype: Any = np.float32) -> None:
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def contains(self, x: np.ndarray) -> bool:
        return bool(np.all(x >= self.low) and np.all(x <= self.high))

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)


class _Dict:
    def __init__(self, spaces: Dict[str, Any]) -> None:
        self.spaces = spaces

    def contains(self, x: Dict[str, np.ndarray]) -> bool:
        return all(self.spaces[k].contains(x[k]) for k in self.spaces)


class _SpacesNamespace:
    Box = _Box
    Dict = _Dict


class _GymEnv:
    """gymnasium.Env stub."""
    observation_space: Any = None
    action_space: Any = None
    metadata: Dict[str, Any] = {}

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        raise NotImplementedError

    def render(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _GymnasiumNamespace:
    Env = _GymEnv
    spaces = _SpacesNamespace()


# ---------------------------------------------------------------------------
# pydantic_settings stub
# ---------------------------------------------------------------------------

class _BaseSettings:
    """pydantic_settings.BaseSettings stub using os.environ + dataclass."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    def __init__(self, **data: Any) -> None:
        # Gather defaults from class annotations
        hints = {}
        for klass in reversed(type(self).__mro__):
            hints.update(getattr(klass, "__annotations__", {}))

        prefix = "CELEGANS_"
        for name in hints:
            env_key = f"{prefix}{name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                # Coerce to the right type
                cur = getattr(self.__class__, name, None)
                try:
                    if isinstance(cur, bool):
                        val: Any = env_val.lower() in ("1", "true", "yes")
                    elif isinstance(cur, int):
                        val = int(env_val)
                    elif isinstance(cur, float):
                        val = float(env_val)
                    elif isinstance(cur, list):
                        import json
                        val = json.loads(env_val)
                    else:
                        val = env_val
                    setattr(self, name, val)
                except (ValueError, TypeError):
                    pass
            elif name in data:
                setattr(self, name, data[name])
            # else keep class default

        # Run validators manually (they're classmethods named like field_validator)
        # Skip for the stub — just ensure attributes exist
        for name in hints:
            if not hasattr(self, name):
                default = getattr(self.__class__, name, None)
                setattr(self, name, default)


class _SettingsConfigDict(dict):
    pass


class _PydanticSettingsNamespace:
    BaseSettings = _BaseSettings
    SettingsConfigDict = _SettingsConfigDict


# ---------------------------------------------------------------------------
# pydantic stub (field_validator, model_validator, BaseSettings)
# ---------------------------------------------------------------------------

def _field_validator(*fields: str, **kw: Any) -> Callable[[Any], Any]:
    def decorator(fn: Any) -> Any:
        return fn
    return decorator


def _model_validator(**kw: Any) -> Callable[[Any], Any]:
    def decorator(fn: Any) -> Any:
        return fn
    return decorator


class _PydanticNamespace:
    field_validator = staticmethod(_field_validator)
    model_validator = staticmethod(_model_validator)
    BaseModel = object


# ---------------------------------------------------------------------------
# torch top-level functions
# ---------------------------------------------------------------------------

class _TorchNamespace:
    Tensor = _Tensor
    nn = _TorchNNNamespace()
    float32 = _float32
    float = _float32
    long = _long
    bool = _bool_type
    int64 = _int64

    @staticmethod
    def tensor(data: Any, dtype: Any = None) -> _Tensor:
        return _Tensor(data, dtype=dtype)

    @staticmethod
    def zeros(*shape: Any, **kw: Any) -> _Tensor:
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return _Tensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def ones(*shape: Any, **kw: Any) -> _Tensor:
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return _Tensor(np.ones(shape, dtype=np.float32))

    @staticmethod
    def full(shape: Any, fill_value: float, **kw: Any) -> _Tensor:
        if isinstance(shape, int):
            shape = (shape,)
        return _Tensor(np.full(shape, fill_value, dtype=np.float32))

    @staticmethod
    def rand(*shape: Any) -> _Tensor:
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return _Tensor(np.random.rand(*shape).astype(np.float32))

    @staticmethod
    def randn(*shape: Any) -> _Tensor:
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return _Tensor(np.random.randn(*shape).astype(np.float32))

    @staticmethod
    def arange(n: int, dtype: Any = None) -> _Tensor:
        arr = np.arange(n)
        if dtype == _long or dtype == _int64:
            arr = arr.astype(np.int64)
        return _Tensor(arr)

    @staticmethod
    def stack(tensors: List[_Tensor], dim: int = 0) -> _Tensor:
        return _Tensor(np.stack([t._arr for t in tensors], axis=dim))

    @staticmethod
    def cat(tensors: List[_Tensor], dim: int = 0) -> _Tensor:
        return _Tensor(np.concatenate([t._arr for t in tensors], axis=dim))

    @staticmethod
    def where(condition: _Tensor) -> Tuple[_Tensor, ...]:
        result = np.where(condition._arr)
        return tuple(_Tensor(r) for r in result)

    @staticmethod
    def equal(a: _Tensor, b: _Tensor) -> bool:
        return bool(np.array_equal(a._arr, b._arr))

    @staticmethod
    def allclose(a: _Tensor, b: _Tensor, **kw: Any) -> bool:
        return bool(np.allclose(a._arr, b._arr, **kw))

    @staticmethod
    def isnan(t: _Tensor) -> _Tensor:
        return _Tensor(np.isnan(t._arr))

    @staticmethod
    def isinf(t: _Tensor) -> _Tensor:
        return _Tensor(np.isinf(t._arr))

    @staticmethod
    def no_grad() -> Any:
        return _NoGrad()

    @staticmethod
    def manual_seed(seed: int) -> None:
        np.random.seed(seed)

    @staticmethod
    def save(obj: Any, path: Any) -> None:
        pass  # stub

    @staticmethod
    def use_deterministic_algorithms(v: bool, warn_only: bool = False) -> None:
        pass

    # backends stub
    class backends:
        class cudnn:
            deterministic = False
            benchmark = True

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def manual_seed_all(seed: int) -> None:
            pass


class _NoGrad:
    def __enter__(self) -> "_NoGrad":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Expose as importable namespace objects
# ---------------------------------------------------------------------------

torch_stub = _TorchNamespace()
torch_geometric_stub = _TorchGeomNamespace()
snntorch_stub = _SnnTorchNamespace()
gymnasium_stub = _GymnasiumNamespace()
pydantic_settings_stub = _PydanticSettingsNamespace()
pydantic_stub = _PydanticNamespace()
