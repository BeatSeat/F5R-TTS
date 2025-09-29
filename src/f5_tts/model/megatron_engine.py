from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


@dataclass
class _LoggerConfig:
    project_name: str
    init_kwargs: dict[str, Any]
    config: dict[str, Any]


class MegatronEngine:
    """Minimal Megatron-based replacement for the Accelerate `Accelerator` API."""

    def __init__(
        self,
        *,
        log_with: str | None = None,
        gradient_accumulation_steps: int = 1,
        megatron_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.log_with = log_with
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self._megatron_kwargs = megatron_kwargs or {}
        self._accum_step = 0
        self._sync_gradients = True
        self.even_batches = True
        self._logger_config: _LoggerConfig | None = None
        self._wandb_run = None

        self._init_distributed()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            return torch.device("cuda", index)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def num_processes(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    @property
    def process_index(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def local_process_index(self) -> int:
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            return int(local_rank)
        return self.process_index

    @property
    def is_main_process(self) -> bool:
        return self.process_index == 0

    @property
    def is_local_main_process(self) -> bool:
        return self.local_process_index == 0

    @property
    def sync_gradients(self) -> bool:
        return self._sync_gradients

    # ------------------------------------------------------------------
    # Public API compatible with Accelerate subset used by the project
    # ------------------------------------------------------------------
    def init_trackers(self, *, project_name: str, init_kwargs: dict | None = None, config: dict | None = None) -> None:
        if self.log_with != "wandb":
            return

        try:
            import wandb
        except ModuleNotFoundError:  # pragma: no cover - wandb optional at runtime
            return

        if not self.is_main_process:
            return

        init_kwargs = init_kwargs or {}
        config = config or {}
        self._logger_config = _LoggerConfig(project_name, init_kwargs, config)
        self._wandb_run = wandb.init(project=project_name, **init_kwargs.get("wandb", {}), config=config)

    def log(self, values: dict[str, Any], *, step: int | None = None) -> None:
        if self.log_with != "wandb" or not self.is_main_process or not values:
            return

        try:
            import wandb
        except ModuleNotFoundError:  # pragma: no cover
            return

        wandb.log(values, step=step)

    def end_training(self) -> None:
        if self.log_with != "wandb" or self._wandb_run is None:
            return

        if self.is_main_process:
            try:
                import wandb
            except ModuleNotFoundError:  # pragma: no cover
                return
            wandb.finish()
        self._wandb_run = None

    def prepare(self, *objects: Any) -> Any:
        prepared: list[Any] = []
        for obj in objects:
            if isinstance(obj, torch.nn.Module):
                prepared.append(self._prepare_module(obj))
            elif isinstance(obj, torch.optim.Optimizer):
                prepared.append(obj)
            elif isinstance(obj, DataLoader):
                prepared.append(self._prepare_dataloader(obj))
            else:
                prepared.append(obj)
        if len(prepared) == 1:
            return prepared[0]
        return tuple(prepared)

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model.module if isinstance(model, DistributedDataParallel) else model

    def wait_for_everyone(self) -> None:
        if dist.is_initialized():
            dist.barrier()

    def save(self, obj: Any, path: str) -> None:
        torch.save(obj, path)

    def clip_grad_norm_(self, parameters: Iterable[torch.nn.Parameter], max_norm: float) -> None:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def skip_first_batches(self, dataloader: DataLoader, *, num_batches: int) -> Iterable:
        iterator = iter(dataloader)
        for _ in range(num_batches):
            try:
                next(iterator)
            except StopIteration:
                return []
        return iterator

    @contextmanager
    def accumulate(self, model: torch.nn.Module):
        should_sync = (self._accum_step + 1) % self.gradient_accumulation_steps == 0
        self._sync_gradients = should_sync
        if should_sync or not hasattr(model, "no_sync"):
            context = nullcontext()
        else:
            context = model.no_sync()
        with context:
            yield
        self._accum_step = (self._accum_step + 1) % self.gradient_accumulation_steps
        self._sync_gradients = self.gradient_accumulation_steps == 1 or self._accum_step == 0

    @contextmanager
    def split_between_processes(self, items: Sequence[Any]):
        if self.num_processes == 1:
            yield items
            return

        subset = [item for idx, item in enumerate(items) if idx % self.num_processes == self.process_index]
        yield subset

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_distributed(self) -> None:
        if dist.is_initialized():
            return

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size <= 1:
            return

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, **self._megatron_kwargs)
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    def _prepare_module(self, module: torch.nn.Module) -> torch.nn.Module:
        module = module.to(self.device)
        if self.num_processes > 1:
            kwargs = {"device_ids": [self.device.index]} if self.device.type == "cuda" else {}
            module = DistributedDataParallel(module, broadcast_buffers=False, **kwargs)
        return module

    def _prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        if self.num_processes <= 1 or not hasattr(dataloader, "dataset"):
            return dataloader
        if isinstance(dataloader.sampler, DistributedSampler) or dataloader.batch_sampler is not None:
            return dataloader

        sampler = DistributedSampler(
            dataloader.dataset,
            num_replicas=self.num_processes,
            rank=self.process_index,
            shuffle=True,
        )
        return DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn,
            persistent_workers=dataloader.persistent_workers,
            prefetch_factor=getattr(dataloader, "prefetch_factor", None),
            pin_memory_device=getattr(dataloader, "pin_memory_device", ""),
        )


__all__ = ["MegatronEngine"]
