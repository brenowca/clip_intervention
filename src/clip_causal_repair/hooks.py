from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerAddress:
    """Simple address pointing to a submodule by its dotted path."""

    name: str


def resolve_modules(model: torch.nn.Module, addresses: Iterable[LayerAddress]) -> Dict[LayerAddress, torch.nn.Module]:
    """Resolve a sequence of :class:`LayerAddress` objects to modules.

    Parameters
    ----------
    model:
        The root module to search within.
    addresses:
        Iterable of addresses describing submodules by their dotted names.

    Returns
    -------
    dict
        Mapping from each address to the resolved ``nn.Module`` instance.
    """
    module_dict = dict(model.named_modules())
    resolved: Dict[LayerAddress, torch.nn.Module] = {}
    for addr in addresses:
        mod = module_dict.get(addr.name)
        if mod is None:
            raise KeyError(f"No submodule named {addr.name!r} in model")
        resolved[addr] = mod
    return resolved


class HookManager:
    """Utility to register and later remove forward hooks on modules.

    Example
    -------
    >>> mgr = HookManager(model)
    >>> mgr.register([LayerAddress("encoder.layer1")], hook_fn)
    >>> ...  # run model
    >>> mgr.remove()
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def register(self, addresses: Sequence[LayerAddress], hook: Callable):
        """Register ``hook`` on all modules indicated by ``addresses``.

        ``addresses`` may be a single :class:`LayerAddress` or a sequence of
        them. The resolved modules are obtained via :func:`resolve_modules`.
        """
        if isinstance(addresses, LayerAddress):
            addresses = [addresses]
        modules = resolve_modules(self.model, addresses)
        for addr, module in modules.items():
            handle = module.register_forward_hook(hook)
            logger.debug("Registered hook on %s", addr.name)
            self._handles.append(handle)

    def remove(self):
        """Remove all registered hooks."""
        while self._handles:
            handle = self._handles.pop()
            handle.remove()
            logger.debug("Removed hook")

    # Context-manager helpers
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()


def tensor_edit_projection_out(
    t: torch.Tensor,
    u: torch.Tensor,
    alpha: float = 1.0,
    token_scope: Optional[object] = None,
) -> torch.Tensor:
    """Project ``t`` away from direction ``u``.

    Parameters
    ----------
    t:
        Activation tensor of shape ``[B, d]`` or ``[B, T, d]``.
    u:
        Direction vector of shape ``[d]``.
    alpha:
        Scale of projection removal.
    token_scope:
        For 3D tensors, specifies which token positions to edit. Accepted
        values are ``None``/"all" for all tokens, "last" for only the last
        token, or an integer index/slice.

    Returns
    -------
    torch.Tensor
        Edited tensor. If the edit is skipped due to safety checks the
        original tensor is returned unchanged.
    """
    if t.shape[-1] != u.shape[-1]:
        logger.warning(
            "tensor_edit_projection_out: mismatched dims t=%s u=%s", t.shape, u.shape
        )
        return t

    u_norm = u.norm()
    if torch.isnan(u_norm) or u_norm < 1e-6:
        logger.warning("tensor_edit_projection_out: tiny edit direction; skipping")
        return t

    u_dir = u / u_norm

    if t.ndim == 2:
        v = t
        proj_coeff = v @ u_dir
        edited = v - alpha * proj_coeff.unsqueeze(-1) * u_dir
        cos_before = F.cosine_similarity(v, u_dir.unsqueeze(0), dim=-1)
        cos_after = F.cosine_similarity(edited, u_dir.unsqueeze(0), dim=-1)
        logger.debug(
            "projection_out: cos %.4f->%.4f | t_norm %.4f | u_norm %.4f",
            cos_before.mean().item(),
            cos_after.mean().item(),
            v.norm(dim=-1).mean().item(),
            u_norm.item(),
        )
        return edited

    if t.ndim == 3:
        if token_scope in (None, "all"):
            idx = slice(None)
        elif token_scope == "last":
            idx = slice(-1, None)
        elif isinstance(token_scope, int):
            idx = slice(token_scope, token_scope + 1)
        elif isinstance(token_scope, slice):
            idx = token_scope
        else:
            logger.warning(
                "tensor_edit_projection_out: unsupported token_scope %r", token_scope
            )
            return t

        subset = t[:, idx, :]
        proj_coeff = torch.matmul(subset, u_dir)
        while proj_coeff.ndim < subset.ndim:
            proj_coeff = proj_coeff.unsqueeze(-1)
        edited = subset - alpha * proj_coeff * u_dir
        cos_before = F.cosine_similarity(subset, u_dir, dim=-1)
        cos_after = F.cosine_similarity(edited, u_dir, dim=-1)
        logger.debug(
            "projection_out: cos %.4f->%.4f | t_norm %.4f | u_norm %.4f",
            cos_before.mean().item(),
            cos_after.mean().item(),
            subset.norm(dim=-1).mean().item(),
            u_norm.item(),
        )
        out = t.clone()
        out[:, idx, :] = edited
        return out

    logger.warning(
        "tensor_edit_projection_out: unsupported tensor rank %d", t.ndim
    )
    return t
