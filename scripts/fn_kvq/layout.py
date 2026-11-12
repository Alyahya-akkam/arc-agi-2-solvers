r"""The packed INT4 row format of the KV4 cache and its byte accounting (no torch).

One cache row holds one token of one KV head. For a head dimension $D$ and a group size $G$
($G \\mid D$, $G$ even) the row is $R = D/2 + 2D/G$ bytes:

- bytes $[0, D/2)$: the packed nibbles; byte $j$ holds element $2j$ in its low nibble and element
  $2j + 1$ in its high nibble, each as the unsigned value $u = q + 8$ of the signed 4-bit code
  $q \\in [-8, 7]$;
- bytes $[D/2, R)$: the $D/G$ FP16 scales, little-endian, scale $g$ at byte $D/2 + 2g$.

The quantizer is symmetric per group: $s_g = \\mathrm{fp16}(\\max_i |x_{g,i}| / 7)$ and
$q_{g,i} = \\mathrm{clip}(\\mathrm{round}(x_{g,i} / s_g), -8, 7)$ with round half to even, so that
$|x - s_g q| \\le s_g / 2$ up to the FP16 rounding of $s_g$. Symmetric was chosen over asymmetric
(a zero point per group costs 2 more bytes per group, 160 instead of 144 bytes per row at $G = 32$)
because the QSA keys and values are not skewed per token: K is RMS-normalized before RoPE and a
group of 32 channels of V mixes channels of both signs, so the extra level of resolution of an
asymmetric grid is worth under half a bit here while the zero point costs 11% of the row.

With the real shapes (1 KV head of 256 per GPU, 12 QSA layers, the BF16 indexer cache of 768 bytes
per token) the per-token cost is $12 \\times 2 \\times 144 + 768 = 4{,}224$ bytes against the FP8
path's $12 \\times 2 \\times 256 + 768 = 6{,}912$: the pool holds $1.636\\times$ the tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

# Group size along the head dimension: 32 channels share one FP16 scale (task P5.5's starting
# point; KIVI and QServe use 32 and 64). 64 would save 8 bytes per row (136) at a coarser scale.
DEFAULT_GROUP = 32

# The signed 4-bit grid [-8, 7]; the symmetric scale maps the group's absmax to +7.
QMIN = -8
QMAX = 7
QLEVELS = 7.0
# The unsigned offset of a nibble: u = q + 8.
NIBBLE_OFFSET = 8
# The Hadamard rotation's block: 64 channels (llama.cpp PR 21038's block-diagonal rotation), so
# a 256-wide head is 4 blocks and a group of 32 never straddles one.
HADAMARD_BLOCK = 64

# The environment variables of the patched engine (configs/P5.5-01.yaml, engine.env).
ENV_MODE = "ARC26_KV4"
ENV_HADAMARD = "ARC26_KV4_HADAMARD"
ENV_GROUP = "ARC26_KV4_GROUP"
# ARC26_KV4 values: "1" or "kv" quantize K and V; "v" quantizes V only (K stays FP8, the
# fallback the methods note names, 1.24x); unset or "0" boots the FP8 path unchanged.
MODES = ("off", "kv", "v")


def mode_from_env(environ: dict[str, str] | None = None) -> str:
    """The cache mode the environment selects: ``off``, ``kv`` or ``v``.

    Args:
        environ: A mapping to read (``os.environ`` when None).

    Returns:
        One of ``MODES``.

    Raises:
        ValueError: On an unknown value of ``ARC26_KV4``.
    """
    import os

    value = (os.environ if environ is None else environ).get(ENV_MODE, "0").strip().lower()
    if value in ("", "0", "off", "false"):
        return "off"
    if value in ("1", "kv", "true"):
        return "kv"
    if value == "v":
        return "v"
    raise ValueError(f"{ENV_MODE}={value!r}: expected 0, 1, kv or v")


def hadamard_from_env(environ: dict[str, str] | None = None) -> bool:
    """Whether the rotation is on: ``ARC26_KV4_HADAMARD`` unset or ``1`` means on, ``0`` off."""
    import os

    value = (os.environ if environ is None else environ).get(ENV_HADAMARD, "1").strip().lower()
    return value not in ("0", "off", "false")


def group_from_env(environ: dict[str, str] | None = None) -> int:
    """The group size: ``ARC26_KV4_GROUP`` or ``DEFAULT_GROUP``."""
    import os

    return int((os.environ if environ is None else environ).get(ENV_GROUP, DEFAULT_GROUP))


def row_bytes(head_dim: int, group: int = DEFAULT_GROUP) -> int:
    """Bytes of one packed row (one token, one KV head): nibbles plus FP16 scales.

    Args:
        head_dim: The head dimension $D$ (256 for Flash-Next's QSA layers).
        group: The quantization group size $G$ along the head dimension.

    Returns:
        $D / 2 + 2 D / G$.

    Raises:
        ValueError: If $G$ is odd, does not divide $D$, or $D$ is not even.
    """
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"head_dim must be a positive even number, got {head_dim}")
    if group <= 0 or group % 2 or head_dim % group:
        raise ValueError(f"group must be even and divide head_dim={head_dim}, got {group}")
    return head_dim // 2 + 2 * (head_dim // group)


def head_dim_from_row(row: int, group: int = DEFAULT_GROUP) -> int:
    """The head dimension whose packed row has ``row`` bytes: $D = 2 G R / (G + 4)$.

    Args:
        row: The row bytes $R$.
        group: The group size $G$.

    Returns:
        $D$.

    Raises:
        ValueError: If no head dimension gives ``row`` bytes at this group size.
    """
    numerator = 2 * group * row
    if numerator % (group + 4):
        raise ValueError(f"no head dimension packs to {row} bytes at group {group}")
    head_dim = numerator // (group + 4)
    if row_bytes(head_dim, group) != row:
        raise ValueError(f"no head dimension packs to {row} bytes at group {group}")
    return head_dim


def bytes_per_token(
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    v_head_dim: int | None = None,
    group: int = DEFAULT_GROUP,
    indexer_bytes: int = 0,
    mode: str = "kv",
    kv_itemsize: int = 1,
) -> int:
    """Per-token KV bytes per GPU, the fork's cell size with the edit applied.

    Args:
        layers: QSA (full-attention) layers on this GPU (12 for Flash-Next).
        kv_heads: KV heads per GPU after tensor parallelism (1 at TP 4).
        head_dim: K head dimension.
        v_head_dim: V head dimension; the K one when None.
        group: The group size.
        indexer_bytes: The unchanged indexer cache per token (768 for Flash-Next).
        mode: ``kv`` (both INT4), ``v`` (K stays in the engine's KV dtype), or ``off``.
        kv_itemsize: Bytes per element of the engine's KV dtype (1 for FP8, 2 for BF16).

    Returns:
        ``layers * kv_heads * (K row + V row) + indexer_bytes``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    v_dim = head_dim if v_head_dim is None else v_head_dim
    k_row = row_bytes(head_dim, group) if mode == "kv" else head_dim * kv_itemsize
    v_row = row_bytes(v_dim, group) if mode != "off" else v_dim * kv_itemsize
    return layers * kv_heads * (k_row + v_row) + indexer_bytes


@dataclass(frozen=True)
class KV4Layout:
    """The row layout for one head dimension and group size.

    Attributes:
        head_dim: $D$.
        group: $G$.
    """

    head_dim: int
    group: int = DEFAULT_GROUP

    def __post_init__(self) -> None:
        row_bytes(self.head_dim, self.group)

    @property
    def groups(self) -> int:
        """Scales per row, $D / G$."""
        return self.head_dim // self.group

    @property
    def packed_bytes(self) -> int:
        """Bytes of the nibble region, $D / 2$."""
        return self.head_dim // 2

    @property
    def scale_offset(self) -> int:
        """Byte offset of the first FP16 scale, $D / 2$."""
        return self.head_dim // 2

    @property
    def row_bytes(self) -> int:
        """Bytes of the whole row, $R$."""
        return row_bytes(self.head_dim, self.group)

    def scale_byte(self, g: int) -> int:
        """Byte offset of scale ``g`` (little-endian FP16, two bytes)."""
        if not 0 <= g < self.groups:
            raise ValueError(f"group index {g} outside [0, {self.groups})")
        return self.scale_offset + 2 * g

    def element_byte(self, i: int) -> tuple[int, int]:
        """Byte offset and nibble shift (0 low, 4 high) of element ``i``."""
        if not 0 <= i < self.head_dim:
            raise ValueError(f"element {i} outside [0, {self.head_dim})")
        return i // 2, 4 * (i % 2)
