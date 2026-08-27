"""
utils/config.py
================
ConfGate v6 — LoRA + Per-Layer Hard-Gated Skip Connections.

Architecture:
    For each transformer block (n_layers depends on backbone):
        router_i   : Linear(hidden, 1) → sigmoid → STE → hard gate gᵢ ∈ {0,1}
        if gᵢ = 1  : h = TransformerBlock_i(h)   with LoRA adapters
        if gᵢ = 0  : h = h                        skip entire block

    Trainable:
        - n_layers router linears  (~21K params at 24 layers)
        - LoRA adapters on Q,K,V,O projections  (~3.6M params for r=8)

    Frozen:
        - All original backbone weights

Training:
    Single pass, single loss:
        loss = CrossEntropy(lm_logits, next_token_labels)
    Data:
        Hermes / Glaive  → tool_call sequences (contain <tool_call> token)
        GSM8K / Turing   → planning / reasoning sequences
    The model learns which blocks to skip via the natural LM signal.
    No binary classification head. No two-stage training.

Straight-Through Estimator (STE):
    Forward : hard {0,1} gate
    Backward: gradient flows through sigmoid as if it were continuous
    Standard trick for discrete gating — used in MoD, VQ-VAE, etc.

Qwen2.5-0.5B specs:
    num_hidden_layers  : 24
    hidden_size        : 896
    num_q_heads        : 14
    num_kv_heads       : 2
    intermediate_size  : 4864
    vocab_size         : 151936

TinyLlama-1.1B specs:
    num_hidden_layers  : 22
    hidden_size        : 2048
    num_q_heads        : 32
    num_kv_heads       : 4
    intermediate_size  : 5632
    vocab_size         : 32000

NOTE on middle-layer boundary (see layerroute_errata_note.pdf):
    RouterConfig.middle_start/middle_end define the biased-init boundary
    (Eq. 7 in the paper). These were originally silent internal defaults
    in router.py (middle_start=8, middle_end=17), hardcoded for Qwen2.5's
    24-layer depth and reused unexamined at other depths. They are now
    explicit RouterConfig fields. Use derive_middle_bounds(n_layers) to
    compute a depth-proportional boundary instead of reusing 8/17 verbatim
    on a backbone with a different layer count.
"""

from dataclasses import dataclass, field


QWEN_SPEC = {
    "hf_name"     : "Qwen/Qwen2.5-1.5B-Instruct",
    "n_layers"    : 28,
    "n_q_heads"   : 12,
    "n_kv_heads"  : 2,
    "hidden"      : 1536,
    "head_dim"    : 128,
    "intermediate": 8960,
    "vocab"       : 151936,
}

LLAMA_SPEC = {
    "hf_name"     : "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "n_layers"    : 22,
    "n_q_heads"   : 32,
    "n_kv_heads"  : 4,
    "hidden"      : 2048,
    "head_dim"    : 64,
    "intermediate": 5632,
    "vocab"       : 32000,
}


def derive_middle_bounds(n_layers: int, qwen_ref_layers: int = 24,
                         qwen_start: int = 8, qwen_end: int = 17):
    """
    Preserve the SAME fractional depth used at Qwen2.5-0.5B's 24 layers
    (middle band ~33%-71% of depth) for any n_layers, instead of reusing
    the same absolute layer indices across different architectures.

    Returns (middle_start, middle_end) as integer layer indices.
    See layerroute_errata_note.pdf for why this matters.
    """
    frac_start = qwen_start / qwen_ref_layers
    frac_end   = qwen_end / qwen_ref_layers
    return round(frac_start * n_layers), round(frac_end * n_layers)


@dataclass
class LoRAConfig:
    r           : int   = 8         # LoRA rank
    alpha       : float = 16.0      # LoRA scaling = alpha / r
    dropout     : float = 0.05
    # Which projections to apply LoRA to
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass
class RouterConfig:
    # Per-layer router: Linear(hidden, 1) → sigmoid → STE
    # init_bias is per-layer: early/late layers start open (1.0),
    # middle layers start closed (-1.0) to break symmetry immediately
    init_bias_early : float = 1.0    # layers outside [middle_start, middle_end): sigmoid(1.0)=0.73
    init_bias_middle: float = -1.0   # layers [middle_start, middle_end): sigmoid(-1.0)=0.27 — below threshold
    threshold       : float = 0.5    # hard gate threshold
    # Gate regularisation: loss += gate_reg_weight * mean(soft_gates)
    # Penalises uniformly high gates — forces router to find skippable layers
    gate_reg_weight : float = 1.0    # increased from 0.05 — more aggressive skipping
    # Middle-layer boundary for biased init (Eq. 7). Defaults match the
    # original Qwen2.5-0.5B (24-layer) design. For any other backbone,
    # compute these via derive_middle_bounds(n_layers) rather than reusing
    # these defaults verbatim -- see module docstring / errata note.
    middle_start    : int = 8
    middle_end      : int = 17


@dataclass
class TrainingConfig:
    lr              : float = 2e-4   # standard LoRA fine-tuning LR
    batch_size      : int   = 4
    grad_accum      : int   = 4      # effective batch = 16
    max_steps       : int   = 1000
    warmup_steps    : int   = 100
    grad_clip       : float = 1.0
    log_every       : int   = 50
    eval_every      : int   = 200
    save_every      : int   = 500
    output_dir      : str   = "./checkpoints"
    dtype           : str   = "bfloat16"
    max_seq_len     : int   = 512


@dataclass
class DataConfig:
    max_train_samples : int  = 5000   # per dataset
    val_split         : float = 0.1
    local_jsonl       : str  = None


@dataclass
class LayerDropConfig:
    rate                 : float = 0.2    # train-time per-layer stochastic drop probability
    inference_skip_ratio : float = 0.25   # fixed, input-independent inference-time skip fraction


@dataclass
class MoDConfig:
    capacity: float = 0.75   # target average fraction of tokens routed through each layer
    gate_reg_weight: float = 1.0  # regularization pressure pushing avg gate toward capacity


@dataclass
class SystemConfig:
    lora      : LoRAConfig      = field(default_factory=LoRAConfig)
    router    : RouterConfig    = field(default_factory=RouterConfig)
    training  : TrainingConfig  = field(default_factory=TrainingConfig)
    data      : DataConfig      = field(default_factory=DataConfig)
    layerdrop : LayerDropConfig = field(default_factory=LayerDropConfig)
    mod       : MoDConfig       = field(default_factory=MoDConfig)
    model_family: str = "qwen"    # "qwen" or "llama" -- selects BACKBONE/tokenizer spec
    method      : str = "layerroute"  # "layerroute", "layerdrop", or "mod" -- selects GATING mechanism
                                       # (independent of model_family; layerdrop/mod currently qwen-only,
                                       # matching this audit's core 2-task x 2-scale matrix)