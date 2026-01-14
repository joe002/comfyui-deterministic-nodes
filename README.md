# ComfyUI Deterministic Nodes

Batch-invariant inference nodes for **guaranteed reproducibility** in ComfyUI.

## The Problem

**temperature=0 is NOT enough for determinism.**

The real culprit is **batch-size variance**. Same prompt, same seed, different batch sizes = different outputs.

## The Solution

These nodes enforce **batch_size=1** processing with fixed RNG states, guaranteeing:

```
Same seed + Same prompt = Identical output (ALWAYS)
```

Based on [ThinkingMachines batch-invariant-ops research](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).

## Nodes

### DeterministicSampler
Batch-invariant sampling with per-item RNG reset.
- Forces `batch_size=1` internally
- Resets RNG for each batch item: `seed + i`
- Disables cuDNN auto-tuning
- Outputs determinism proof with checksum

### ChecksumValidator
Verify reproducibility with configurable tolerance:
- `exact`: Byte-for-byte match
- `epsilon_1e-6`: Allow tiny floating-point variance
- `epsilon_1e-4`: Allow small floating-point variance
- `structural`: Shape and dtype match only

### MoERouterNode
Deterministic Mixture-of-Experts routing:
- Hash-based expert selection (no MCMC variance)
- Lexicographic tie-breaking
- Supports up to 4 expert models
- Same input = same expert (ALWAYS)

### CascadeRefiner
Sequential 3-stage refinement (Nemotron-Cascade pattern):
- Stage 1: Coarse pass
- Stage 2: Refinement
- Stage 3: Detail
- Different seed per stage for diversity within determinism

### ECHOContextNode
4-tier context memory retrieval (ECHO 2.0 pattern):
- Hot: GPU VRAM (active)
- Warm: System RAM (recent)
- Cold: NVMe (historical)
- Archive: Network (full)

## Installation

Copy to ComfyUI custom_nodes:
```
ComfyUI/custom_nodes/ComfyUI-DeterministicNodes/
```

Restart ComfyUI.

## Usage

### Reproducibility Workflow

```
[Model] --> [DeterministicSampler] --> [ChecksumValidator] --> [Output]
                   |                            |
                   +-- seed=42 ----------------+
                   |                            |
                   +-- checksum output --------+-- verify on next run
```

### Multi-Model Workflow

```
[Prompt] --> [MoERouterNode] --> [Selected Model] --> [DeterministicSampler]
                  |
                  +-- expert_0: General
                  +-- expert_1: Code
                  +-- expert_2: Domain
                  +-- expert_3: Math
```

### Context-Aware Workflow

```
[Query] --> [ECHOContextNode] --> [Prompt with Context] --> [Model]
                  |
                  +-- hot_only: Fast, GPU cache
                  +-- hot_warm: Recent context
                  +-- all_tiers: Full history
```

## Technical Details

### Why Batch-Invariance Matters

```
Batch=1:  "The answer is 42"
Batch=4:  "The answer is 41"  <-- DIFFERENT!
Batch=8:  "The answer is 43"  <-- DIFFERENT!
```

GPU parallel operations have floating-point accumulation order variance.
Different batch sizes = different accumulation order = different results.

### The Fix

1. Process items one at a time (`batch_size=1`)
2. Reset RNG before each item (`seed + item_index`)
3. Disable cuDNN auto-tuning (`cudnn.benchmark=False`)
4. Use deterministic algorithms (`torch.use_deterministic_algorithms(True)`)

## Framework Integration

These nodes integrate with:
- **ECHO 2.0**: 4-tier context memory (ECHOContextNode)
- **CSQMF-R1**: Expert routing (MoERouterNode)
- **Nemotron**: Cascade refinement (CascadeRefiner)
- **ThinkingMachines**: Batch-invariant inference (DeterministicSampler)

## License

Apache License 2.0 - See [LICENSE](LICENSE)

---

*Determinism is not optional. It's a requirement.*
