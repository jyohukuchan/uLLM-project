# Importance-score GPU resume: fallback safety stop

## Scope

- Resumed from git HEAD `74731f3d85a8babc8fb921d6b9c1ab8796e92fc4`.
- Did not start or stop `ullm-openai.service`.
- Every PyTorch GPU command used `HIP_VISIBLE_DEVICES=1`, with `CUDA_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` unset.
- No additional model, corpus, Python package, or other download was performed.

## Preflight

- `/usr/bin/python3` had CPU-only `torch 2.12.0+cpu`, so it was not used for GPU work.
- Reused the existing uv cache copy of `torch 2.11.0+gitd0c8b1f`, built for HIP `7.2.53211` / ROCm `7.2.3`.
- PyTorch reported CUDA/HIP available, exactly one visible logical device, `gcnArchName=gfx1201`, and 31.859 GiB total memory. The driver returned the generic product name `AMD Radeon Graphics`; `gfx1201` verified the intended RDNA4 R9700 selected by physical HIP index 1.
- In this ROCm build, the collector device argument is `cuda` (logical `cuda:0` after masking).

## Qwen D_stats smoke

The smoke used the formal Qwen source, formal D_stats shard 00, sequence length 128, batch size 1, two requested samples, full module matching, BF16, `python3 -u`, and `--progress-every-batches 1`.

Before any sample forward completed, stderr emitted:

```text
[transformers] The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d
```

The job was interrupted immediately during `model.to(cuda)` in accordance with the required fallback stop rule. It exited with status 130 after 20.56 seconds. Completed samples: 0. No safetensors or metadata output was written, so no per-sample timing or full-run estimate can be computed.

## Stop state

- No activation collector, block covariance collector, or perturbation process remained.
- A final masked check reported 31.791 GiB free of 31.859 GiB on the single visible `gfx1201` device.
- Qwen C0 through C6, Qwen report/candidate freeze, Gemma evaluation, and the worst-model admission decision were not started.
- Gemma label data was not opened or joined; the lockbox remains intact.
- GPU processing is fully stopped. Production service restoration remains with the external operator as requested.

## Human decision required

Decide whether to install/build and validate ROCm-compatible `flash-linear-attention` and `causal-conv1d`, or explicitly authorize the GPU PyTorch fallback despite the warning. A future formal GPU continuation also needs GPU-compatible provenance/merge handling because the current formal manifest and activation merge validation encode CPU-only execution.
