# Dependencies

```bash
julia deps/setup_env.jl          # once; then: julia --project=env --threads=8 bin/tcc.jl <run.toml>
```

Packages are installed into the first entry of `JULIA_DEPOT_PATH` (default `~/.julia`). For a self-contained
installation (one depot, offline; can be copied to a cluster without internet) source `cluster/environment.sh` first:
see `cluster/environment.example.sh` and docs/en/DATA_LAYOUT.md §6. `env/Manifest.toml` refers to `deps/FuzzifiED.jl` by a
relative path, so the repository with its depot can be moved.

| Package | Version | As used in production |
|---|---|---|
| Julia | 1.11.7 | |
| FuzzifiED | 2.0.1 (General registry, git tree `c2b1d162…`) | **one file replaced**: `src/so3lver/core/comp_operator.jl` → `FuzzifiED-2.0.1/comp_operator.jl` (diff: `comp_operator.patch`). Multi-threaded `BuildCompOperator` assembly with a `num_th` keyword and a lock-free, per-destination-channel H·v. Mathematically the same operator; only the floating-point summation order differs. Tcc passes `num_th` to `BuildCompOperator` everywhere, so the unmodified release does not run Tcc. FuzzifiED is MIT-licensed (`FuzzifiED-2.0.1/LICENSE`). |
| KrylovKit | 0.10.4 (registry, git tree `a3babd26…`) | the production copy also contains `LockedLanczos` (`src/eigsolve/lockedlanczos.jl` and its export); Tcc never calls it, so the registered release is used |
| FuzzifiED_jll | 1.3.0+0 | from the registry |

`production_sources.toml` holds the SHA-256 of every source file of both packages as used by all production runs
(the cluster copies, identical to the copies of the local tests; compared on 2026-11-02). `setup_env.jl` checks the
environment it builds against it and stops on any other difference.

Note: earlier documents (METHODS ≤ 2026-11-02 morning) called FuzzifiED "unmodified"; the comparison with the registered
release on 2026-11-02 found the replaced file above.

中文：`julia deps/setup_env.jl` 一次即可建好环境。包装进 `JULIA_DEPOT_PATH` 的第一项（默认 `~/.julia`）。要做只用一个 depot、
离线运行、可以拷到不联网集群的自包含安装，先 source `cluster/environment.sh`（见 `cluster/environment.example.sh` 与
docs/zh/DATA_LAYOUT.md §6）；`env/Manifest.toml` 用相对路径指向 `deps/FuzzifiED.jl`，仓库连同 depot 可以整体搬动。FuzzifiED 2.0.1 用 registry 版本，但替换了 `comp_operator.jl` 一个文件
（多线程组装、带 `num_th` 关键字、无锁的 H·v；算符相同，只是求和顺序不同）；Tcc 处处向 `BuildCompOperator` 传 `num_th`，
未修改的版本跑不了 Tcc。KrylovKit 0.10.4 的生产副本多了 Tcc 不用的 `LockedLanczos`，所以直接用 registry 版本。
`setup_env.jl` 会逐文件核对生产时所用源码的 SHA-256。此前文档称 FuzzifiED "未修改"，2026-11-02 与 registry 版本比对后更正。
