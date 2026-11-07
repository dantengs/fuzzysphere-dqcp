# Credits

| Contributor | Contribution |
|---|---|
| Claude Code (Anthropic) | Tcc: `src/`, `bin/`, `bench/`, `test/`, `tools/`, `configs/`, `docs/`; with Codex, the analyses in `analysis/` |
| Codex (OpenAI) | parts of the KK, OPE and tuning stages (`src/kkstage.jl`, `src/kk_projection.jl`, `src/kk_generator.jl`, `src/ope.jl`, `src/tuning.jl`, `src/so5_c2_density_phase.jl`); with Claude Code, the analyses in `analysis/` |
| FuzzifiED (Zheng Zhou, MIT licence) | the fuzzy-sphere library Tcc is built on, and the SO(5) DQCP example the model construction follows; one file replaced, `deps/FuzzifiED-2.0.1/` |

The papers the model, the KK projection and the analyses build on are listed under *Acknowledgements* in
[README.md](README.md).

Add yourself and your collaborators here when you use this template.

中文：Claude Code 编写 Tcc；Codex 编写了 KK、OPE 与 tuning 部分代码；`analysis/` 中的分析由两者共同完成。FuzzifiED 由周正开发、
MIT 许可，Tcc 以它为基础，模型构造参考其 SO(5) DQCP 示例（替换了一个文件）。所依据的文献见 [README.md](README.md) 的致谢部分。
使用本模板时请在此添加你和合作者。
