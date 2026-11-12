# Credits

All code in this repository was developed by two AI models, ChatGPT (OpenAI) and Claude Code (Anthropic).

| Contributor | Contribution |
|---|---|
| ChatGPT (OpenAI), including its Codex coding agent | the earlier production code from which the KK, OPE and tuning stages were ported (`src/kkstage.jl`, `src/kk_projection.jl`, `src/kk_generator.jl`, `src/ope.jl`, `src/tuning.jl`, `src/so5_c2_density_phase.jl`); with Claude Code, the analyses in `analysis/` |
| Claude Code (Anthropic) | Tcc: `src/`, `bin/`, `bench/`, `test/`, `tools/`, `configs/`, `cluster/`, `deps/`, `docs/`; with ChatGPT, the analyses in `analysis/` |

## Code we learned from

| Code | What we used |
|---|---|
| [FuzzifiED](https://github.com/FuzzifiED/FuzzifiED.jl) by Zheng Zhou and the FuzzifiED.jl contributors (MIT licence) | the fuzzy-sphere library Tcc is built on; its SO(5) DQCP example `examples/so3lver/so5_dqcp_spectrum.jl`, which the model construction follows. One file of FuzzifiED 2.0.1 is replaced, see `deps/FuzzifiED-2.0.1/` |

We learned from the open-source code of Zheng Zhou and collaborators while developing Tcc and thank them for making
it available. The papers that the model, the KK projection and the analyses build on are listed under
*Acknowledgements* in [README.md](README.md).

Add yourself and your collaborators here when you build on this code.

---

中文：

本仓库的全部代码由两个 AI 模型开发：ChatGPT（OpenAI）和 Claude Code（Anthropic）。

- **ChatGPT（包括其编程助手 Codex）**：编写了早期的生产代码，Tcc 的 KK、OPE 与 tuning 部分就是从这些代码移植来的；与 Claude Code
  共同完成 `analysis/` 中的分析。
- **Claude Code**：编写了 Tcc 的其余部分（`src/`、`bin/`、`bench/`、`test/`、`tools/`、`configs/`、`cluster/`、`deps/`、`docs/`）。

开发 Tcc 时，我们学习了周正与 FuzzifiED 贡献者的开源代码：
- Tcc 以 FuzzifiED 为基础；
- 模型构造参考其 SO(5) DQCP 示例；
- FuzzifiED 2.0.1 中替换了一个文件，见 `deps/FuzzifiED-2.0.1/`。

感谢他们开源这些代码。模型、KK projection 与分析所依据的文献见 [README.md](README.md) 的致谢部分。在本代码基础上开展工作时，
请在此添加你和合作者。
