# RXNGraphormer Model Engineering Plan

目标：在兼容既有预训练权重的基础上，将当前偏学术脚本式的模型训练、评估和推理代码重构为可扩展、可复现、可维护的 Lightning 工程实现，并系统性提升训练和推理效率。

## 总体原则

1. 旧权重兼容优先：重构过程中保留核心模型 `state_dict` 的语义和可映射性，任何结构调整都必须提供 legacy checkpoint adapter。
2. 先包外壳，再改内核：第一阶段 Lightning 只作为训练、评估、日志、checkpoint 的工程外壳，避免一次性改动模型数学实现。
3. 指标口径统一：训练内验证、最终测试、发布权重评估、回归测试必须使用同一 evaluator。
4. 数据协议显式化：所有 split、target、标准化、clamp、预处理 schema、特征版本都必须进入可持久化 manifest。
5. 性能优化可回退：AMP、compile、bucketing、多 worker、复杂图距离替代方案都要支持开关，便于定位数值或性能回归。
6. 模型定义与工程流程解耦：模型结构、graph layer、checkpoint key 语义属于 model core；训练、评估、推理、预处理和 CLI 只通过稳定 factory/API 使用模型 core，不能继续与模型实现平铺混杂。

## 源码结构目标

仍然只发布一个 `rxngraphormer` 包，但包内需要形成清晰的工程边界。目标结构如下：

```text
rxngraphormer/
  models/          # 纯 torch 模型定义、graph layers、model factory、state_dict 语义
  training/        # LightningModule、DataModule、trainer/workflow、callbacks
  inference/       # Predictor、batch/table/reaction inputs、exporters
  evaluation/      # task metrics、report writers、standalone evaluators
  data/            # dataset protocol、split manifest、dataloaders、collate
  preprocessing/   # rxn_smiles/materialization、mapping、graph artifact generation
  compatibility/   # legacy config/checkpoint/import-path adapters
  cli/             # thin command entry points only
```

`models/` 内部也不能再回到单文件堆叠，应按实现职责继续细化：

```text
rxngraphormer/models/
  layers/          # PyG graph conv、attention block、dynamic message passing
  heads.py         # regression/classification MLP heads
  transformer.py   # shared transformer encoder and pooling helpers
  attention_xl.py  # legacy/experimental relative attention encoder
  encoders.py      # RXN graph encoder and graph-to-sequence encoder wrapper
  sequence.py      # sequence generation decoder/beam-search model
  tasks.py         # RXNGRegressor/RXNGClassifier/RXNGraphormer task assembly
  factory.py       # config-to-model construction API
```

`training/` 内部也应区分新工程路径和 legacy 回退路径：

```text
rxngraphormer/training/
  legacy.py        # old SPLIT*/Sequence trainers kept only for fallback compatibility
  lightning.py     # compatibility export for Lightning workflow
```

平铺 wrapper 只允许作为短期迁移工具存在；API 迁移完成后必须删除对应文件，不能在平铺代码
上继续打补丁。当前根层平铺 wrapper 窗口已经关闭，包内调用、测试和文档必须使用
`models/`、`training/`、`inference/`、`evaluation/`、`data/`、`preprocessing/`、`compatibility/`
等目标边界。旧用户代码应按 release notes 迁移到替代 API。

## 非破坏性兼容边界

以下行为在完成 Lightning 工程化前应保持稳定：

- `RXNGRegressor`、`RXNGEncoder` 和主要 graph layer 的参数含义不变。
- legacy 权重可以加载到裸 `torch.nn.Module`。
- legacy 权重也可以通过 adapter 加载到 `LightningModule.model`。
- 新生成的预处理缓存和推理/评估权重默认使用 `.safetensors`；旧 `.pt/.pth/.ckpt` 仅作为兼容输入和训练恢复产物保留。
- 旧 JSON/YAML/argparse 风格配置可以转换到 dataclass/TOML 配置。
- 同一份数据和 split manifest 下，legacy trainer、Lightning trainer、standalone evaluator 的指标应在容忍范围内一致。

## 阶段计划表

| 阶段 | 重构目标 | 具体任务 | 验收标准 | 主要风险 |
| --- | --- | --- | --- | --- |
| P0. 基线冻结 | 固化当前可接受的复现状态，防止后续重构失去参照 | 整理 BH OOS、C_H_func、发布权重评估结果；保存使用的 config、split、checkpoint、环境信息；记录当前吞吐、显存、CPU/GPU 利用率；确认 TensorBoard/log 路径约定 | 生成一份 baseline manifest；至少包含数据版本、权重路径、指标、命令、环境、git diff 摘要；后续任意阶段可一键复跑核心 smoke 测试 | 当前已有随机 split 不完全协议化，需要标记为 legacy seed split，不能作为强等价协议 |
| P1. 数据协议与配置边界 | 将模型、预处理和共享协议代码统一在主包内，依赖通过 extras 分层 | 将 dataset schema、split manifest、target schema、normalizer schema、feature schema 放入 `rxngraphormer.data_protocol`；全部配置改为 dataclass；标准配置格式为 TOML；保留 legacy config reader；定义 train/eval/predict/preprocess 配置层级 | 低 torch 预处理环境安装主包 `preprocess` extra 可生成协议数据；高 torch Blackwell 环境只安装主包默认依赖可训练和推理；legacy 配置可无损转换为 TOML | 配置字段过多时容易把隐式默认值遗漏，需要对照旧脚本逐项迁移 |
| P2. Legacy 权重兼容层 | 把旧权重加载变成稳定 API，而不是临时脚本逻辑 | 实现 `CheckpointAdapter`；支持 `.pt/.pth/.ckpt/.safetensors`；支持去除/补充 `model.` 前缀；记录 missing/unexpected keys；提供 strict、shape-compatible、diagnostic 三种加载模式；支持保存 canonical state_dict | 旧发布权重可加载到裸 `RXNGRegressor` 和 Lightning wrapper；加载报告清晰列出 key 映射；BH OOS 发布权重评估与当前脚本一致 | 如果后续模型模块改名，必须维护 key mapping 表；不能依赖 Lightning checkpoint 的内部格式作为唯一格式 |
| P3. Lightning 外壳迁移 | 用 Lightning 接管训练循环，但不改模型数学实现 | 新增 `RXNGraphormerLitModule`；封装 `training_step`、`validation_step`、`test_step`、`predict_step`；迁移 optimizer/scheduler 配置；修复 gradient accumulation 语义；加入 checkpoint callback、logger、resume；保留裸模型 forward | 相同 config 和 split 下，Lightning 单卡训练指标与 legacy trainer 在统计波动内一致；legacy 权重可进入 Lightning test/predict；支持 resume from checkpoint | Lightning 的默认行为可能改变梯度累计、日志频率、设备放置，需要显式配置 |
| P4. 统一 Evaluator | 消除训练、评估、推理之间的指标口径漂移 | 建立独立 evaluator；统一 MAE/RMSE/R2、target unscale、yield clamp、prediction export；训练中默认只按 validation 选 checkpoint；test 只在最终或指定频率运行；预测聚合改为 list/CPU 累积 | 同一 checkpoint 用 train 内 validation、CLI eval、standalone evaluator 得到一致指标；不再出现训练集/测试集指标混淆；大数据评估无重复 `torch.cat` 开销 | 旧报告中不同脚本可能使用不同 clamp/target 口径，需要显式标注 legacy 模式 |
| P5. DataModule 与数据加载性能 | 降低 CPU 和 Python overhead，提升 GPU 饱和度 | 新增 Lightning `DataModule`；DataLoader 支持 `num_workers`、`pin_memory`、`persistent_workers`、`prefetch_factor`；实现 split manifest 加载；支持按图规模 bucketing；将 `pad_feat`、`update_batch_idx` 等 Python 循环路径逐步向向量化迁移 | 单卡训练吞吐相对 P0 有明确提升；GPU idle 时间下降；相同 batch 下显存变化可解释；DataLoader 参数可在 TOML 中配置 | 多 worker 下 RDKit/PyG 对象序列化和缓存路径可能暴露兼容问题 |
| P6. 复杂度爆炸路径治理 | 隔离或替换 dense graph distance 和 max-node padding | 将 `attentionxl` 标记为 legacy/experimental；给 `calc_batch_graph_distance` 加尺寸保护、profile 和警告；评估替代方案：预处理阶段缓存 shortest path、sparse distance、bucketed dense attention、PyG/SDPA mask；默认回归路径禁用高复杂度分支 | 常规回归训练不会无意进入 `O(N_total^2)` 或 `O(n^3)` 路径；大图 batch 超限时明确报错或自动降级；复杂路径有独立 benchmark | 如果旧分类/序列任务依赖 `attentionxl`，需要保留 legacy 兼容模式 |
| P7. 训练效率提升 | 面向 Blackwell 优化训练吞吐和稳定性 | 默认启用 BF16/AMP 可选策略；修复 dropout eval 行为；加入 deterministic 开关；支持 gradient clipping、scheduler、early stopping、accumulate grad；评估 `torch.compile` 对核心 forward 的收益；支持 DDP/单机多卡；记录每 epoch 时间和显存峰值 | BH OOS 和至少一个其他数据集在指标不退化前提下训练更快；支持双卡训练或并发单卡任务；训练日志含吞吐、显存、学习率、metric 曲线 | `torch.compile` 对动态图/PyG 可能收益不稳定，需要保留关闭开关 |
| P8. 推理效率与部署接口 | 提供稳定、快速、低依赖的推理入口 | 实现 `Predictor` API；支持裸模型和 Lightning checkpoint；批量 CSV/Parquet 输入；导出 predictions、embeddings、uncertainty placeholder；支持 safetensors；避免推理强依赖 Trainer；评估 batch size、AMP、CPU/GPU 数据搬运优化 | 发布权重可直接用于 CLI 和 Python API 推理；推理结果与 evaluator 一致；Blackwell GPU 上批量推理吞吐有 profile 报告 | 推理输入格式必须与预处理协议严格绑定，否则容易产生隐式特征差异 |
| P9. 回归测试矩阵 | 将复现结果变成持续可执行的质量门 | 建立 smoke、medium、full 三档测试；覆盖 legacy weight load、BH OOS、C_H_func、随机 seed split、config conversion、DataModule dataloader；记录容忍阈值；新增性能基准输出 | 每个 PR/阶段至少跑 smoke；重要改动跑 medium；发布前跑 full；测试报告输出到 `reports/` | full 训练耗时较长，需要用已有权重和小 epoch 检查覆盖快速路径 |
| P10. 发布与迁移文档 | 让单包发布、extras 分层和用户迁移路径清晰 | 更新 README；补充 `docs/environments.md`；新增 legacy-to-toml 配置迁移指南；新增 checkpoint compatibility 文档；定义版本号策略和 deprecation 策略；给出单环境端到端和分离式预处理/训练/推理命令 | 用户可以按文档用一个包完成单环境或双环境流程；旧权重使用方式明确；默认依赖、`preprocess` extra、`sequence` extra 边界清晰 | 文档如果不绑定真实命令和测试，很快会与代码漂移 |
| P11. 预训练分类器 Lightning 迁移 | 将分类预训练链路纳入与回归一致的工程外壳 | 扩展 LightningModule/DataModule 支持 `task="classification"`；迁移分类 loss/accuracy/optimizer/scheduler；默认分类训练入口切到 Lightning；保留 legacy 分类回退；验证预训练分类权重可加载到裸模型和 Lightning wrapper；确认回归 fine-tune 初始化不受影响 | 发布的预训练分类权重兼容；分类 forward/predict 与 legacy 一致；分类训练 smoke 可写 manifest/checkpoint/report；回归 fine-tune 使用预训练权重的指标不退化 | 分类权重是回归 fine-tune 的基础资产，任何 state_dict 或 encoder 行为变化都可能影响下游回归 |
| P12. 源码结构收敛 | 将模型定义/实现与训练、推理、评估框架解耦 | 建立 `models/`、`training/`、`inference/`、`evaluation/`、`data/`、`preprocessing/`、`compatibility/` 边界；删除完成迁移的根层平铺 wrapper；集中 temporary CSV/materialization helper；把 CLI 变成薄入口 | 模型 core 可在不导入 Lightning/CLI 的情况下构建和 forward；训练和推理仅依赖 model factory/checkpoint adapter；旧根层 wrapper 文件不存在且源码无旧路径 import；全量 smoke 通过 | 模块移动容易破坏旧权重 key、pickle/import path、用户脚本，需要 release notes 明确替代 API |
| P13. 序列化现代化 | 将数据缓存和推理权重默认迁移到 safetensors | 数据集预处理默认产出/消费 `.safetensors`；`mol_index` 等图字段统一为 tensor，并通过专用 `ReactionGraphData` 明确 PyG batching 语义，禁止默认 `index` 字段偏移；权重加载优先 `.safetensors` 并回退 `.pt/.pth/.ckpt`；legacy trainer 保存完整 `.pt` 恢复检查点并同步导出 `.safetensors` 模型权重；Lightning 训练后导出 canonical `model/valid_checkpoint.safetensors`；提供 `scripts/convert_pt_to_safetensors.py` | 新训练/预处理产物默认不再依赖 pickle；旧 `.pt` 数据缓存和权重仍可读；转换脚本可处理 checkpoint 与 processed-data；smoke/单测覆盖 safetensors roundtrip 和 fallback | safetensors 不适合完整 optimizer/scheduler 恢复状态，训练 resume 仍需 `.pt/.ckpt`；PyG 对字段名含 `index` 的 tensor 有默认递增规则，必须保留专用数据容器测试 |
| P14. 类型系统收敛 | 让源码逐步通过 ruff、mypy、pyright，形成可维护的类型边界 | 在 `pyproject.toml` 和 `pyrightconfig.json` 中维护严格 typed surface 清单；优先迁移 data/config/serialization/runtime/model layer 基础模块；避免新增 `Any`，动态库边界用 `object`、`Protocol`、`TypeGuard` 和必要 `cast` 收敛；每完成一批模块即加入默认检查清单 | 默认 `ruff check`、`mypy`、`pyright` 对 typed surface 通过；未迁移模块不能作为静默 wrapper 留在平铺结构中，而应按目标边界逐步细化后纳入 typed surface；最终清单覆盖全包 | RDKit、PyG、OpenNMT、safetensors 等库的 stub 不完整，边界处需要显式隔离动态类型，避免把 unknown/Any 扩散到模型核心 |

## 当前阶段成果

截至当前阶段，工程化改造已经从迁移期进入系统测试期。项目已经收敛为 `src` 布局下的单个 `rxngraphormer` Python 包，根层平铺 wrapper 窗口已经关闭；回归和预训练分类器都已进入 Lightning/DataModule/evaluator/manifest 工作流；模型、数据、预处理、训练、推理、评估、兼容和运行时设备管理边界已经固定。新生成的数据缓存和发布权重默认优先使用 `.safetensors`，旧 `.pt/.pth/.ckpt` 作为兼容输入和训练恢复格式保留。当前重点不再是继续给平铺代码打补丁，而是补齐完整功能测试、指标复现和发布门禁。

### 已完成的工程面

- 源码已经切换到现代 `src/` 布局，并按工程边界拆分；临时文件、数据文件、训练产物等已纳入 `.gitignore` 管理。
- 包结构已确定为单包发布：
  - 默认依赖面向模型训练、评估和推理。
  - `preprocess` extra 包含 RDKit、atom mapping、DGL 等预处理依赖。
  - `all` extra 支持兼容硬件上的单环境端到端流程。
  - `sequence` extra 保留 sequence generation 依赖边界。
- 自动部署脚本已支持单环境和分离式环境，用户命令统一通过路由脚本执行：
  - `scripts/rxngraphormer_pipeline.sh --layout single <command> [args...]` 使用一个环境执行预处理、训练和评估。
  - `scripts/rxngraphormer_pipeline.sh --layout split <command> [args...]` 使用预处理环境生成数据，模型环境训练/推理。
  - 自动模式按 GPU compute capability 选择布局。
- 配置读取已通过 `rxngraphormer.config` 兼容 legacy JSON，并引入 typed dataclass 配置层。
- 表格数据协议已补齐：
  - `data.input_table` 支持 CSV/Parquet。
  - 标准 `rxn_smiles` 输入可直接用于端到端训练预处理。
  - 旧 `rct_smiles`/`pdt_smiles` 分离列继续兼容。
  - 标准 `rct>agt>pdt` 会按原 RXNGraphormer 约定将 agent 同时复制到 reactant/product 两侧。
  - 显式 atom mapping 和显式 mapped H 会被保留；已验证 `dataset/full_df.xlsx` 生成结果与旧 CSV canonical 一致。
- legacy 权重兼容层已落地：
  - `CheckpointAdapter` 支持 legacy `.pt`、Lightning `.ckpt`、`.safetensors`。
  - 支持 bare model 与 Lightning wrapper 的 `model.` 前缀适配。
  - 已验证 BH 发布权重 bare/Lightning 双路径均可完整加载。
- 序列化默认已切换：
  - 数据集预处理默认写入/读取 `.safetensors`，并自动回退旧 `.pt` 缓存。
  - `mol_index` 已从 Python list 迁移为 `torch.long` tensor，新缓存可直接以 safetensors 保存；通过 `ReactionGraphData` 覆盖 PyG `__inc__`/`__cat_dim__`，避免 batch 时被误当成 edge index 递增。
  - 推理/评估默认 checkpoint 文件名为 `valid_checkpoint.safetensors`，旧 `.pt/.pth/.ckpt` 自动 fallback。
  - legacy trainer 保存完整 `.pt` 恢复检查点，同时导出相邻 `.safetensors` 模型权重。
- Lightning fit 后导出 canonical `model/valid_checkpoint.safetensors`，manifest 的 `checkpoint.best_model_path` 指向该权重，`trainer_best_model_path` 保留 Lightning `.ckpt` 路径。
- 新增 `scripts/convert_pt_to_safetensors.py` 支持 checkpoint 和 processed-data 转换。
- 类型检查覆盖已扩展到完整 `src/`：
  - `ruff check src` 通过。
  - `mypy src` 通过，覆盖当前 88 个源码文件。
  - `pyright` 通过，`0 errors, 0 warnings`。
  - RDKit、PyG、safetensors 等动态库边界通过局部 `Protocol`、`TypeGuard` 和必要 `cast` 收敛，没有再通过排除 legacy 模块维持通过。
- 回归训练已迁移到标准 Lightning 工作流：
  - `rxngraphormer.lightning.module`
  - `rxngraphormer.lightning.datamodule`
  - `rxngraphormer.lightning.trainer`
  - `rxngraphormer.lightning.workflow`
  - `scripts/rxngraphormer_pipeline.sh train ...` 对 `task="regression"` 默认走 Lightning。
  - `scripts/rxngraphormer_pipeline.sh train-legacy ...` 保留旧训练路径。
- 回归评估/推理已统一：
  - `RXNGraphormerPredictor` 支持 legacy 权重和 Lightning checkpoint。
  - `RXNGraphormerPredictor.predict_table()` 支持标准 `rxn_smiles` 表格和旧分离列。
  - `reaction_prediction()`、`RXNEMB.gen_rxn_emb()`、`RXNClassifier.rxn_pred()` 已改用统一 reaction parser，而不是字符串 `>>` 切分。
  - `rxngraphormer.evaluation.regression_workflow` 统一 split 构建、checkpoint 加载、metrics、JSON/CSV 报告。
  - `eval_regression_performance()` 保持旧 API，但内部委托新 workflow。
  - `scripts/rxngraphormer_pipeline.sh eval ...` 对 regression 默认走新 evaluator。
- 实验产物标准化已完成：
  - 每次 Lightning fit 默认写 `manifest.json`。
  - 支持 `--eval_after_fit`，训练后自动用 best checkpoint 生成 `reports/eval.json` 和 `reports/eval.csv`。
  - 标准目录为：
    - `<root>/<tag>/version_<n>/checkpoints/`
    - `<root>/<tag>/version_<n>/reports/`
    - `<root>/<tag>/version_<n>/manifest.json`
    - `<root>/manifest.json`

### 已完成的验证

- 拆分后的 smoke 单测已覆盖核心兼容路径，目前为 `107 tests OK`。
- `scripts/reproduce/run_smoke_tests.py --skip-finetune` 已通过：
  - compile 检查通过。
  - algorithm-invariants 通过。
  - legacy BH 权重兼容检查通过。
  - classification 发布权重 bare/Lightning 兼容检查通过。
  - regression finetune 阶段按命令参数跳过。
- legacy BH 权重兼容检查通过：
  - bare model: `445/445` keys loaded
  - Lightning wrapper: `445/445` keys loaded
  - missing/unexpected/shape mismatch 均为 0
- `dataset/full_df.xlsx` 的 `rxn_smiles` 已用新 parser 全量处理并与旧 CSV 对比：
  - row count: `4578 / 4578 / 4578`
  - target max abs diff: `0.0`
  - new parser rct canonical match old CSV: `4578 / 4578`
  - new parser pdt canonical match old CSV: `4578 / 4578`
  - map-set differences: `0`
  - old CSV mapping issues: `0`
- 两组标准 OOS 回归复现已完成：
  - Br_OOS: Lightning 600 epoch 约 2.00h，legacy 约 5.20h，指标接近。
  - I_OOS: Lightning 600 epoch 约 1.92h，legacy 约 5.75h，指标接近。
  - 说明训练管线优化有效，且模型算法未被改坏。
- 新推理管线已对当前重训、上次重训、文献发布权重做过对比，结果已写入：
  - `reports/oos_predictor_weight_comparison.csv`
  - `reports/oos_predictor_weight_comparison.json`
- manifest/自动评估 smoke 已通过，产物示例：
  - `experiments/smoke_manifest/smoke/version_0/manifest.json`
  - `experiments/smoke_manifest/smoke/version_0/reports/eval.json`
  - `experiments/smoke_manifest/smoke/version_0/reports/eval.csv`

### 当前兼容边界

- 回归训练、评估、推理已进入新工作流；legacy regression trainer 仅作为显式回退路径保留。
- 分类预训练训练、评估、manifest 和 checkpoint 兼容 smoke 已进入 Lightning 工作流；legacy classification trainer 仅通过显式回退开关保留。
- 表格输入、Python API 和 CLI 推理已支持标准 `rxn_smiles` 与旧分离列两种协议。
- 传统 PyTorch 训练、legacy eval、standalone inference 和 compatibility smoke 由 `DeviceManager` 统一设备管理；Lightning 路径由 Lightning Trainer 接管设备放置，两者共用同一套 `rxngraphormer.models` 模型定义。
- sequence generation 仍保留 legacy 路径。该路径依赖 OpenNMT、vocab、beam search 和旧 decoder 语义，后续只做兼容性测试和边界隔离，优先级低于回归/分类发布链路。
- 预训练分类器仍是回归 fine-tune 的基础权重来源；任何后续重构必须继续覆盖 state_dict、encoder forward 和 fine-tune 初始化兼容。
- 现在已将模型前向语义拆成显式 `forward_compat_mode={modern,legacy}`，用于对照原始实现的 padding mask、readout 和 dropout 行为；后续复现与消融都必须标注该模式。

## P11/P12 工程化完成记录

P11/P12 已经完成主体实现。该阶段没有重写分类模型算法，而是把分类预训练纳入与回归一致的 Lightning 工程外壳，并将模型定义/实现从训练、推理、评估和 CLI 中解耦出来。下面保留该阶段目标和验收项，作为后续回归测试的依据。

### 已固化目标边界

- 保持 `RXNGClassifier` 裸模型结构和参数语义不变。
- 保持已发布预训练分类权重可加载。
- 保持回归 fine-tune 通过 `pretrained_model_path` 初始化 encoder 的行为不变。
- 保持用户侧训练入口稳定，新增回退开关而不是删除 legacy 路径。
- 保持标准 `rxn_smiles` 与旧 `rct/pdt` 分离输入两种协议在训练、评估和推理路径上的等价语义。
- 继续维持单包发布和 extras 分层，不拆分 package。
- 平铺模块不再作为迁移期 API 兼容层保留；包内调用方必须使用 `models/`、`training/`、`inference/`、`evaluation/`、`compatibility/` 等目标边界。
- `rxngraphormer.model_factory`、`rxngraphormer.predictor`、`rxngraphormer.evaluator`、`rxngraphormer.checkpointing`、`rxngraphormer.classification_workflow`、`rxngraphormer.regression_workflow` 等平铺 import path 的兼容窗口已关闭，替代 API 记录在对应边界包中。

### 已落地任务

1. 源码边界已经落地：
   - `rxngraphormer.models` 作为模型 core，承载模型结构、graph layer、head、encoder、task assembly 和 factory。
   - `rxngraphormer.lightning` 承载 LightningModule、DataModule、trainer/workflow 和 callbacks。
   - `rxngraphormer.training` 承载 legacy trainer 回退、scheduler、diagnostics 和训练共享工具。
   - `rxngraphormer.inference` 承载 Predictor、table/reaction input adapter、embedding/classifier API 和 exporters。
   - `rxngraphormer.evaluation` 承载 metrics、report writer、standalone evaluator 和 task workflow。
   - `rxngraphormer.compatibility` 承载 legacy checkpoint/config/import adapters。
   - 当前平铺 wrapper 已完成删除；后续新增功能不得恢复根层 wrapper 文件。
   - 每迁移一个平铺模块，都必须补充自动化断言：新边界 API 指向真实实现，旧平铺路径文件不存在且源码无旧路径 import。
   - 新代码、包内代码和持续维护脚本禁止新增旧平铺 import；旧 import path 只允许出现在历史文档或迁移说明中。
   - wrapper 删除验收条件：包内搜索无旧平铺 import、文档已改用新 API、smoke 覆盖新 API、release notes 已给出替代路径。

2. 收敛重复胶水逻辑：
   - 将 reaction table materialization、temporary CSV 构建、side canonicalization 的重复逻辑集中到共享 helper。
   - 训练、推理和 embedding API 复用同一套 reaction/table input adapter。
   - CLI 只解析参数并调用 training/inference/evaluation workflow。

3. 扩展 LightningModule 支持分类任务：
   - `training_step` 支持分类 loss。
   - `validation_step`/`test_step` 记录 loss、accuracy、confidence 等分类指标。
   - `predict_step` 输出分类 logits/probabilities/prediction。
   - 保持 `self.model` 的 state_dict namespace 与裸模型兼容。

4. 新增或扩展分类 DataModule：
   - 对齐 `SPLITClassifierTrainer` 的数据输入方式。
   - 先支持现有预训练分类数据格式。
   - 支持通过 `data.input_table` 预先 materialize 后的 rct/pdt legacy CSV。
   - 支持 `batch_size`、`num_workers`、`pin_memory`、`persistent_workers`。
   - 继续支持 split manifest。

5. 迁移分类训练入口：
   - `rxngraphormer-train` 对 `task="classification"` 默认走 Lightning。
   - 增加 `--legacy_classification` 作为回退。
   - `sequence_generation` 继续 legacy。

6. 权重兼容验证：
   - 发布的预训练分类权重可加载到裸 `RXNGClassifier`。
   - 同一权重可加载到 Lightning wrapper。
   - 同一 batch 下 bare 与 Lightning forward 输出一致。
   - 回归 fine-tune 使用该预训练权重时，加载报告无异常，指标无系统性退化。

7. 分类评估/推理标准化：
   - 统一 accuracy/loss/confidence/probability 的报告口径。
   - 支持分类 `reports/eval.json` / `reports/eval.csv`。
   - 将分类训练 smoke 纳入 manifest。
   - 确认 classification `RXNGraphormerPredictor`、Lightning `test_step`、CLI eval 使用同一 evaluator。

### P11/P12 回归验收标准

- `scripts/rxngraphormer_pipeline.sh train --config <classification_config>` 默认使用 Lightning。
- `--legacy_classification` 可回退旧分类 trainer。
- 分类预训练 smoke 可完成训练、写 checkpoint、写 manifest。
- 模型 core 可独立 import/build/forward，不导入 Lightning、CLI 或 Predictor。
- `rxngraphormer.model`、`rxngraphormer.layer`、`rxngraphormer.predictor` 等旧 import path 已从源码树移除，用户代码应迁移到新边界。
- 发布预训练分类权重 bare/Lightning 双路径加载成功。
- 分类 Predictor 输出与 legacy 路径一致或在浮点容忍范围内一致。
- 标准 `rxn_smiles` 表格和旧 `rct_smiles`/`pdt_smiles` 表格在分类训练/推理路径上均可运行。
- 至少一次回归 fine-tune 使用新路径/旧预训练权重完成 smoke，确认下游不受影响。

## 关键技术任务清单

### 权重兼容

- 定义 canonical state dict：裸 `RXNGRegressor.state_dict()` 为内部标准。
- Lightning checkpoint 只作为训练恢复格式，不作为唯一发布格式。
- 发布模型建议同时提供：
  - `model.safetensors`
  - `model_config.toml`
  - `data_protocol.toml`
  - `split_manifest.json`
  - `normalizer.json`
  - `metrics.json`
- `CheckpointAdapter` 应支持以下 key 处理：
  - remove prefix: `model.`
  - add prefix: `model.`
  - legacy module rename mapping
  - shape mismatch diagnostics
  - missing/unexpected key report

### 配置系统

- 配置类全部使用 dataclass。
- 可选枚举字段使用 `Literal[...]` 明确约束。
- 标准格式为 TOML。
- legacy config reader 只做兼容读取，不作为新配置写出格式。
- 配置层级建议：
  - `ProjectConfig`
  - `DataConfig`
  - `SplitConfig`
  - `ModelConfig`
  - `TrainConfig`
  - `OptimizerConfig`
  - `SchedulerConfig`
  - `EvalConfig`
  - `RuntimeConfig`
  - `CheckpointConfig`

### Lightning 模块边界

- `RXNGRegressor` 保持为纯 `torch.nn.Module`。
- `RXNGraphormerLitModule` 只拥有：
  - `self.model`
  - loss
  - metric aggregator
  - optimizer/scheduler factory
  - train/val/test/predict step
- `RXNGraphormerDataModule` 只拥有：
  - dataset protocol reader
  - split manifest reader
  - dataloader factory
  - batch/collate strategy
- Evaluator 必须能独立于 Lightning 使用。

### 设备管理边界

- 模型定义不得做全局设备选择；模型内部只能跟随输入张量或已有参数/buffer 的 device。
- 传统 PyTorch 训练、legacy eval、standalone inference 和 compatibility smoke 通过 `rxngraphormer.runtime.DeviceManager` 统一解析 device、迁移模型和递归迁移 batch。
- Lightning 训练管线不使用 `DeviceManager` 选择设备；batch/model 放置由 Lightning Trainer 管理，`RXNGraphormerLitModule` 只使用 Lightning 注入的 `self.device` 调用共享 batch helper。
- 传统训练和 Lightning 必须共用同一套 `rxngraphormer.models` 模型定义；差异只存在于训练框架外壳。
- 设备管理验收必须覆盖快速训练和推理：传统路径检查 batch 被 `DeviceManager` 移动、loss 可 backward、eval 推理不产生梯度；Lightning 路径检查 `training_step` 可 backward 且不引入手动设备解析。

### 性能治理重点

- 立即修复：
  - eval 中重复 `torch.cat`
  - gradient accumulation 语义
  - dropout eval 行为
  - DataLoader worker/pin memory 配置
- 中期优化：
  - vectorize `pad_feat`
  - vectorize 或预计算 `update_batch_idx`
  - bucketing 减少 padding
  - test 频率从每 epoch 改为最终或低频
- 高风险优化：
  - 替换 `calc_batch_graph_distance`
  - `torch.compile`
  - DDP/FSDP
  - graph feature cache

## 验收指标

### 功能兼容

- 所有已发布 BH OOS 权重可加载并完成评估。
- 至少一个 C_H_func 权重或训练流程可完成回归。
- legacy config 可转换为 TOML，并且转换后指标一致。
- Lightning checkpoint 可恢复训练并继续记录日志。

### 数值一致性

- 同一权重、同一数据、同一 split 下，legacy evaluator 与新 evaluator 的预测逐样本一致或在浮点容忍范围内一致。
- 重训指标允许随机波动，但不得出现系统性退化。
- 修复 dropout/accumulation 等 bug 后，如果指标变化，需要单独记录原因和影响范围。
- 复现对照必须至少包含 `modern` 和 `legacy` 两条前向语义分支，不能只在现代实现上调参；必要时再叠加 batch size 和 warmup scaling 的消融。

### 复现差距诊断

- 当前已确认的主要训练轴不止前向语义，还包括 batch size、warmup scaling 和 precision。
- 已观察到：`legacy + batch1280 + bf16` 完整 Br_OOS 训练的 best TensorBoard `val_mae` 约 `8.03`，最终 evaluator `valid MAE 8.21`、`R2 0.815`，与 `modern + batch1280 + bf16` 的 `valid MAE 7.98` 同级，但仍显著高于公开 reproduction 参考的 `5.934` 和论文原始报告的 `5.810`。
- 已观察到：`legacy + batch32 + no warmup scaling` 的 20 epoch 短跑中，`fp32` 明显优于 `bf16`，说明训练稳定性对 precision 更敏感，而不是只由 forward 语义决定。
- 当前 `legacy + batch32 + no warmup scaling + fp32` Lightning full run 已达到 best TensorBoard `val_mae ~= 6.19`，明显优于 batch1280/bf16 轴，但仍高于公开 reproduction/paper 的 `5.93/5.81`。
- 同条件 `modern + batch32 + no warmup scaling + fp32` 目前 best TensorBoard `val_mae ~= 6.63`，仍在继续收敛；因此不能用中途曲线过早判定 legacy/modern 最终优劣，必须等两个 600 epoch run 完整结束后再定性。
- 与 `5a25ab5ae512cbd5525470e06963e12aa0ddde35` 对照后，当前模型 core 在 legacy 模式下已经还原原始关键路径：Transformer 无 padding mask、padded sequence raw mean、graph dropout 在 eval 中保持旧版 always-on 行为；`mol_index` tensor 化通过 `ReactionGraphData` 禁止 PyG 自动 index 偏移，并在 `update_batch_idx` 中按 atom `batch` 切分重建，语义应与旧 list 路径一致。
- 剩余差距优先按训练协议定位，而不是继续猜模型结构：固定/扫描初始化 seed、current legacy trainer vs Lightning trainer loop、validation checkpoint 保存口径、DataLoader shuffle 随机性、gradient clipping/optimizer step 时序。
- 已新增 `scripts/reproduce/train_legacy_oos_probe.py`，用于用当前 legacy trainer 跑同一套 OOS 配置，并自动记录 manifest、checkpoint、TensorBoard `log/` 和可选 evaluator 报告；它用于隔离 Lightning 训练循环与模型实现差异。
- `current legacy trainer + legacy + batch32 + no warmup scaling + fp32 + 40 epoch` 短跑已经跑通，验证了 legacy trainer、`.pt/.safetensors` 双 checkpoint 和 post-fit evaluator 报告；由于 40 epoch 仍在 warmup 初段且曲线波动大，best evaluator `valid MAE ~= 18.65` 只作为链路 smoke，不作为性能复现结论。
- 当前完整复现主轴应优先采用 `legacy + batch32 + no warmup scaling + fp32`，再与 `modern` 分支做同条件对照，避免只在现代实现上调参。
- 公开 reproduction 参考：`MJ-Zeng/RXNGraphormer-Reproduction` 使用原始预训练权重和原始 `train_model.py` 路径跑 OOS 案例；当前工程化复现必须按同一 OOS 案例逐项对比，不再依赖随机 seed 划分口径。
- 复现诊断已脚本化为 `scripts/reproduce/summarize_oos_reproduction.py`，默认将 evaluator MAE 乘以 100 后与论文/公开 reproduction 表格比较，并写入 `reports/oos_reproduction_diagnostics.{json,csv}`；脚本同时支持 Lightning `val_mae` 和 legacy trainer `log/valid_mae` TensorBoard 标量。

### 性能目标

- 单卡训练 epoch time 相对 P0 明确下降。
- 评估阶段显存峰值下降，避免随样本数重复 concat 增长。
- DataLoader 不再成为明显单点瓶颈。
- Blackwell 上 BF16/AMP 路径可稳定运行。
- 推理 API 支持大批量输入，吞吐和显存有基准记录。

## 推荐执行顺序

1. P0 + P2：先冻结指标并完成权重 adapter，保证重构有回退锚点。
2. P3 + P4：引入 Lightning 外壳和统一 evaluator，不改模型数学。
3. P5 + P7：修训练性能和正确性问题。
4. P6：治理复杂度爆炸路径，避免未来扩展踩坑。
5. P8 + P9 + P10：完善推理、测试矩阵和发布文档。
6. P9 复现矩阵当前应优先等待 `legacy + batch32 + no warmup scaling + fp32` 与 `modern + batch32 + no warmup scaling + fp32` 的完整 Br_OOS 结果，同时跑短程 legacy trainer loop 对照；若 legacy full run 仍停在 `~6.4`，下一轮应做 seed scan 和 current legacy trainer full run，而不是继续放大 batch size。

## 第一批建议落地任务

早期落地任务已经被后续边界迁移吸收，当前实现不再新增根层平铺模块：

1. checkpoint adapter 已归入 `compatibility` 边界。
2. Lightning module、DataModule、trainer/workflow 和 callbacks 已归入 `lightning`/`training` 边界。
3. evaluator、classification/regression workflow 和 legacy eval helper 已归入 `evaluation` 边界。
4. dataloader、dataset、collate、split、graph preprocessing helper 已归入 `data` 边界。
5. reaction/table materialization 已归入 `preprocessing` 边界。
6. validation/test prediction 聚合、dropout/accumulation 等训练正确性修复已纳入 smoke。
7. legacy weight compatibility smoke 保留在 `scripts/reproduce`。

## P11/P12 推进记录

- 工程边界已收敛到目标包结构：
  - `models` 承载模型 core、graph layers、heads、transformer、encoder、task assembly、model factory 和模型内部 tensor/attention 小算子。
  - `training` 承载 legacy trainer 回退、shared scheduler 和 Lightning 训练边界导出。
  - `lightning` 承载 LightningModule、DataModule、trainer/workflow 和 callbacks。
  - `inference` 承载 Predictor、embedding/classifier API、batch/table/reaction inputs 和 exporter。
  - `evaluation` 承载 classification/regression metrics、workflow、report writer 和 legacy eval helper。
  - `data` 承载 dataset、collate、splits、graph construction、feature、batch tensor helper 和 dataloader settings。
  - `preprocessing` 承载 reaction parser、chemistry helper、table materialization 和 temporary input generation。
  - `compatibility` 承载 checkpoint adapter、legacy torch load helper 和 checkpoint compatibility smoke API。
- 根层平铺 wrapper 已删除；后续新增功能不得恢复这些 wrapper 文件。
- `utils.py` 已删除；原有日志、norm、LR、seed、sequence accuracy 和 state_dict key helper 分别迁移到 `training/diagnostics.py`、`evaluation/sequence_metrics.py`、`compatibility/state_dict.py`，不再保留根层平铺工具文件。
- `preprocess` 旧包已删除；预处理 CLI/workflow 迁移到 `preprocessing/workflow.py`，`rxngraphormer-preprocess` 入口直接调用 `rxngraphormer.preprocessing` 边界。
- `lightning`、`tqdm`、`safetensors` 已作为必需依赖直接导入；源码不再为这些基础依赖保留 ImportError fallback。
- 已新增 P12 import-boundary 自动化断言：
  - 包内代码、新维护脚本和新测试禁止直接 import 已删除的根层平铺路径。
  - 断言对应根层 wrapper 文件不存在，避免后续重构重新引入旧结构。
  - 断言新边界 API 与具体实现模块保持同一对象或等价行为。
- `RXNGraphormerLitModule` 已支持 `task="classification"`：
  - `training_step` 使用 `model.logits()` + `CrossEntropyLoss`。
  - `validation_step`/`test_step` 聚合 logits/targets 并记录 loss、accuracy、confidence。
  - `predict_step` 输出 logits、probabilities、prediction、confidence。
  - `torch.compile` 分类路径编译 `model.logits`，避免将 softmax probability 误当 logits。
- `RXNGraphormerDataModule` 已支持分类预训练数据：
  - 支持 `MultiRXNDataset` 的 `rct_name_regrex`/`pdt_name_regrex` 格式。
  - 支持 materialized train/val/test CSV 格式。
  - 分类路径默认只使用 pair collate，不进入 regression mid graph 分支。
- `rxngraphormer-train` 对 `task="classification"` 默认走 Lightning。
  - `--legacy_classification` 保留旧 `SPLITClassifierTrainer` 回退路径。
  - `sequence_generation` 仍保留 legacy 路径。
- 已新增分类 evaluator/workflow：
  - `classification_metrics`
  - `evaluate_classification`
  - `classification_workflow.py`
  - `rxngraphormer-eval` 支持 classification JSON/CSV 报告。
  - `--eval_after_fit` 对 classification 写 `reports/eval.json` 和 `reports/eval.csv`。
- 已新增分类 checkpoint 兼容 smoke：
  - `rxngraphormer.compatibility.check_classification_checkpoint_compatibility`
  - `rxngraphormer-compat classification-checkpoint`
  - 检查发布分类 checkpoint 可严格加载到 bare `RXNGClassifier` 和 `RXNGraphormerLitModule.model`。
  - 可选 batch 检查 bare/Lightning logits/probabilities 一致。

### 已完成 P11/P12 验证

- 单测 smoke：`python -m unittest discover -s tests -p 'test_*.py'`
  - 当前通过：`107 tests OK`。
- P12 边界迁移 smoke：
  - `scripts/reproduce/run_smoke_tests.py` 的 compile 门禁已覆盖新实现文件，包括 `rxngraphormer/evaluation/classification_workflow.py`、`rxngraphormer/evaluation/legacy_eval.py`、`rxngraphormer/evaluation/regression_workflow.py`，以及 `rxngraphormer/models/*` 和 `rxngraphormer/models/layers/*`。
  - compile 门禁也已覆盖 `rxngraphormer/training/legacy.py`，根层 legacy train wrapper 已删除。
  - compile 门禁也已覆盖 `rxngraphormer/preprocessing/chemistry.py`，避免化学 helper 再回落到平铺 `utils.py` 实现。
  - compile 门禁也已覆盖 `rxngraphormer/preprocessing/workflow.py`，根层 legacy `preprocess` 包已删除。
  - compile 门禁也已覆盖 `rxngraphormer/data/batch.py`、`rxngraphormer/models/ops.py`、`rxngraphormer/config_utils.py`，避免 batch/model/config helper 再回落到平铺 `utils.py` 实现。
  - compile 门禁也已覆盖 `rxngraphormer/training/schedulers.py`，根层 legacy scheduler wrapper 已删除。
  - compile 门禁也已覆盖 `rxngraphormer/inference/embeddings.py`，根层 legacy rxn_emb wrapper 已删除。
  - compile 门禁也已覆盖 `rxngraphormer/runtime/{__init__,device}.py`，防止设备管理重新分散到训练、推理和评估入口。
  - compile 门禁也已覆盖 `rxngraphormer/data/__init__.py`、`rxngraphormer/data/legacy.py`、`rxngraphormer/data/datasets.py`、`rxngraphormer/data/loader.py`、`rxngraphormer/data/{constants,files,graph,reaction_graph,tokenization,collate,splits}.py`、`rxngraphormer/data/{sequence_dataset,reaction_dataset,multi_reaction_dataset,pairing}.py`、`scripts/reproduce/{check_legacy_weight_compat,evaluate_bh_matrix}.py`、`scripts/full_df/{evaluate_legacy_train_ratios_extrapolation,evaluate_legacy_train_ratios_full_space,predict_full_space,prepare_baseline_train_ratio_data,prepare_rxngraphormer_full_df,summarize_train_ratio_sample_spaces}.py`，根层 legacy dataloader wrapper 已删除。
  - `rxngraphormer.models.*` 已承载模型层真实实现；根层 `rxngraphormer.model`、`rxngraphormer.layer`、`rxngraphormer.model_factory` 已删除。
  - `rxngraphormer.training.legacy` 承载 legacy trainer 回退；根层 `rxngraphormer.train` 已删除。
  - `rxngraphormer.data` 承载 dataset/loader API；根层 `rxngraphormer.dataloader`、`rxngraphormer.datasets`、`rxngraphormer.datamodule` 已删除。
  - targeted model smoke 已确认 `GCNConv` forward 可运行，`build_regression_model()` 构建出的 state_dict key 仍保持 `rct_encoder...` 等旧权重语义。
  - `rxngraphormer.evaluation` 承载 evaluator/workflow API；根层 `rxngraphormer.evaluator`、`rxngraphormer.eval`、`rxngraphormer.*_workflow` 已删除。
  - `rxngraphormer.compatibility` 承载 checkpoint API；根层 `rxngraphormer.checkpointing` 已删除。
  - `rxngraphormer.inference` 承载 Predictor/embedding API；根层 `rxngraphormer.predictor`、`rxngraphormer.rxn_emb` 已删除。
- 发布预训练分类权重兼容：
  - 默认权重：`model_path/pretrained_classification_model/model/valid_checkpoint.safetensors`
  - legacy fallback：`model_path/pretrained_classification_model/model/valid_checkpoint.pt`
  - bare load：`309/309` keys
  - Lightning wrapper load：`309/309` keys
  - missing/unexpected/shape mismatch 均为 0。
- 同 batch 输出一致性：
  - 数据：`dataset/680w_rxn_data_rct_0_401460.csv` 与 `dataset/680w_rxn_data_pdt_0_401460.csv` 构建 batch。
  - batch size：4
  - bare vs Lightning max logits abs diff：`0.0`
  - bare vs Lightning max probabilities abs diff：`0.0`
- 分类 Lightning 训练 smoke：
  - 数据：`dataset/680w_rxn_data_rct_0_401460.csv` 与 `dataset/680w_rxn_data_pdt_0_401460.csv`
  - 截断：8 条
  - 设备：CPU
  - epoch：1
  - 产物：
    - `tmp/classification_lit_smoke/classification_smoke/version_1/checkpoints/epoch=000-val_acc=0.500000.ckpt`
    - `tmp/classification_lit_smoke/classification_smoke/version_1/manifest.json`
    - `tmp/classification_lit_smoke/classification_smoke/version_1/reports/eval.json`
    - `tmp/classification_lit_smoke/classification_smoke/version_1/reports/eval.csv`
  - valid count：2
  - valid accuracy：`0.5`
  - valid loss：`1.1428941488265991`
- 回归 fine-tune 使用旧分类预训练权重 smoke：
  - 脚本：`scripts/reproduce/check_regression_finetune_with_classification_pretrain.py`
  - 数据：`dataset/BH_rct_tot.csv`、`dataset/BH_pdt_tot.csv`、`dataset/BH_mech_tot.csv`
  - 截断：8 条
  - 设备：CPU
  - epoch：1
  - 预训练权重：`model_path/pretrained_classification_model`
  - rct encoder keys：147
  - pdt encoder keys：147
  - 回归模型 rct encoder vs 分类权重 rct encoder max abs diff：`0.0`
  - 回归模型 pdt encoder vs 分类权重 pdt encoder max abs diff：`0.0`
  - 产物：
    - `tmp/regression_finetune_smoke_gate/regression_finetune_smoke/version_0/checkpoints/epoch=000-val_mae=3.072271.ckpt`
    - `tmp/regression_finetune_smoke_gate/regression_finetune_smoke/version_0/manifest.json`
    - `tmp/regression_finetune_smoke_gate/regression_finetune_smoke/version_0/reports/eval.json`
    - `tmp/regression_finetune_smoke_gate/regression_finetune_smoke/version_0/reports/eval.csv`
- `scripts/reproduce/run_smoke_tests.py --skip-legacy-weight --finetune-output-root tmp/regression_finetune_smoke_gate`
  已通过，覆盖 compile、算法不变量、分类权重兼容、回归 fine-tune smoke。

### 已知边界

- `dataset/50k_with_rxn_type` 是多类别 reaction type 数据，标签包含大于 1 的类别编号，不能直接用于当前二分类预训练 smoke。
- 当前分类 Lightning smoke 验证工程路径与权重兼容，不作为指标复现基线。

## 下一阶段：完整测试矩阵

当前阶段的下一步不是继续搬模块，而是把工程化成果固化成分层测试门禁。测试应按 `static -> smoke -> medium -> full -> release` 逐层推进；每一层都要写出命令、输入资产、输出报告和容忍阈值。默认每次源码结构、序列化、设备管理、checkpoint 或 evaluator 改动后都至少跑 static + smoke。

### 已完成基线

- 静态门禁已经覆盖完整 `src/`：
  - `uv run ruff check src tests scripts/reproduce/check_environment.py`
  - `uv run mypy src`
  - `uv run pyright`
- 核心 smoke 已通过：
  - `uv run python -m unittest discover -s tests -p 'test_*.py'`
  - `uv run python scripts/reproduce/run_smoke_tests.py --skip-finetune`
- 兼容性 smoke 已覆盖：
  - legacy BH 权重 bare/Lightning 双路径加载。
  - 发布分类预训练权重 bare/Lightning 双路径加载。
  - 分类 bare/Lightning 同 batch logits/probabilities 一致。
  - 回归 fine-tune 使用旧分类预训练权重完成 1 epoch smoke。
- 结构边界 smoke 已覆盖：
  - 根层平铺 wrapper 文件不存在。
  - 包内、新维护脚本和测试不再 import 已删除旧路径。
  - `utils.py`、旧 `preprocess` 包、模型层平铺文件、dataloader 平铺文件、scheduler/rxn_emb 平铺文件已迁移到目标边界。
- CLI 边界已收敛：
  - `pyproject.toml` 已注册 `rxngraphormer` 总入口，以及 `rxngraphormer-train`、`rxngraphormer-train-lit`、`rxngraphormer-train-legacy`、`rxngraphormer-eval`、`rxngraphormer-eval-legacy`、`rxngraphormer-compat`、`rxngraphormer-predict`、`rxngraphormer-predict-sequence`、`rxngraphormer-preprocess`。
  - 数据预处理、回归/分类训练、回归/分类评估、回归/分类推理和 forward/retro sequence prediction 都可通过正式 console script 访问。
  - 根目录 `main.py`、`train_model.py`、`eval_model.py`、`data_preprocess.py` 兼容入口已删除；复现脚本改用正式 CLI。
  - CLI 注册和根入口删除已纳入 `tests/test_cli_workflows.py` 自动化断言。
  - CLI entrypoint help smoke 已纳入 `scripts/reproduce/check_cli_entrypoints.py`，并作为 `run_smoke_tests.py` 的默认子门禁。

### 立即补齐：Smoke Gate

- 固化一个单命令 smoke：
  - `uv run python scripts/reproduce/run_smoke_gate.py` 作为 static + smoke 总入口。
  - static 三件套。
  - algorithm invariants。
  - `scripts/reproduce/run_smoke_tests.py` 不带 `--skip-finetune` 的完整快速路径。
  - regression/classification 各 1 个 CPU 小 epoch Lightning 训练。
  - regression/classification 各 1 个 evaluator CLI 报告。
  - regression/classification 各 1 个 Predictor/API 推理。
- 设备与梯度 smoke：
  - 传统路径：`DeviceManager` 移动模型和 batch，loss 可 backward，eval/predict 在 `torch.no_grad()` 下无梯度。
  - Lightning 路径：Trainer 接管设备放置，`training_step` 可 backward，不引入手动 `torch.device` 解析。
  - CPU 必测；CUDA 可用时追加单卡 CUDA smoke。
- 序列化 smoke：
  - processed-data `.safetensors` roundtrip。
  - processed-data legacy `.pt` fallback。
  - checkpoint `.safetensors` 优先加载。
  - `.pt/.pth/.ckpt` fallback 加载。
  - `scripts/convert_pt_to_safetensors.py` 对 checkpoint 和 processed-data 各跑一个最小样例。
  - `scripts/reproduce/check_serialization_smoke.py` 已作为 `run_smoke_tests.py` 的默认子门禁。

### 中等功能测试

- 预处理与数据协议：
  - `rxn_smiles` CSV/Parquet 输入。
  - 旧 `rct_smiles`/`pdt_smiles` 分离列输入。
  - 单环境 `--layout single` 和分离环境 `--layout split` 的 dry/smoke 路径。
  - 显式 atom mapping、mapped H、agent 复制到两侧的 canonical 对比。
  - split manifest、target schema、normalizer schema 与 manifest 写入/读取。
- DataLoader 与 batch：
  - `num_workers=0/2`、`pin_memory`、`persistent_workers`、`prefetch_factor` 组合。
  - regression mid graph collate 与 classification pair collate。
  - `mol_index` tensor schema、专用 PyG batching 语义和 legacy list fallback。
  - 小 batch、末尾不足 batch、空 split 或单样本 split 的错误信息。
- 权重与 checkpoint：
  - BH legacy `.pt`、canonical `.safetensors`、Lightning `.ckpt` 三类加载。
  - 分类预训练权重加载到 bare classifier、Lightning classifier、回归 fine-tune encoder。
  - strict、shape-compatible、diagnostic 三种加载模式的报告内容。
  - resume from Lightning `.ckpt` 后继续训练和日志写入。
- 回归功能：
  - Lightning 训练小 epoch、eval、predict table、predict dataset。
  - legacy regression 回退路径与新 evaluator 的逐样本预测对比。
  - `eval_after_fit` 生成 `manifest.json`、`reports/eval.json`、`reports/eval.csv`。
  - 标准 `rxn_smiles` 与旧分离列在同一权重下预测一致。
- 分类功能：
  - Lightning 训练小 epoch、eval、predict。
  - pretrained checkpoint 兼容报告。
  - evaluator 输出 loss、accuracy、confidence、probabilities。
  - legacy classification 回退路径与新分类 evaluator 的可比输出。
- 推理与导出：
  - `RXNGraphormerPredictor.predict_table()` CSV/Parquet。
  - embedding API、classifier API 与统一 reaction parser。
  - batch size、AMP/BF16 开关、CPU/GPU 推理路径。

### Full / Release Gate

- 当前 OOS 复现诊断记录：
  - 公开 reproduction 参考 Br_OOS：`MAE 5.934`、`R2 0.869`；论文原始报告 Br_OOS：`MAE 5.810`、`R2 0.890`。
  - 当前 bugfix 后 `modern + batch1280 + bf16` 完整 Br_OOS：`valid MAE 7.98`、`R2 0.823`。
  - 当前 `legacy + batch1280 + bf16` 完整 Br_OOS 已完成，best TensorBoard `val_mae ~= 8.03`，最终 evaluator `valid MAE 8.21`、`R2 0.815`。
  - 20 epoch 消融显示 `legacy + batch32 + no warmup scaling + fp32` best `val_mae ~= 24.52`，优于同条件 `bf16` best `31.81`，因此下一条完整主跑是 fp32 原始 batch 轴。
  - 当前 `legacy + batch32 + no warmup scaling + fp32` Lightning full run 已达到 best TensorBoard `val_mae ~= 6.19`，显著优于 batch1280/bf16 轴，但仍未到公开 reproduction `5.934`。
  - 当前 `modern + batch32 + no warmup scaling + fp32` Lightning full run 已达到 best TensorBoard `val_mae ~= 6.63`，低于早期 batch1280 结果且仍在继续收敛；最终优劣需要等待 600 epoch 完成。
  - 当前运行中的完整主跑：
    - `legacy + batch32 + no warmup scaling + fp32`: `experiments/lightning_oos_reproduce_legacy_bs32_fp32_full`，TensorBoard 端口 `6010`。
    - `modern + batch32 + no warmup scaling + fp32`: `experiments/lightning_oos_reproduce_modern_bs32_fp32_full`，TensorBoard 端口 `6011`。
  - 训练循环隔离短跑已完成：
    - `current legacy trainer + legacy + batch32 + no warmup scaling + fp32 + 40 epoch`: `experiments/legacy_oos_loop_probe_short40`，TensorBoard 端口 `6012`，manifest 为 `reports/legacy_oos_loop_probe_short40_manifest.csv`。
    - 该短跑 best/evaluator `valid MAE ~= 18.65`，仅作为 legacy trainer 链路 smoke，不作为性能复现结论。
  - 新增 legacy trainer OOS launcher：`scripts/reproduce/train_legacy_oos_probe.py`。
  - 当前 OOS 复现诊断表：`reports/oos_reproduction_diagnostics.json` 和 `reports/oos_reproduction_diagnostics.csv`。
- 发布级性能回归门禁已落地为报告判定层：
  - `scripts/reproduce/check_release_performance.py` 读取训练/评估任务产出的 JSON 报告，并按 `scripts/reproduce/release_performance_baselines.json` 中的固定基线和容忍阈值判定 pass/fail。
  - 默认覆盖 BH 发布 seed0 评估、BH 600 epoch fine-tune、BH OOS Br/I Lightning full training、full_df `G_T_activate`/`G_T_rxn` Lightning 训练、C_H_func 发布权重评估。
  - 输出为 `reports/release_performance_gate.json` 和 `reports/release_performance_gate.csv`；当前现有报告判定通过。
  - `scripts/reproduce/run_release_gate.py` 作为发布总入口，默认先跑 static + smoke gate，再跑 release performance gate；已有 full 训练报告可用时可用 `--skip-smoke` 只做指标回归判定。
- 指标复现：
  - BH OOS Br/I full training，记录 MAE/RMSE/R2、epoch time、显存峰值和 manifest。
  - C_H_func 至少一个发布权重评估或完整训练流程。
  - `dataset/full_df.xlsx` parser/materialization 全量对比。
  - 回归 fine-tune from classification pretrained checkpoint 的代表性训练。
- 分类预训练：
  - 代表性子集训练，不再只用 8 条 CPU smoke。
  - 发布分类权重评估，记录 accuracy/loss/confidence 分布。
  - 多类别 reaction type 数据如果要纳入，必须先明确 task/head/loss 语义，不能复用二分类 smoke 配置。
  - 新增 GraphNorm 预训练主线：`config/pretrain_graphnorm.json` 使用 `dataset/680w_rxn_data_{rct,pdt}_*.csv`、`rct_norm=pdt_norm=graphnorm`、`head_norm=layernorm`，沿用原始分类预训练 `AdamW + NoamLR + lr=6.0 + warmup_step=60000 + batch_size=5120 + epoch=20`。
  - GraphNorm 预训练 smoke 配置：`config_toml/pretrain_graphnorm_smoke.json`，已验证 Lightning classification、GraphNorm encoder、LayerNorm head、`parameters.json` 与 `model/valid_checkpoint.safetensors` 同目录导出链路。
  - GraphNorm 发布权重目录约定：`model_path/pretrained_classification_graphnorm/pretrain_graphnorm/version_<n>/` 可直接作为下游回归 `pretrained_model_path`；目录内必须包含 `parameters.json` 和 `model/valid_checkpoint.safetensors`。
  - 完整 GraphNorm 预训练前应先跑文件级并行预处理：`bash scripts/run_pretrain_graphnorm_preprocess.sh`；等价用户命令为 `scripts/rxngraphormer_pipeline.sh preprocess --config config/pretrain_graphnorm.json --multi_process --preprocess_num_workers 64 --preprocess_batch_size 256 --preprocess_parallel_mode file`。
  - `preprocess_parallel_mode=file` 的语义是把 worker 分配到 raw CSV 文件粒度；当前分类预处理会先并行处理 rct 文件组、再并行处理 pdt 文件组，每个文件 worker 内部关闭 reaction 级 multiprocessing，避免嵌套进程池。
  - `RXNDataset`、`MultiRXNDataset`、`rxngraphormer-preprocess` 和 Lightning DataModule 已统一接入 `data.multi_process`、`data.preprocess_num_workers`、`data.preprocess_batch_size`、`data.preprocess_parallel_mode`；即使直接启动训练，缓存缺失时也会按配置并行构图。
  - 完整 GraphNorm 预训练首次启动若缺少 processed safetensors 缓存，会先从 raw CSV 生成 `dataset/processed/*.safetensors`，该阶段应在 manifest/log 中单独记录耗时；建议显式预处理后再训练，方便区分预处理耗时和训练吞吐。
- 性能与稳定性：
  - 单卡 epoch time 与 P0/P11 记录对比。
  - 推理吞吐和显存随 batch size 的曲线。
  - DataLoader worker/pin/persistent 配置对吞吐的影响。
  - CUDA 可用时覆盖 AMP/BF16；Blackwell 环境单独记录。
- 发布与环境：
  - `scripts/rxngraphormer_pipeline.sh --layout single rxngraphormer --help`。
  - `scripts/rxngraphormer_pipeline.sh --layout split rxngraphormer --help`。
  - `scripts/rxngraphormer_pipeline.sh --check-env rxngraphormer --help` 和 `scripts/rxngraphormer_pipeline.sh --check-env preprocess --help` 覆盖 model/preprocess 两类环境检查。
  - README、environment docs、checkpoint compatibility docs 中的命令必须能对应到真实测试。

### 测试产物规范

- smoke 产物写入 `tmp/` 或 `experiments/smoke_*`，可清理，不作为基线。
- medium/full 产物写入 `reports/` 或 `experiments/<task>/<tag>/version_<n>/`，必须保留 manifest。
- 每个测试报告至少记录：
  - git diff 摘要。
  - 命令和配置文件路径。
  - 数据版本、split manifest、checkpoint 路径。
  - 设备、CUDA/torch/lightning 版本。
  - 指标、运行时间、显存峰值。
  - 与历史基线的容忍阈值和结论。
