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

迁移期间可保留当前 backward-compatible import path，例如 `rxngraphormer.model`,
`rxngraphormer.layer`, `rxngraphormer.predictor`, `rxngraphormer.eval`，但这些文件应逐步变成
薄 wrapper，实际实现移动到上述边界内。重构顺序必须先移动和包裹，再删除或改名；每一步都用
checkpoint load、bare/Lightning forward、CLI smoke 和 Predictor smoke 做回归验证。

## 非破坏性兼容边界

以下行为在完成 Lightning 工程化前应保持稳定：

- `RXNGRegressor`、`RXNGEncoder` 和主要 graph layer 的参数含义不变。
- legacy 权重可以加载到裸 `torch.nn.Module`。
- legacy 权重也可以通过 adapter 加载到 `LightningModule.model`。
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
| P12. 源码结构收敛 | 将模型定义/实现与训练、推理、评估框架解耦 | 建立 `models/`、`training/`、`inference/`、`evaluation/`、`data/`、`preprocessing/`、`compatibility/` 边界；把当前平铺模块逐步迁移为 wrapper；集中 temporary CSV/materialization helper；把 CLI 变成薄入口 | 模型 core 可在不导入 Lightning/CLI 的情况下构建和 forward；训练和推理仅依赖 model factory/checkpoint adapter；旧 import path 仍可用；全量 smoke 通过 | 模块移动容易破坏旧权重 key、pickle/import path、用户脚本，需要分阶段保留 wrapper 和 deprecation 说明 |

## 当前阶段成果

截至当前重构阶段，项目已经收敛为单个 `rxngraphormer` Python 包。模型训练/推理默认依赖保持宽松，预处理和 atom mapping 通过 `preprocess` extra 进入可选环境；在较老硬件上可用 `.[all]` 完整端到端运行，在更新 GPU 上可用分离式预处理环境生成数据，再由主模型环境训练和推理。回归链路已经完成从脚本式训练到标准 Lightning 工作流的主要迁移，且保留了旧权重、旧配置和用户侧 API 的兼容性。

### 已完成的工程面

- 项目结构已按工程边界拆分，临时文件、数据文件、训练产物等已纳入 `.gitignore` 管理。
- 包结构已确定为单包发布：
  - 默认依赖面向模型训练、评估和推理。
  - `preprocess` extra 包含 RDKit、atom mapping、DGL 等预处理依赖。
  - `all` extra 支持兼容硬件上的单环境端到端流程。
  - `sequence` extra 保留 sequence generation 依赖边界。
- 自动部署脚本已支持单环境和分离式环境：
  - `scripts/rxngraphormer_pipeline.sh --layout single` 使用一个环境执行预处理、训练和评估。
  - `scripts/rxngraphormer_pipeline.sh --layout split` 使用预处理环境生成数据，模型环境训练/推理。
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
- 回归训练已迁移到标准 Lightning 工作流：
  - `rxngraphormer.lightning.module`
  - `rxngraphormer.lightning.datamodule`
  - `rxngraphormer.lightning.trainer`
  - `rxngraphormer.lightning.workflow`
  - `rxngraphormer-train` 对 `task="regression"` 默认走 Lightning。
  - `rxngraphormer-train-legacy` 和 `--legacy_regression` 保留旧路径。
- 回归评估/推理已统一：
  - `RXNGraphormerPredictor` 支持 legacy 权重和 Lightning checkpoint。
  - `RXNGraphormerPredictor.predict_table()` 支持标准 `rxn_smiles` 表格和旧分离列。
  - `reaction_prediction()`、`RXNEMB.gen_rxn_emb()`、`RXNClassifier.rxn_pred()` 已改用统一 reaction parser，而不是字符串 `>>` 切分。
  - `rxngraphormer.regression_workflow` 统一 split 构建、checkpoint 加载、metrics、JSON/CSV 报告。
  - `eval_regression_performance()` 保持旧 API，但内部委托新 workflow。
  - `rxngraphormer-eval` 对 regression 默认走新 evaluator。
- 实验产物标准化已完成：
  - 每次 Lightning fit 默认写 `manifest.json`。
  - 支持 `--eval_after_fit`，训练后自动用 best checkpoint 生成 `reports/eval.json` 和 `reports/eval.csv`。
  - 标准目录为：
    - `<root>/<tag>/version_<n>/checkpoints/`
    - `<root>/<tag>/version_<n>/reports/`
    - `<root>/<tag>/version_<n>/manifest.json`
    - `<root>/manifest.json`

### 已完成的验证

- 模型算法不变量测试已覆盖核心兼容路径，目前 smoke 单测为 `79 tests OK`。
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

- 回归训练、评估、推理已进入新工作流。
- 表格输入、Python API 和 CLI 推理已支持标准 `rxn_smiles` 与旧分离列两种协议。
- `RXNGraphormerLitModule` 和 `rxngraphormer.lightning.workflow` 当前仍只支持 regression；classification 训练和 sequence generation 训练暂时仍保留 legacy 路径。
- classification 推理已有 `RXNGraphormerPredictor` API，但训练、评估报告和 manifest 尚未进入 Lightning 工作流。
- sequence generation 依赖 OpenNMT、vocab、beam search 和旧 decoder 语义，后续迁移优先级低于分类预训练器。
- 预训练分类器是回归 fine-tune 的基础权重来源，下一阶段必须优先保证 state_dict、encoder forward 和 fine-tune 初始化兼容。

## 下一阶段目标：P11 预训练分类器 Lightning 迁移与 P12 源码结构收敛

下一阶段聚焦 `task="classification"`，尤其是预训练分类器，同时启动源码结构收敛。目标不是重写分类模型算法，而是把分类预训练纳入与回归一致的 Lightning 工程外壳，并将模型定义/实现从训练、推理、评估和 CLI 中解耦出来。

### 目标边界

- 保持 `RXNGClassifier` 裸模型结构和参数语义不变。
- 保持已发布预训练分类权重可加载。
- 保持回归 fine-tune 通过 `pretrained_model_path` 初始化 encoder 的行为不变。
- 保持用户侧训练入口稳定，新增回退开关而不是删除 legacy 路径。
- 保持标准 `rxn_smiles` 与旧 `rct/pdt` 分离输入两种协议在训练、评估和推理路径上的等价语义。
- 继续维持单包发布和 extras 分层，不拆分 package。

### 具体任务

1. 先定义并落地源码边界：
   - 新建或规划 `rxngraphormer.models` 作为模型 core，承载当前 `model.py`、`layer.py`、`model_factory.py` 中与模型结构直接相关的实现。
   - 新建或规划 `rxngraphormer.training` 作为训练框架，承载 LightningModule、DataModule、trainer/workflow、callbacks。
   - 新建或规划 `rxngraphormer.inference` 作为推理框架，承载 Predictor、table/reaction input adapter、exporters。
   - 新建或规划 `rxngraphormer.evaluation` 作为评估框架，承载 metrics、report writer、standalone evaluator。
   - 新建或规划 `rxngraphormer.compatibility` 作为 legacy checkpoint/config/import adapters。
   - 当前平铺文件先变成 backward-compatible wrapper，不立即删除。

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

### P11/P12 验收标准

- `rxngraphormer-train --config <classification_config>` 默认使用 Lightning。
- `--legacy_classification` 可回退旧分类 trainer。
- 分类预训练 smoke 可完成训练、写 checkpoint、写 manifest。
- 模型 core 可独立 import/build/forward，不导入 Lightning、CLI 或 Predictor。
- `rxngraphormer.model`、`rxngraphormer.layer`、`rxngraphormer.predictor` 等旧 import path 在迁移期仍可用。
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

## 第一批建议落地任务

1. 新增 `rxngraphormer/checkpointing.py`，实现 legacy checkpoint adapter。
2. 新增 `rxngraphormer/lightning_module.py`，封装现有 `RXNGRegressor`。
3. 新增 `rxngraphormer/datamodule.py`，迁移 dataloader 和 split manifest 读取。
4. 新增 `rxngraphormer/evaluator.py`，统一 train/eval/predict 指标逻辑。
5. 修复 `F.dropout(..., training=True)` 和 gradient accumulation。
6. 将 validation/test prediction 聚合改为 list accumulation。
7. 增加 `scripts/reproduce/check_legacy_weight_compat.py`，作为第一条 smoke test。
