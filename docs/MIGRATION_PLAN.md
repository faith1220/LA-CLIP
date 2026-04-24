# AF-CLIP Clean Rebuild Plan

## 目标

在官方 AF-CLIP 上游代码基础上，逐步迁移当前主线方法，保持：

- 上游 baseline 可独立运行
- 每次只迁移一个模块
- 每个模块都有独立开关
- 所有实验日志与临时结果都放到 `runs/` 或 `log/`，不再混入代码目录

## 当前仓库定位

- 上游干净仓库：`/home/hyz/MyDisk/prompt/AF-CLIP-clean`
- 现有工作仓库：`/home/hyz/MyDisk/prompt/AF-CLIP-main`

## 迁移原则

1. 不把旧仓库的 `log/`、`paper/`、`reports/`、临时脚本整体复制过来。
2. 先做 baseline 对齐，再逐模块迁移。
3. 每个模块迁移完成后都要：
   - `py_compile`
   - 最小单测
   - 一条本地 smoke
4. 当前论文主线优先顺序：
   - `LSAR`
   - `MVTI`
   - `ASA`
   - `test-time image score fusion`

## 建议迁移顺序

### Phase 0: Baseline parity

- 保持上游 AF-CLIP 原始结构
- 先验证 `mvtec -> visa` 和 `visa -> mvtec` baseline 可以跑通
- 不加入任何我们自己的模块

### Phase 1: LSAR

- 先迁移 layer-specific adaptor residual
- 原因：结构清晰、对主线贡献明确、与聚合逻辑解耦
- 要求：
  - 默认关闭
  - 开启后只影响 adaptor 后的 layer-specific residual path

### Phase 2: MVTI

- 迁移测试时水平翻转双视角推理
- 原因：纯 inference 模块，边界清晰
- 要求：
  - 默认关闭
  - 开启后只影响 test-time output fusion

### Phase 3: ASA

- 迁移当前主线使用的最小 adaptive spatial aggregation
- 当前口径：
  - `agg_mode=fixed`
  - `use_adaptive_sa=1`
  - `adaptive_sa_patch_only=1`
- 要求：
  - 不改 fixed kernel / window
  - 默认关闭
  - 只引入全局单标量 residual gate

### Phase 4: Eval-only image score fusion

- 迁移 image-level test-time score reconstruction
- 只放在 eval 路径
- 不影响 pixel map 生成与 pixel metrics

## 暂不迁移的内容

以下内容先不进入新主线：

- `ARSA`
- `DASA`
- `DA3`
- `DCN`
- `NCMR`
- `CDCR`
- `prototype bank`
- 其他失败路线或故事未定的分支

这些内容如果后续需要，单独分支迁移，不混入主线仓库。
