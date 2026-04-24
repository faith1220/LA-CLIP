# Module Source Map

这份表只记录“当前主线需要迁移的模块”。

| 模块 | 当前仓库源码位置 | 新仓库目标位置 | 关键开关 | 备注 |
|---|---|---|---|---|
| LSAR | `clip/model.py` 中 `LayerAdaptorResidual` 与 `use_lsar` 接线 | `clip/model.py`, `main.py` | `--use_lsar` | 每层一个 residual MLP，同层跨尺度共享 |
| MVTI | `util/utils.py` 中 `multi_view_inference` 与 `use_mvti` 调用 | `util/utils.py`, `main.py` | `--use_mvti` | 只做 test-time flip fusion |
| ASA | `clip/dynscale.py` 中 `AdaptiveSpatialAggregator`；`clip/model.py` 中 `use_adaptive_sa` 接线 | `clip/dynscale.py`, `clip/model.py`, `main.py` | `--use_adaptive_sa`, `--adaptive_sa_max_alpha`, `--adaptive_sa_patch_only`, `--adaptive_sa_hidden_dim` | 当前主线只用最小 adaptive residual gate |
| Image score fusion | `util/utils.py` 中 `_compute_eval_image_scores` 及相关 helper | `util/utils.py`, `main.py` | `--img_score_mode` 等 | 只改 eval image score，不动 pixel path |

## 不进入当前主线的来源文件

- `clip/mask_refiner.py`
- `clip/prototype_bank.py`
- `clip/domain_consistency.py`
- `clip/feature_regularizers.py`
- `clip/clsa.py`
- `util/ncmr_diagnostics.py`

这些文件代表其他实验路线，不应在 clean rebuild 初期引入。
