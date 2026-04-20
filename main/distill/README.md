`distill_codebook.py`:
- 用冻结的 teacher residual tokenizer（如 `prepare/res_kmeans.py` 训练得到的 checkpoint）和目标域 item embedding，蒸馏出更轻量的 target codebook。
- 输入：target-domain embedding parquet + teacher tokenizer checkpoint。
- 输出：`distilled_codebook.pt`、`distilled_item_sid_map.parquet`、`transport_map.parquet`、`stats.json`。
- 主要流程：先用 teacher codebook 对目标域 embedding 编码，再按 teacher code 使用频次压缩 centroid，并在目标域 residual embedding 上 warm-start refine。
**Arguments:**
- `--teacher_model_path`: teacher ResKmeans checkpoint 路径
- `--emb_path`: 目标域 embedding parquet 路径，需要包含 `item_id/pid/parent_asin` 之一和 `embedding`
- `--output_dir`: 蒸馏结果输出目录
- `--target_codebook_sizes`: 目标 codebook 每层大小，逗号分隔；单个值会自动扩展到所有层，默认 `256,256,256`
- `--n_layers`: 使用的 residual 层数，默认使用 teacher 的全部层
- `--count_power`: teacher code 使用计数的幂次缩放，默认 `1.0`
- `--teacher_init_n_init`: teacher centroid 压缩阶段 weighted k-means 的重启次数，默认 `8`
- `--teacher_init_max_iter`: teacher centroid 压缩阶段最大迭代轮数，默认 `50`
- `--refine_max_iter`: 目标域 residual refine 阶段最大迭代轮数，默认 `20`
- `--batch_size`: teacher 编码 batch size，默认 `4096`
- `--seed`: 随机种子，默认 `42`
- `--device`: 运行设备，默认自动选择 `cuda/cpu`

**Input format:** Parquet with columns `item_id`/`pid`/`parent_asin` and `embedding`

**Output format:**
- `distilled_codebook.pt`: 蒸馏后的 residual codebook checkpoint，供后续 `infer.py` 直接加载
- `distilled_item_sid_map.parquet`: 包含 `item_id`、`teacher_codes`、`teacher_sid`、`codes`、`sid`
- `transport_map.parquet`: teacher code 到 target code 的层内映射信息，包含距离与 usage count
- `stats.json`: 蒸馏统计信息，包括每层 codebook size、teacher/distilled reconstruction MSE、SID collision 等

`infer.py`:
- 用蒸馏后的 target codebook 对 item embedding 做编码推理。
- 输入：embedding parquet + `distilled_codebook.pt`。
- 输出：`*_distilled_codes.parquet`。
- 适合在 codebook 蒸馏完成后，对新一批目标域 item embedding 批量生成 residual codes。
**Arguments:**
- `--model_path`: `distilled_codebook.pt` 路径
- `--emb_path`: embedding parquet 路径，需要包含 `item_id/pid/parent_asin` 之一和 `embedding`
- `--output_path`: 输出 parquet 路径，默认 `{emb_path}_distilled_codes.parquet`
- `--batch_size`: 推理 batch size，默认 `4096`
- `--device`: 编码设备，默认自动选择 `cuda/cpu`

**Input format:** Parquet with columns `item_id`/`pid`/`parent_asin` and `embedding`

**Output format:** Parquet with columns `item_id`/`pid`/`parent_asin` and `codes`

`distill_utils.py`:
- 蒸馏与推理共用的工具模块，不直接作为命令行入口。
- 主要内容：
- `ResidualCodebook`: residual codebook 的 encode/decode/load/save
- `load_item_embeddings` / `load_item_codes`: 统一读取 embedding 与 code parquet/csv
- `weighted_kmeans`: teacher centroid 压缩与 target residual refine 所用的加权 k-means
- `sid_from_codes` / `parse_sid` / `collision_stats`: SID 字符串构造、解析与碰撞统计
- `batched_encode_embeddings` / `batched_reconstruction_mse`: 批量编码与重构误差评估
