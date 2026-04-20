`res_kmeans.py`:
- 把 item_embeddings.parquet 编成 residual code。
- 输入：embedding parquet + tokenizer checkpoint。
- 输出：*_codes.parquet。
这是所有 item_sid_map 构造脚本的共同上游。
**Arguments:**
- `--model_path`: Path to trained model checkpoint
- `--emb_path`: Path to parquet file with `pid` and `embedding` columns
- `--output_path`: Output path (default: `{emb_path}_codes.parquet`)
- `--batch_size`: Inference batch size (default: 10000)
- `--device`: Device to use (default: cuda if available)
- `--n_layers`: Number of layers to use (default: all)

**Input format:** Parquet with columns `pid`, `embedding`

**Output format:** Parquet with columns `pid`, `codes`

`s0.py`:
- 从交互 CSV 和商品 metadata 收集 item 文本，调用 embedding model 生成 item embedding。
- 输入：raw interaction CSV、target/source metadata。(raw interaction CSV来源：https://amazon-reviews-2023.github.io/data_processing/5core.html里Statistics的link)
- 输出：item_metadata.parquet、item_embeddings.parquet。
- `user_sequences_csv` 仅支持原始交互 CSV：`user_id,parent_asin,rating,timestamp`。
- 脚本只使用 `parent_asin` 的全局去重集合来筛选需要处理的 item。
- 当 `--target_item_scope labels_only` 时，会用这组去重后的 `parent_asin` 过滤 target meta。

`s1.py`:
- 把 metadata 和 tokenizer codes 合并，生成最终 item_id -> codes -> sid 映射。
- 输入：item_metadata.parquet、*_codes.parquet。
- 输出：item_sid_map.parquet。
