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
- 从用户序列和商品 metadata 收集 item 文本，调用 embedding model 生成 item embedding。
-输入：user_sequences_csv、target/source metadata。
- 输出：item_metadata.parquet、item_embeddings.parquet。

`s1.py`:
- 把 metadata 和 tokenizer codes 合并，生成最终 item_id -> codes -> sid 映射。
- 输入：item_metadata.parquet、*_codes.parquet。
- 输出：item_sid_map.parquet。