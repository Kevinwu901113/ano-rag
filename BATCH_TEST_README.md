# 批量测试脚本使用说明

本目录包含了用于批量处理musique数据集的测试脚本。

## 文件说明

- `batch_test.py`: 核心批量处理类和函数
- `test_batch.py`: 使用example.jsonl的测试脚本
- `run_musique_batch.py`: 专门用于处理musique数据集的优化脚本（推荐使用）
- `BATCH_TEST_README.md`: 本说明文件

## 功能特性

### batch_test.py

这是主要的批量处理脚本，具有以下功能：

1. **读取JSONL文件**: 读取`data/musique_ans_v1.0_train.jsonl`文件中的每一行数据
2. **独立处理**: 为每行数据创建独立的临时工作目录，避免数据冲突
3. **完整流程**: 对每行数据执行完整的文档处理和查询流程
4. **结果收集**: 将所有查询结果按照`query.json`格式保存到`anorag.json`

### 核心处理流程

对于每一行数据，脚本会：

1. 将`paragraphs`字段中的文档提取并保存为临时文件
2. 修改配置文件，指向临时数据目录
3. 运行`python main.py process --new --force`处理文档
4. 运行`python main.py query "问题内容"`进行查询
5. 收集查询结果并格式化
6. 清理临时文件

## 使用方法

### 1. 测试运行（推荐先执行）

```bash
# 使用example.jsonl进行测试
python test_batch.py
```

这会处理example.jsonl中的单个样本，结果保存到`test_result.json`。

### 2. 批量处理完整数据集（推荐方法）

```bash
# 使用优化版脚本处理musique数据集
python run_musique_batch.py

# 限制处理数量（用于测试）
python run_musique_batch.py --limit 10

# 从特定位置开始处理
python run_musique_batch.py --start 100 --limit 50

# 恢复中断的处理（从已有结果继续）
python run_musique_batch.py --resume

# 指定输出文件名
python run_musique_batch.py --output my_results.json
```

### 3. 使用基础批量处理脚本

```bash
# 处理完整的musique数据集
python batch_test.py

# 或者指定参数
python batch_test.py --input data/musique_ans_v1.0_train.jsonl --output anorag.json

# 限制处理数量（用于测试）
python batch_test.py --limit 10
```

### 4. 参数说明

#### run_musique_batch.py（推荐）
- `--limit, -l`: 限制处理的数据条数（用于测试）
- `--start, -s`: 从指定索引开始处理（0-based）
- `--output, -o`: 输出JSON文件路径（默认: `anorag.json`）
- `--resume, -r`: 从现有输出文件恢复处理

#### batch_test.py（基础版）
- `--input, -i`: 输入JSONL文件路径（默认: `data/musique_ans_v1.0_train.jsonl`）
- `--output, -o`: 输出JSON文件路径（默认: `anorag.json`）
- `--limit, -l`: 限制处理的数据条数（用于测试）

## 输出格式

输出文件`anorag.json`包含一个JSON数组，每个元素的格式如下：

```json
{
    "id": "数据项ID",
    "predicted_answer": "预测答案",
    "predicted_support_idxs": [],
    "predicted_answerable": true/false
}
```

如果处理过程中出现错误，还会包含`error`字段。

## 注意事项

1. **数据文件**: 确保`data/musique_ans_v1.0_train.jsonl`文件存在
2. **依赖环境**: 确保所有依赖包已安装（见`requirements.txt`）
3. **存储空间**: 批量处理会创建多个工作目录，确保有足够的磁盘空间
4. **处理时间**: 完整数据集处理时间较长，建议先用`--limit`参数测试
5. **GPU资源**: 如果使用GPU，确保显存充足
6. **中断恢复**: 使用`run_musique_batch.py --resume`可以从中断处继续
7. **进度监控**: 脚本会显示详细的进度信息和预估完成时间

## 工作目录管理

脚本会自动管理工作目录：
- 每个数据项使用独立的临时目录
- 处理完成后自动清理临时文件
- 主工作目录在`result/`下按数字编号创建

## 故障排除

1. **配置文件错误**: 检查`config.yaml`文件格式是否正确
2. **依赖缺失**: 运行`pip install -r requirements.txt`
3. **权限问题**: 确保脚本有读写权限
4. **内存不足**: 减少batch_size或使用`--limit`参数
5. **处理中断**: 使用Ctrl+C安全中断，脚本会保存当前进度
6. **数据格式错误**: 检查输入JSONL文件每行是否为有效JSON
7. **工作目录冲突**: 脚本会自动创建独立工作目录，避免冲突

## 性能优化建议

1. **小批量测试**: 先用`--limit 5`测试几个样本
2. **分批处理**: 对于大数据集，可以分批处理：
   ```bash
   # 处理前100个
   python run_musique_batch.py --limit 100
   # 继续处理接下来的100个
   python run_musique_batch.py --start 100 --limit 100
   ```
3. **监控资源**: 注意CPU、内存和GPU使用情况
4. **定期保存**: 脚本每5个项目自动保存一次进度

## 日志

脚本使用loguru进行日志记录，会显示处理进度和错误信息。每个工作目录下也会生成`ano-rag.log`日志文件。