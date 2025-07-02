# Ano-RAG

一个基于图谱增强的高级检索增强生成（RAG）系统，结合了向量检索、知识图谱和智能上下文调度，为复杂查询提供精准的答案生成。

## 🌟 核心特性

### 📚 智能文档处理
- **多格式支持**: JSON、JSONL、DOCX等格式文档
- **原子笔记生成**: 将文档分解为语义完整的原子笔记单元
- **增量处理**: 智能检测文档变更，避免重复处理
- **批量处理**: 支持大规模文档的高效并行处理

### 🧠 知识图谱构建
- **多层关系提取**: 支持引用、实体共现、上下文关系、主题链接等多种关系类型
- **图谱索引**: 高效的图结构索引和检索
- **K-hop检索**: 基于图结构的多跳关系检索
- **关系权重**: 智能计算不同关系类型的重要性权重

### 🔍 混合检索策略
- **向量检索**: 基于语义相似度的密集向量检索
- **图谱检索**: 基于知识图谱的关系推理检索
- **智能融合**: 多种检索结果的智能融合和排序

### 🎯 上下文调度
- **多维度评分**: 语义相似度、图谱重要性、主题相关性、反馈质量等
- **冗余消除**: 智能去除重复和冗余信息
- **动态权重**: 根据查询类型动态调整各维度权重

### 🚀 性能优化
- **GPU加速**: 支持CUDA加速的向量计算和图处理
- **RAPIDS集成**: 使用NVIDIA RAPIDS进行高性能数据处理
- **缓存机制**: 多层缓存提升响应速度
- **并行处理**: 充分利用多核CPU和GPU资源

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   文档输入      │───▶│   文档处理      │───▶│   原子笔记      │
│  (多种格式)     │    │  (分块+生成)    │    │   (结构化)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   查询处理      │◀───│   上下文调度    │◀───│   知识图谱      │
│  (重写+扩展)    │    │  (智能选择)     │    │  (关系提取)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       ▲
         ▼                       ▼                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   向量检索      │    │   答案生成      │    │   向量存储      │
│  (语义匹配)     │───▶│   (LLM生成)     │    │  (嵌入索引)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📦 安装配置

### 环境要求
- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- 16GB+ RAM (推荐)
- Ollama (用于本地LLM服务)

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd anorag
```

2. **安装依赖**
```bash
# 基础依赖
pip install -r requirements.txt

# GPU加速依赖 (可选)
conda install -c rapidsai -c conda-forge cudf cuml cugraph
```

3. **配置Ollama**
```bash
# 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下载模型
ollama pull llama3.1:8b
```

4. **配置系统**
```bash
# 复制配置文件
cp config.yaml.example config.yaml

# 根据需要修改配置
vim config.yaml
```

### 配置说明

主要配置项包括：

- **文档处理**: 分块大小、重叠度、批处理大小
- **嵌入模型**: 模型选择、批处理大小、设备配置
- **聚类算法**: HDBSCAN、K-means、DBSCAN参数
- **图谱构建**: 关系类型、权重配置、相似度阈值
- **LLM配置**: 本地模型和Ollama服务配置
- **性能优化**: GPU使用、并行度、缓存设置

## 🚀 快速开始

### 1. 处理文档

```bash
# 使用最近的结果目录处理 data/ 下的文档
python main.py process

# 创建新的工作目录并重新处理
python main.py process --new

# 强制重新处理所有文档
python main.py process --force
```

### 2. 查询问答

```bash
# 简单查询
python main.py query "什么是机器学习？"

# 从指定目录查询
python main.py query "深度学习和传统机器学习的主要区别是什么？" --work-dir result/1
```

### 3. Python API使用

```python
from doc import DocumentProcessor
from query import QueryProcessor

# 处理文档
processor = DocumentProcessor(output_dir="result/1")
result = processor.process_documents(["data/document.json"]) 

# 查询处理
query_processor = QueryProcessor(result['atomic_notes'])
response = query_processor.process("你的问题")
print(response['answer'])
```

## 📊 性能特性

### 处理能力
- **文档处理**: 1000+ 文档/小时
- **查询响应**: <2秒 (典型查询)
- **并发支持**: 10+ 并发查询
- **内存效率**: 智能批处理和缓存

### 准确性指标
- **检索精度**: 85%+ (在标准数据集上)
- **答案质量**: 基于BLEU、ROUGE等指标评估
- **关系准确性**: 90%+ (实体关系提取)

## 🔧 高级功能

### 自定义配置

```yaml
# 自定义聚类算法
clustering:
  algorithm: "hdbscan"
  min_cluster_size: 10
  custom_params:
    cluster_selection_epsilon: 0.5

# 自定义图谱权重
graph:
  weights:
    reference: 1.0
    entity_coexistence: 0.8
    semantic_similarity: 0.6
```

### 评估和监控

```bash
# 运行评估
python -m eval.evaluator --dataset eval_data.json

# 性能监控
python -m utils.monitor --watch-queries
```

### 扩展开发

```python
# 自定义关系提取器
from graph.relation_extractor import RelationExtractor

class CustomRelationExtractor(RelationExtractor):
    def extract_custom_relations(self, notes):
        # 实现自定义关系提取逻辑
        pass
```

## 📁 项目结构

```
anorag/
├── config/              # 配置管理
├── doc/                 # 文档处理模块
│   ├── chunker.py      # 文档分块
│   ├── clustering.py   # 主题聚类
│   └── document_processor.py  # 主处理器
├── graph/              # 知识图谱模块
│   ├── graph_builder.py    # 图谱构建
│   ├── graph_retriever.py  # 图谱检索
│   └── relation_extractor.py  # 关系提取
├── llm/                # 语言模型模块
│   ├── atomic_note_generator.py  # 原子笔记生成
│   ├── query_rewriter.py      # 查询重写
│   └── ollama_client.py       # Ollama客户端
├── query/              # 查询处理模块
├── vector_store/       # 向量存储模块
├── utils/              # 工具模块
├── eval/               # 评估模块
└── main.py            # 主入口
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) - 嵌入模型
- [Ollama](https://ollama.ai/) - 本地LLM服务
- [NVIDIA RAPIDS](https://rapids.ai/) - GPU加速计算
- [NetworkX](https://networkx.org/) - 图处理库

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件
- 参与讨论

---

**Ano-RAG** - 让知识检索更智能，让答案生成更精准！