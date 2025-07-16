"""增强原子笔记生成器的配置文件"""

# 增强NER配置
ENHANCED_NER_CONFIG = {
    # 模型配置
    'model_name': 'dbmdz/bert-large-cased-finetuned-conll03-english',  # 可选: 'spacy', 'bert-base-cased'
    'model_type': 'transformers',  # 'transformers' 或 'spacy'
    'device': 'auto',  # 'auto', 'cpu', 'cuda'
    
    # 实体过滤配置
    'entity_types': ['PERSON', 'ORG', 'GPE', 'WORK_OF_ART', 'EVENT'],  # 保留的实体类型
    'min_entity_length': 2,
    'max_entity_length': 50,
    
    # 人物识别配置
    'person_indicators': [
        'mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'lady', 'lord',
        'actor', 'actress', 'director', 'writer', 'author', 'singer',
        'musician', 'artist', 'politician', 'president', 'minister'
    ],
    'non_person_indicators': [
        'company', 'corporation', 'inc', 'ltd', 'llc', 'organization',
        'university', 'college', 'school', 'hospital', 'museum'
    ],
    
    # 实体归一化配置
    'similarity_threshold': 0.85,  # 实体相似度阈值
    'use_fuzzy_matching': True,
    'normalize_case': True,
    'remove_titles': True,
    
    # 实体追踪配置
    'enable_entity_tracking': True,
    'tracking_window_size': 5,  # 追踪窗口大小
    'min_tracking_confidence': 0.7
}

# 关系抽取配置
RELATION_EXTRACTION_CONFIG = {
    # 关系模板配置
    'relation_templates': {
        'spouse': [
            r'(.+?)\s+(?:is|was)\s+(?:the\s+)?(?:wife|husband|spouse)\s+of\s+(.+)',
            r'(.+?)\s+(?:married|wed)\s+(.+)',
            r'(.+?)\s+and\s+(.+?)\s+(?:are|were)\s+married'
        ],
        'parent_child': [
            r'(.+?)\s+(?:is|was)\s+(?:the\s+)?(?:son|daughter|child)\s+of\s+(.+)',
            r'(.+?)\s+(?:is|was)\s+(?:the\s+)?(?:father|mother|parent)\s+of\s+(.+)'
        ],
        'actor_of': [
            r'(.+?)\s+(?:voiced|played|portrayed)\s+(.+)',
            r'(.+?)\s+(?:is|was)\s+(?:the\s+)?voice\s+(?:actor\s+)?(?:of|for)\s+(.+)',
            r'(.+?)\s+(?:stars|starred)\s+(?:as\s+)?(.+)'
        ],
        'creator_of': [
            r'(.+?)\s+(?:created|wrote|directed|produced)\s+(.+)',
            r'(.+?)\s+(?:is|was)\s+(?:the\s+)?(?:creator|author|director|writer)\s+of\s+(.+)'
        ],
        'associated_with': [
            r'(.+?)\s+(?:is|was)\s+(?:known|famous)\s+for\s+(.+)',
            r'(.+?)\s+(?:works|worked|collaborates|collaborated)\s+(?:with|on)\s+(.+)',
            r'(.+?)\s+(?:appears|appeared)\s+in\s+(.+)'
        ],
        'location': [
            r'(.+?)\s+(?:is|was)\s+(?:located|situated|based)\s+in\s+(.+)',
            r'(.+?)\s+(?:lives|lived|resides|resided)\s+in\s+(.+)',
            r'(.+?)\s+(?:from|of)\s+(.+?)(?:\s+(?:city|state|country|region))?'
        ],
        'temporal': [
            r'(.+?)\s+(?:was\s+born|born)\s+(?:in\s+|on\s+)?(.+)',
            r'(.+?)\s+(?:died|passed\s+away)\s+(?:in\s+|on\s+)?(.+)',
            r'(.+?)\s+(?:happened|occurred|took\s+place)\s+(?:in\s+|on\s+|during\s+)?(.+)'
        ]
    },
    
    # 上下文关系配置
    'context_window_size': 100,  # 上下文窗口大小
    'min_entity_distance': 1,    # 实体间最小距离
    'max_entity_distance': 50,   # 实体间最大距离
    
    # 关系验证配置
    'min_relation_confidence': 0.6,
    'enable_relation_validation': True,
    'max_relations_per_note': 10
}

# 增强去噪配置
NOISE_FILTER_CONFIG = {
    # 评分权重
    'weights': {
        'importance_score': 0.4,
        'summary_length_score': 0.3,
        'verified_entity_ratio': 0.3
    },
    
    # 阈值设置
    'usefulness_threshold': 0.65,
    'min_content_length': 20,
    'max_summary_length_for_score': 100,
    
    # 质量指标权重
    'quality_weights': {
        'content_completeness': 0.25,
        'entity_relevance': 0.25,
        'information_density': 0.20,
        'linguistic_quality': 0.15,
        'factual_consistency': 0.15
    },
    
    # 噪声模式
    'noise_patterns': [
        r'^\s*$',  # 空内容
        r'^\s*\.\.\.+\s*$',  # 只有省略号
        r'^\s*[^a-zA-Z0-9\u4e00-\u9fff]*$',  # 只有标点符号
        r'^\s*(?:the|a|an|and|or|but)\s*$',  # 只有停用词
        r'^\s*\d+\s*$',  # 只有数字
    ],
    
    # 高质量内容指示词
    'quality_indicators': {
        'factual': ['born', 'died', 'created', 'founded', 'established', 'married', 'divorced'],
        'descriptive': ['known for', 'famous for', 'characterized by', 'described as'],
        'relational': ['son of', 'daughter of', 'married to', 'worked with', 'collaborated'],
        'temporal': ['in', 'during', 'before', 'after', 'since', 'until'],
        'quantitative': ['first', 'last', 'most', 'least', 'many', 'few', 'several']
    }
}

# 笔记相似度配置
NOTE_SIMILARITY_CONFIG = {
    # 模型配置
    'model_name': 'all-MiniLM-L6-v2',  # sentence-transformers模型
    'similarity_threshold': 0.75,
    'max_related_notes': 5,
    'batch_size': 32,
    
    # 相似度计算配置
    'use_content': True,
    'use_summary': True,
    'content_weight': 0.6,
    'summary_weight': 0.4,
    
    # 实体相似度配置
    'entity_similarity_weight': 0.3,
    'min_shared_entities': 1,
    
    # 聚类过滤配置
    'exclude_same_cluster': True,
    'cluster_similarity_bonus': 0.1
}

# 主配置
ENHANCED_ATOMIC_NOTE_CONFIG = {
    # 功能开关
    'enable_enhanced_ner': True,
    'enable_relation_extraction': True,
    'enable_enhanced_noise_filter': True,
    'enable_note_similarity': True,
    
    # 子模块配置
    'ner': ENHANCED_NER_CONFIG,
    'relation_extraction': RELATION_EXTRACTION_CONFIG,
    'noise_filter': NOISE_FILTER_CONFIG,
    'similarity': NOTE_SIMILARITY_CONFIG,
    
    # 性能配置
    'batch_size': 32,
    'use_gpu': True,
    'max_workers': 4,
    
    # 输出配置
    'export_results': True,
    'export_directory': './enhanced_results',
    'export_formats': ['json', 'csv'],
    
    # 日志配置
    'log_level': 'INFO',
    'detailed_logging': True,
    'log_statistics': True
}

# 预设配置模板
CONFIG_PRESETS = {
    'fast': {
        # 快速模式：优先速度
        'ner': {
            **ENHANCED_NER_CONFIG,
            'model_type': 'spacy',
            'enable_entity_tracking': False
        },
        'relation_extraction': {
            **RELATION_EXTRACTION_CONFIG,
            'enable_relation_validation': False,
            'max_relations_per_note': 5
        },
        'similarity': {
            **NOTE_SIMILARITY_CONFIG,
            'model_name': 'paraphrase-MiniLM-L3-v2',
            'batch_size': 64,
            'max_related_notes': 3
        }
    },
    
    'accurate': {
        # 精确模式：优先准确性
        'ner': {
            **ENHANCED_NER_CONFIG,
            'model_name': 'dbmdz/bert-large-cased-finetuned-conll03-english',
            'similarity_threshold': 0.9,
            'enable_entity_tracking': True
        },
        'relation_extraction': {
            **RELATION_EXTRACTION_CONFIG,
            'enable_relation_validation': True,
            'min_relation_confidence': 0.8
        },
        'noise_filter': {
            **NOISE_FILTER_CONFIG,
            'usefulness_threshold': 0.7
        },
        'similarity': {
            **NOTE_SIMILARITY_CONFIG,
            'model_name': 'all-mpnet-base-v2',
            'similarity_threshold': 0.8
        }
    },
    
    'balanced': {
        # 平衡模式：速度和准确性平衡
        'ner': ENHANCED_NER_CONFIG,
        'relation_extraction': RELATION_EXTRACTION_CONFIG,
        'noise_filter': NOISE_FILTER_CONFIG,
        'similarity': NOTE_SIMILARITY_CONFIG
    }
}

def get_config(preset: str = 'balanced') -> dict:
    """获取指定预设的配置"""
    if preset not in CONFIG_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available presets: {list(CONFIG_PRESETS.keys())}")
    
    base_config = ENHANCED_ATOMIC_NOTE_CONFIG.copy()
    preset_config = CONFIG_PRESETS[preset]
    
    # 合并配置
    for key, value in preset_config.items():
        if key in base_config:
            if isinstance(value, dict) and isinstance(base_config[key], dict):
                base_config[key].update(value)
            else:
                base_config[key] = value
    
    return base_config

def validate_config(config: dict) -> bool:
    """验证配置的有效性"""
    required_keys = ['ner', 'relation_extraction', 'noise_filter', 'similarity']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    # 验证阈值范围
    if not 0 <= config['similarity']['similarity_threshold'] <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")
    
    if not 0 <= config['noise_filter']['usefulness_threshold'] <= 1:
        raise ValueError("usefulness_threshold must be between 0 and 1")
    
    return True

def print_config_summary(config: dict) -> None:
    """打印配置摘要"""
    print("Enhanced Atomic Note Generator Configuration Summary:")
    print(f"  Enhanced NER: {'Enabled' if config.get('enable_enhanced_ner') else 'Disabled'}")
    print(f"  Relation Extraction: {'Enabled' if config.get('enable_relation_extraction') else 'Disabled'}")
    print(f"  Enhanced Noise Filter: {'Enabled' if config.get('enable_enhanced_noise_filter') else 'Disabled'}")
    print(f"  Note Similarity: {'Enabled' if config.get('enable_note_similarity') else 'Disabled'}")
    print(f"  NER Model: {config['ner']['model_name']}")
    print(f"  Similarity Model: {config['similarity']['model_name']}")
    print(f"  Similarity Threshold: {config['similarity']['similarity_threshold']}")
    print(f"  Usefulness Threshold: {config['noise_filter']['usefulness_threshold']}")
    print(f"  Batch Size: {config.get('batch_size', 32)}")
    print(f"  GPU Enabled: {'Yes' if config.get('use_gpu') else 'No'}")