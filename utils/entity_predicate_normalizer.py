#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实体与谓词标准化模块

实现功能：
1. 实体名称标准化与别名管理
2. 谓词标准化与映射规则
3. 基于规则和相似度的标准化
4. 动态别名词典维护
5. 标准化结果缓存
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import difflib
from pathlib import Path

try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    logging.warning("fuzzywuzzy not available, using basic string similarity")

logger = logging.getLogger(__name__)

@dataclass
class NormalizationRule:
    """标准化规则"""
    pattern: str                        # 匹配模式（正则表达式）
    replacement: str                    # 替换内容
    rule_type: str = "regex"           # 规则类型：regex, exact, fuzzy
    confidence: float = 1.0             # 规则置信度
    description: str = ""               # 规则描述
    enabled: bool = True                # 是否启用
    
    def apply(self, text: str) -> Tuple[str, float]:
        """
        应用规则
        
        Args:
            text: 输入文本
            
        Returns:
            (标准化结果, 置信度)
        """
        if not self.enabled:
            return text, 0.0
        
        try:
            if self.rule_type == "exact":
                if text == self.pattern:
                    return self.replacement, self.confidence
                return text, 0.0
            
            elif self.rule_type == "regex":
                if re.search(self.pattern, text, re.IGNORECASE):
                    result = re.sub(self.pattern, self.replacement, text, flags=re.IGNORECASE)
                    return result, self.confidence
                return text, 0.0
            
            elif self.rule_type == "fuzzy" and FUZZYWUZZY_AVAILABLE:
                similarity = fuzz.ratio(text.lower(), self.pattern.lower()) / 100.0
                if similarity >= 0.8:  # 相似度阈值
                    return self.replacement, similarity * self.confidence
                return text, 0.0
            
            else:
                return text, 0.0
                
        except Exception as e:
            logger.warning(f"Rule application failed: {e}")
            return text, 0.0

@dataclass
class AliasEntry:
    """别名条目"""
    canonical_name: str                 # 标准名称
    aliases: Set[str] = field(default_factory=set)  # 别名集合
    confidence_scores: Dict[str, float] = field(default_factory=dict)  # 别名置信度
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    last_updated: float = 0.0           # 最后更新时间
    
    def add_alias(self, alias: str, confidence: float = 1.0) -> None:
        """添加别名"""
        self.aliases.add(alias)
        self.confidence_scores[alias] = confidence
    
    def get_best_match(self, query: str) -> Tuple[str, float]:
        """获取最佳匹配"""
        if query == self.canonical_name:
            return self.canonical_name, 1.0
        
        if query in self.aliases:
            return self.canonical_name, self.confidence_scores.get(query, 0.8)
        
        # 模糊匹配
        if FUZZYWUZZY_AVAILABLE:
            all_names = [self.canonical_name] + list(self.aliases)
            best_match, score = process.extractOne(query, all_names)
            if score >= 80:  # 相似度阈值
                return self.canonical_name, score / 100.0
        
        return query, 0.0

class EntityNormalizer:
    """实体标准化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('entity_normalizer', {})
        
        # 配置参数
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.enable_fuzzy_matching = self.config.get('enable_fuzzy_matching', True) and FUZZYWUZZY_AVAILABLE
        self.cache_size = self.config.get('cache_size', 10000)
        
        # 别名词典
        self.alias_dict: Dict[str, AliasEntry] = {}
        
        # 标准化规则
        self.normalization_rules: List[NormalizationRule] = []
        
        # 缓存
        self.normalization_cache: Dict[str, Tuple[str, float]] = {}
        
        # 统计信息
        self.stats = {
            'total_normalizations': 0,
            'cache_hits': 0,
            'rule_applications': 0,
            'fuzzy_matches': 0
        }
        
        # 加载配置
        self._load_default_rules()
        self._load_alias_dict()
        
        logger.info(f"EntityNormalizer initialized: {len(self.normalization_rules)} rules, "
                   f"{len(self.alias_dict)} aliases, fuzzy={self.enable_fuzzy_matching}")
    
    def normalize(self, entity: str) -> Tuple[str, float]:
        """
        标准化实体名称
        
        Args:
            entity: 原始实体名称
            
        Returns:
            (标准化名称, 置信度)
        """
        if not entity or not entity.strip():
            return entity, 0.0
        
        entity = entity.strip()
        self.stats['total_normalizations'] += 1
        
        # 1. 检查缓存
        if entity in self.normalization_cache:
            self.stats['cache_hits'] += 1
            return self.normalization_cache[entity]
        
        # 2. 应用标准化规则
        normalized_entity = entity
        max_confidence = 0.0
        
        for rule in self.normalization_rules:
            result, confidence = rule.apply(normalized_entity)
            if confidence > max_confidence:
                normalized_entity = result
                max_confidence = confidence
                self.stats['rule_applications'] += 1
        
        # 3. 别名词典查找
        alias_result, alias_confidence = self._find_in_alias_dict(normalized_entity)
        if alias_confidence > max_confidence:
            normalized_entity = alias_result
            max_confidence = alias_confidence
        
        # 4. 模糊匹配
        if self.enable_fuzzy_matching and max_confidence < self.min_confidence:
            fuzzy_result, fuzzy_confidence = self._fuzzy_match(normalized_entity)
            if fuzzy_confidence > max_confidence:
                normalized_entity = fuzzy_result
                max_confidence = fuzzy_confidence
                self.stats['fuzzy_matches'] += 1
        
        # 5. 缓存结果
        result = (normalized_entity, max_confidence)
        if len(self.normalization_cache) < self.cache_size:
            self.normalization_cache[entity] = result
        
        return result
    
    def add_alias(self, canonical_name: str, alias: str, confidence: float = 1.0) -> None:
        """
        添加别名
        
        Args:
            canonical_name: 标准名称
            alias: 别名
            confidence: 置信度
        """
        if canonical_name not in self.alias_dict:
            self.alias_dict[canonical_name] = AliasEntry(canonical_name=canonical_name)
        
        self.alias_dict[canonical_name].add_alias(alias, confidence)
        
        # 清除相关缓存
        keys_to_remove = [k for k in self.normalization_cache.keys() if k == alias]
        for key in keys_to_remove:
            del self.normalization_cache[key]
    
    def add_rule(self, rule: NormalizationRule) -> None:
        """添加标准化规则"""
        self.normalization_rules.append(rule)
        # 清除缓存
        self.normalization_cache.clear()
    
    def _load_default_rules(self) -> None:
        """加载默认规则"""
        default_rules = [
            # 公司名称标准化
            NormalizationRule(
                pattern=r'(.+?)(?:有限公司|有限责任公司|股份有限公司|集团有限公司)',
                replacement=r'\1',
                rule_type="regex",
                confidence=0.9,
                description="公司后缀移除"
            ),
            NormalizationRule(
                pattern=r'(.+?)(?:Co\.?\s*Ltd\.?|Corporation|Corp\.?|Inc\.?|LLC)',
                replacement=r'\1',
                rule_type="regex",
                confidence=0.9,
                description="英文公司后缀移除"
            ),
            
            # 地名标准化
            NormalizationRule(
                pattern=r'(.+?)(?:市|省|县|区|自治区|特别行政区)',
                replacement=r'\1',
                rule_type="regex",
                confidence=0.8,
                description="行政区划后缀移除"
            ),
            
            # 人名标准化
            NormalizationRule(
                pattern=r'(.+?)(?:先生|女士|教授|博士|院士|主席|总裁|CEO)',
                replacement=r'\1',
                rule_type="regex",
                confidence=0.7,
                description="人名称谓移除"
            ),
            
            # 空格和标点标准化
            NormalizationRule(
                pattern=r'\s+',
                replacement=' ',
                rule_type="regex",
                confidence=1.0,
                description="多余空格移除"
            ),
            NormalizationRule(
                pattern=r'[\u3000\xa0]',  # 全角空格和不间断空格
                replacement=' ',
                rule_type="regex",
                confidence=1.0,
                description="特殊空格标准化"
            )
        ]
        
        # 从外部配置文件加载标准化规则
        normalization_rules_file = self.config.get('normalization_rules_file', '')
        if normalization_rules_file:
            try:
                import yaml
                from pathlib import Path
                
                config_path = Path(normalization_rules_file)
                if not config_path.is_absolute():
                    # 相对路径，相对于项目根目录
                    config_path = Path(__file__).parent.parent / config_path
                
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        external_config = yaml.safe_load(f)
                    
                    if external_config and 'normalization_rules' in external_config:
                        rules_config = external_config['normalization_rules']
                        
                        # 处理每个规则类别
                        for rule_category, rule_info in rules_config.items():
                            if not rule_info.get('enabled', True):
                                continue
                            
                            confidence = rule_info.get('confidence', 1.0)
                            patterns = rule_info.get('patterns', [])
                            
                            for pattern in patterns:
                                # 根据规则类别确定替换方式
                                if rule_category in ['company_suffixes', 'location_suffixes', 'person_titles']:
                                    replacement = r'\1'  # 提取第一个捕获组
                                elif rule_category == 'punctuation':
                                    if pattern == "\\s+":
                                        replacement = ' '
                                    elif pattern == "[，。；：！？、]":
                                        replacement = ''
                                    else:
                                        replacement = ''
                                else:
                                    replacement = ''
                                
                                rule = NormalizationRule(
                                    pattern=pattern,
                                    replacement=replacement,
                                    rule_type="regex",
                                    confidence=confidence,
                                    description=f"{rule_category} rule",
                                    enabled=True
                                )
                                default_rules.append(rule)
                                logger.debug(f"Added rule from external config: {pattern} -> {replacement}")
                        
                        logger.info(f"Loaded normalization rules from {config_path}")
                else:
                    logger.warning(f"Normalization rules file not found: {config_path}")
            except Exception as e:
                logger.error(f"Failed to load normalization rules file {normalization_rules_file}: {e}")
        
        # 从配置加载自定义规则
        custom_rules = self.config.get('custom_rules', [])
        for rule_config in custom_rules:
            rule = NormalizationRule(
                pattern=rule_config.get('pattern', ''),
                replacement=rule_config.get('replacement', ''),
                rule_type=rule_config.get('rule_type', 'regex'),
                confidence=rule_config.get('confidence', 1.0),
                description=rule_config.get('description', ''),
                enabled=rule_config.get('enabled', True)
            )
            default_rules.append(rule)
        
        self.normalization_rules.extend(default_rules)
        
        # 添加多语言实体别名映射
        self._load_multilingual_aliases()
    
    def _load_multilingual_aliases(self) -> None:
        """加载多语言实体别名映射"""
        # Portuguese/葡萄牙相关实体别名
        portuguese_aliases = {
            "Portugal": ["葡萄牙", "Portuguese Republic", "República Portuguesa", "葡萄牙共和国"],
            "Lisbon": ["里斯本", "Lisboa", "葡萄牙首都"],
            "Porto": ["波尔图", "波图", "Oporto"],
            "Portuguese": ["葡萄牙语", "葡语", "Português"],
            "Macau": ["澳门", "Macao", "澳门特别行政区", "Região Administrativa Especial de Macau"],
            "Brazil": ["巴西", "Brasil", "巴西联邦共和国", "República Federativa do Brasil"],
            "Angola": ["安哥拉", "安哥拉共和国", "República de Angola"],
            "Mozambique": ["莫桑比克", "莫桑比克共和国", "República de Moçambique"]
        }
        
        # Myanmar/缅甸相关实体别名
        myanmar_aliases = {
            "Myanmar": ["缅甸", "Burma", "缅甸联邦共和国", "Republic of the Union of Myanmar", "မြန်မာ"],
            "Yangon": ["仰光", "Rangoon", "ရန်ကုန်"],
            "Naypyidaw": ["内比都", "奈比多", "နေပြည်တော်", "缅甸首都"],
            "Mandalay": ["曼德勒", "မန္တလေး"],
            "Burmese": ["缅甸语", "缅语", "မြန်မာဘာသာ"],
            "Rohingya": ["罗兴亚", "罗兴亚人", "Rohingya people"],
            "Shan State": ["掸邦", "ရှမ်းပြည်နယ်"],
            "Kachin State": ["克钦邦", "ကချင်ပြည်နယ်"]
        }
        
        # Laos/老挝相关实体别名
        laos_aliases = {
            "Laos": ["老挝", "Lao PDR", "老挝人民民主共和国", "Lao People's Democratic Republic", "ລາວ"],
            "Vientiane": ["万象", "永珍", "ວຽງຈັນ", "老挝首都"],
            "Luang Prabang": ["琅勃拉邦", "ຫຼວງພະບາງ"],
            "Lao": ["老挝语", "老语", "ພາສາລາວ"],
            "Mekong": ["湄公河", "澜沧江", "ແມ່ນ້ຳຂອງ"],
            "Champasak": ["占巴塞", "ຈຳປາສັກ"],
            "Savannakhet": ["沙湾拿吉", "ສະຫວັນນະເຂດ"]
        }
        
        # 其他东南亚国家别名
        other_sea_aliases = {
            "Thailand": ["泰国", "泰王国", "Kingdom of Thailand", "ประเทศไทย", "สยาม"],
            "Bangkok": ["曼谷", "กรุงเทพฯ", "泰国首都"],
            "Vietnam": ["越南", "Việt Nam", "越南社会主义共和国"],
            "Hanoi": ["河内", "Hà Nội", "越南首都"],
            "Ho Chi Minh City": ["胡志明市", "西贡", "Thành phố Hồ Chí Minh"],
            "Cambodia": ["柬埔寨", "Kampuchea", "柬埔寨王国", "ព្រះរាជាណាចក្រកម្ពុជា"],
            "Phnom Penh": ["金边", "ភ្នំពេញ", "柬埔寨首都"],
            "Singapore": ["新加坡", "新加坡共和国", "Republic of Singapore"],
            "Malaysia": ["马来西亚", "Malaysia", "马来西亚联邦"],
            "Kuala Lumpur": ["吉隆坡", "马来西亚首都"],
            "Indonesia": ["印度尼西亚", "印尼", "Indonesia", "印度尼西亚共和国"],
            "Jakarta": ["雅加达", "Jakarta", "印尼首都"],
            "Philippines": ["菲律宾", "Philippines", "菲律宾共和国"],
            "Manila": ["马尼拉", "Manila", "菲律宾首都"]
        }
        
        # 合并所有别名映射
        all_aliases = {}
        all_aliases.update(portuguese_aliases)
        all_aliases.update(myanmar_aliases)
        all_aliases.update(laos_aliases)
        all_aliases.update(other_sea_aliases)
        
        # 添加到别名词典
        for canonical_name, aliases in all_aliases.items():
            if canonical_name not in self.alias_dict:
                self.alias_dict[canonical_name] = AliasEntry(canonical_name=canonical_name)
            
            for alias in aliases:
                self.alias_dict[canonical_name].add_alias(alias, 0.95)
        
        logger.info(f"Loaded {len(all_aliases)} multilingual entity aliases covering Portuguese, Myanmar, Laos and other SEA countries")
    
    def _load_alias_dict(self) -> None:
        """加载别名词典"""
        alias_file = self.config.get('alias_dict_file', '')
        if alias_file and Path(alias_file).exists():
            try:
                with open(alias_file, 'r', encoding='utf-8') as f:
                    alias_data = json.load(f)
                
                for canonical_name, aliases_info in alias_data.items():
                    entry = AliasEntry(canonical_name=canonical_name)
                    
                    if isinstance(aliases_info, list):
                        for alias in aliases_info:
                            entry.add_alias(alias, 0.9)
                    elif isinstance(aliases_info, dict):
                        for alias, confidence in aliases_info.items():
                            entry.add_alias(alias, confidence)
                    
                    self.alias_dict[canonical_name] = entry
                
                logger.info(f"Loaded {len(self.alias_dict)} alias entries from {alias_file}")
            except Exception as e:
                logger.error(f"Failed to load alias dict from {alias_file}: {e}")
    
    def _find_in_alias_dict(self, entity: str) -> Tuple[str, float]:
        """在别名词典中查找"""
        for entry in self.alias_dict.values():
            result, confidence = entry.get_best_match(entity)
            if confidence >= self.min_confidence:
                return result, confidence
        
        return entity, 0.0
    
    def _fuzzy_match(self, entity: str) -> Tuple[str, float]:
        """模糊匹配"""
        if not FUZZYWUZZY_AVAILABLE:
            return entity, 0.0
        
        # 收集所有候选名称
        candidates = []
        for entry in self.alias_dict.values():
            candidates.append(entry.canonical_name)
            candidates.extend(entry.aliases)
        
        if not candidates:
            return entity, 0.0
        
        # 查找最佳匹配
        try:
            best_match, score = process.extractOne(entity, candidates)
            if score >= 80:  # 相似度阈值
                # 找到对应的标准名称
                for entry in self.alias_dict.values():
                    if best_match == entry.canonical_name or best_match in entry.aliases:
                        return entry.canonical_name, score / 100.0
        except Exception as e:
            logger.warning(f"Fuzzy matching failed: {e}")
        
        return entity, 0.0
    
    def save_alias_dict(self, file_path: str) -> None:
        """保存别名词典"""
        try:
            alias_data = {}
            for canonical_name, entry in self.alias_dict.items():
                alias_data[canonical_name] = entry.confidence_scores
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(alias_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved alias dict to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save alias dict to {file_path}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        cache_hit_rate = self.stats['cache_hits'] / max(self.stats['total_normalizations'], 1)
        
        return {
            'total_normalizations': self.stats['total_normalizations'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': cache_hit_rate,
            'rule_applications': self.stats['rule_applications'],
            'fuzzy_matches': self.stats['fuzzy_matches'],
            'rules_count': len(self.normalization_rules),
            'aliases_count': len(self.alias_dict),
            'cache_size': len(self.normalization_cache)
        }

class PredicateNormalizer:
    """谓词标准化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('predicate_normalizer', {})
        
        # 配置参数
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.enable_fuzzy_matching = self.config.get('enable_fuzzy_matching', True) and FUZZYWUZZY_AVAILABLE
        
        # 谓词映射表
        self.predicate_mapping: Dict[str, str] = {}
        
        # 谓词分类
        self.predicate_categories: Dict[str, List[str]] = {}
        
        # 缓存
        self.normalization_cache: Dict[str, Tuple[str, float]] = {}
        
        # 统计信息
        self.stats = {
            'total_normalizations': 0,
            'cache_hits': 0,
            'mapping_hits': 0,
            'fuzzy_matches': 0
        }
        
        # 加载配置
        self._load_predicate_mapping()
        self._load_predicate_categories()
        
        logger.info(f"PredicateNormalizer initialized: {len(self.predicate_mapping)} mappings, "
                   f"{len(self.predicate_categories)} categories")
    
    def normalize(self, predicate: str) -> Tuple[str, float]:
        """
        标准化谓词
        
        Args:
            predicate: 原始谓词
            
        Returns:
            (标准化谓词, 置信度)
        """
        if not predicate or not predicate.strip():
            return predicate, 0.0
        
        predicate = predicate.strip()
        self.stats['total_normalizations'] += 1
        
        # 1. 检查缓存
        if predicate in self.normalization_cache:
            self.stats['cache_hits'] += 1
            return self.normalization_cache[predicate]
        
        # 2. 精确映射
        if predicate in self.predicate_mapping:
            result = (self.predicate_mapping[predicate], 1.0)
            self.stats['mapping_hits'] += 1
        
        # 3. 模糊匹配
        elif self.enable_fuzzy_matching:
            fuzzy_result, fuzzy_confidence = self._fuzzy_match_predicate(predicate)
            if fuzzy_confidence >= self.min_confidence:
                result = (fuzzy_result, fuzzy_confidence)
                self.stats['fuzzy_matches'] += 1
            else:
                result = (predicate, 0.0)
        
        else:
            result = (predicate, 0.0)
        
        # 4. 缓存结果
        self.normalization_cache[predicate] = result
        
        return result
    
    def add_mapping(self, original: str, normalized: str) -> None:
        """添加谓词映射"""
        self.predicate_mapping[original] = normalized
        
        # 清除相关缓存
        if original in self.normalization_cache:
            del self.normalization_cache[original]
    
    def get_category(self, predicate: str) -> Optional[str]:
        """获取谓词分类"""
        normalized_predicate, _ = self.normalize(predicate)
        
        for category, predicates in self.predicate_categories.items():
            if normalized_predicate in predicates:
                return category
        
        return None
    
    def _load_predicate_mapping(self) -> None:
        """加载谓词映射"""
        # 默认映射
        default_mapping = {
            # 中文谓词映射
            '创立': 'founded_by',
            '成立': 'founded_by',
            '建立': 'founded_by',
            '创建': 'founded_by',
            '创办': 'founded_by',
            
            '位于': 'located_in',
            '在': 'located_in',
            '坐落于': 'located_in',
            '地处': 'located_in',
            
            '属于': 'member_of',
            '隶属于': 'member_of',
            '归属于': 'member_of',
            
            '拥有': 'owns',
            '持有': 'owns',
            '控制': 'controls',
            '管理': 'manages',
            
            '担任': 'serves_as',
            '任职': 'serves_as',
            '就职于': 'works_at',
            '工作于': 'works_at',
            
            '毕业于': 'graduated_from',
            '就读于': 'studied_at',
            '学习于': 'studied_at',
            
            '发行': 'published_by',
            '出版': 'published_by',
            '发布': 'released_by',
            
            # 英文谓词映射
            'founded': 'founded_by',
            'established': 'founded_by',
            'created': 'founded_by',
            
            'located': 'located_in',
            'situated': 'located_in',
            
            'belongs to': 'member_of',
            'part of': 'member_of',
            
            'owns': 'owns',
            'possesses': 'owns',
            'controls': 'controls',
            
            'works at': 'works_at',
            'employed by': 'works_at',
            
            'graduated from': 'graduated_from',
            'studied at': 'studied_at',
            
            'published by': 'published_by',
            'released by': 'released_by'
        }
        
        # 从配置加载自定义映射
        custom_mapping = self.config.get('custom_mapping', {})
        
        # 从外部配置文件加载自定义映射
        custom_config_file = self.config.get('custom_config_file', '')
        if custom_config_file:
            try:
                import yaml
                from pathlib import Path
                
                config_path = Path(custom_config_file)
                if not config_path.is_absolute():
                    # 相对路径，相对于项目根目录
                    config_path = Path(__file__).parent.parent / config_path
                
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        external_config = yaml.safe_load(f)
                    
                    if external_config and 'custom_mapping' in external_config:
                        custom_mapping.update(external_config['custom_mapping'])
                        logger.info(f"Loaded custom predicate mappings from {config_path}")
                else:
                    logger.warning(f"Custom config file not found: {config_path}")
            except Exception as e:
                logger.error(f"Failed to load custom config file {custom_config_file}: {e}")
        
        self.predicate_mapping.update(default_mapping)
        self.predicate_mapping.update(custom_mapping)
    
    def _load_predicate_categories(self) -> None:
        """加载谓词分类"""
        default_categories = {
            'organizational': ['founded_by', 'member_of', 'owns', 'controls', 'manages'],
            'geographical': ['located_in', 'situated_in'],
            'personal': ['works_at', 'serves_as', 'graduated_from', 'studied_at'],
            'publication': ['published_by', 'released_by', 'distributed_by'],
            'temporal': ['occurred_on', 'started_at', 'ended_at'],
            'causal': ['caused_by', 'resulted_in', 'led_to']
        }
        
        # 从配置加载自定义分类
        custom_categories = self.config.get('predicate_categories', {})
        
        # 从外部配置文件加载自定义分类
        custom_config_file = self.config.get('custom_config_file', '')
        if custom_config_file:
            try:
                import yaml
                from pathlib import Path
                
                config_path = Path(custom_config_file)
                if not config_path.is_absolute():
                    # 相对路径，相对于项目根目录
                    config_path = Path(__file__).parent.parent / config_path
                
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        external_config = yaml.safe_load(f)
                    
                    if external_config and 'predicate_categories' in external_config:
                        custom_categories.update(external_config['predicate_categories'])
                        logger.info(f"Loaded custom predicate categories from {config_path}")
                else:
                    logger.warning(f"Custom config file not found: {config_path}")
            except Exception as e:
                logger.error(f"Failed to load custom config file {custom_config_file}: {e}")
        
        self.predicate_categories.update(default_categories)
        self.predicate_categories.update(custom_categories)
    
    def _fuzzy_match_predicate(self, predicate: str) -> Tuple[str, float]:
        """谓词模糊匹配"""
        if not FUZZYWUZZY_AVAILABLE:
            return predicate, 0.0
        
        candidates = list(self.predicate_mapping.keys())
        if not candidates:
            return predicate, 0.0
        
        try:
            best_match, score = process.extractOne(predicate, candidates)
            if score >= 80:  # 相似度阈值
                normalized = self.predicate_mapping[best_match]
                return normalized, score / 100.0
        except Exception as e:
            logger.warning(f"Predicate fuzzy matching failed: {e}")
        
        return predicate, 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        cache_hit_rate = self.stats['cache_hits'] / max(self.stats['total_normalizations'], 1)
        
        return {
            'total_normalizations': self.stats['total_normalizations'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': cache_hit_rate,
            'mapping_hits': self.stats['mapping_hits'],
            'fuzzy_matches': self.stats['fuzzy_matches'],
            'mappings_count': len(self.predicate_mapping),
            'categories_count': len(self.predicate_categories),
            'cache_size': len(self.normalization_cache)
        }

class EntityPredicateNormalizer:
    """实体与谓词标准化器统一接口"""
    
    def __init__(self, config: Dict[str, Any]):
        from utils.config_loader_helper import load_external_config, merge_configs
        
        # 尝试从外部配置文件加载配置
        config_file_path = config.get('entity_predicate_normalizer_config_file')
        
        if config_file_path:
            try:
                external_config = load_external_config(config_file_path)
                # 合并外部配置和内联配置
                self.config = merge_configs(external_config, config)
                logger.info(f"Loaded EntityPredicateNormalizer config from {config_file_path}")
            except Exception as e:
                logger.warning(f"Failed to load external config {config_file_path}: {e}, using inline config")
                self.config = config
        else:
            self.config = config
        
        # 初始化子组件
        self.entity_normalizer = EntityNormalizer(self.config)
        self.predicate_normalizer = PredicateNormalizer(self.config)
        
        logger.info("EntityPredicateNormalizer initialized")
    
    def normalize_entity(self, entity: str) -> Tuple[str, float]:
        """标准化实体"""
        return self.entity_normalizer.normalize(entity)
    
    def normalize_predicate(self, predicate: str) -> Tuple[str, float]:
        """标准化谓词"""
        return self.predicate_normalizer.normalize(predicate)
    
    def normalize_triple(self, subject: str, predicate: str, object_: str) -> Tuple[Tuple[str, float], Tuple[str, float], Tuple[str, float]]:
        """
        标准化三元组
        
        Args:
            subject: 主语实体
            predicate: 谓词
            object_: 宾语实体
            
        Returns:
            ((标准化主语, 置信度), (标准化谓词, 置信度), (标准化宾语, 置信度))
        """
        norm_subject = self.normalize_entity(subject)
        norm_predicate = self.normalize_predicate(predicate)
        norm_object = self.normalize_entity(object_)
        
        return norm_subject, norm_predicate, norm_object
    
    def add_entity_alias(self, canonical_name: str, alias: str, confidence: float = 1.0) -> None:
        """添加实体别名"""
        self.entity_normalizer.add_alias(canonical_name, alias, confidence)
    
    def add_predicate_mapping(self, original: str, normalized: str) -> None:
        """添加谓词映射"""
        self.predicate_normalizer.add_mapping(original, normalized)
    
    def get_predicate_category(self, predicate: str) -> Optional[str]:
        """获取谓词分类"""
        return self.predicate_normalizer.get_category(predicate)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'entity_normalizer': self.entity_normalizer.get_stats(),
            'predicate_normalizer': self.predicate_normalizer.get_stats()
        }
    
    def save_alias_dict(self, file_path: str) -> None:
        """保存别名词典"""
        self.entity_normalizer.save_alias_dict(file_path)

# 便利函数
def create_entity_predicate_normalizer(config: Dict[str, Any]) -> EntityPredicateNormalizer:
    """创建实体谓词标准化器实例"""
    return EntityPredicateNormalizer(config)

def normalize_entities_and_predicates(entities: List[str], 
                                    predicates: List[str],
                                    normalizer: Optional[EntityPredicateNormalizer] = None,
                                    **kwargs) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    批量标准化实体和谓词的便利函数
    
    Args:
        entities: 实体列表
        predicates: 谓词列表
        normalizer: 标准化器实例
        **kwargs: 其他参数
        
    Returns:
        (标准化实体列表, 标准化谓词列表)
    """
    if normalizer is None:
        logger.warning("No normalizer provided, returning original entities and predicates")
        return [(e, 0.0) for e in entities], [(p, 0.0) for p in predicates]
    
    try:
        normalized_entities = [normalizer.normalize_entity(entity) for entity in entities]
        normalized_predicates = [normalizer.normalize_predicate(predicate) for predicate in predicates]
        
        return normalized_entities, normalized_predicates
    except Exception as e:
        logger.error(f"Batch normalization failed: {e}")
        return [(e, 0.0) for e in entities], [(p, 0.0) for p in predicates]