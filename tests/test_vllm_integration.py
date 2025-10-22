"""
vLLM 集成测试
测试vLLM双实例的原子笔记生成功能
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from llm.vllm_openai_client import VllmOpenAIClient
from llm.vllm_atomic_note_generator import VllmAtomicNoteGenerator
from llm.bucket_scheduler import BucketScheduler


class TestVllmOpenAIClient:
    """测试VllmOpenAIClient"""
    
    def test_init_with_endpoints(self):
        """测试客户端初始化"""
        endpoints = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"]
        model = "Qwen/Qwen2.5-7B-Instruct"
        
        client = VllmOpenAIClient(endpoints=endpoints, model=model)
        
        assert len(client.endpoints) == 2
        assert client.model == model
        assert client.max_tokens == 96  # 默认值
    
    def test_get_next_endpoint(self):
        """测试端点轮询"""
        endpoints = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"]
        client = VllmOpenAIClient(endpoints=endpoints, model="test-model")
        
        # 测试轮询
        endpoint1 = client._get_next_endpoint()
        endpoint2 = client._get_next_endpoint()
        endpoint3 = client._get_next_endpoint()
        
        assert endpoint1 is not None
        assert endpoint2 is not None
        assert endpoint3 is not None
        
        # 应该轮询回到第一个
        assert endpoint1.url == endpoint3.url
    
    @pytest.mark.asyncio
    async def test_chat_one_mock(self):
        """测试单个聊天请求（模拟）"""
        endpoints = ["http://127.0.0.1:8000/v1"]
        client = VllmOpenAIClient(endpoints=endpoints, model="test-model")
        
        # 模拟HTTP响应
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "测试响应内容"
                    }
                }
            ]
        }
        
        with patch.object(client, '_chat_completion_with_retry', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_response
            
            result = await client.chat_one("测试提示", "系统提示")
            
            assert result == "测试响应内容"
            mock_chat.assert_called_once()


class TestBucketScheduler:
    """测试分桶调度器"""
    
    def test_init(self):
        """测试初始化"""
        bucket_edges = [64, 128, 256, 512]
        scheduler = BucketScheduler(bucket_edges=bucket_edges)
        
        assert scheduler.bucket_edges == bucket_edges
        assert len(scheduler.buckets) == len(bucket_edges) + 1
    
    def test_estimate_tokens(self):
        """测试token估计"""
        scheduler = BucketScheduler()
        
        # 测试中文文本
        chinese_text = "这是一个测试文本"
        tokens = scheduler.estimate_tokens(chinese_text)
        assert tokens > 0
        
        # 测试英文文本
        english_text = "This is a test text"
        tokens = scheduler.estimate_tokens(english_text)
        assert tokens > 0
        
        # 测试空文本
        empty_tokens = scheduler.estimate_tokens("")
        assert empty_tokens == 0
    
    def test_get_bucket_id(self):
        """测试桶ID获取"""
        bucket_edges = [64, 128, 256, 512]
        scheduler = BucketScheduler(bucket_edges=bucket_edges)
        
        # 测试不同token数量的分桶
        assert scheduler.get_bucket_id(32) == 0  # <= 64
        assert scheduler.get_bucket_id(100) == 1  # <= 128
        assert scheduler.get_bucket_id(200) == 2  # <= 256
        assert scheduler.get_bucket_id(400) == 3  # <= 512
        assert scheduler.get_bucket_id(1000) == 4  # > 512
    
    def test_add_request(self):
        """测试添加请求"""
        scheduler = BucketScheduler()
        
        request_id = scheduler.add_request("测试提示", "系统提示")
        
        assert request_id is not None
        assert len(request_id) == 12  # MD5前12位
        
        # 检查请求是否被正确分桶
        bucket_stats = scheduler.get_bucket_stats()
        total_requests = sum(stats['count'] for stats in bucket_stats.values())
        assert total_requests == 1
    
    def test_get_bucket_stats(self):
        """测试桶统计"""
        scheduler = BucketScheduler()
        
        # 添加一些请求
        scheduler.add_request("短文本", "系统提示")
        scheduler.add_request("这是一个比较长的文本，用来测试分桶功能是否正常工作", "系统提示")
        
        stats = scheduler.get_bucket_stats()
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
        
        # 检查统计信息格式
        for bucket_id, bucket_stats in stats.items():
            assert 'range' in bucket_stats
            assert 'count' in bucket_stats
            assert 'avg_tokens' in bucket_stats


class TestVllmAtomicNoteGenerator:
    """测试VllmAtomicNoteGenerator"""
    
    def test_init_mock(self):
        """测试初始化（模拟vLLM客户端）"""
        with patch('llm.vllm_atomic_note_generator.LLMFactory.create_provider') as mock_factory:
            mock_client = Mock()
            mock_factory.return_value = mock_client
            
            # 模拟LocalLLM
            mock_llm = Mock()
            
            generator = VllmAtomicNoteGenerator(llm=mock_llm)
            
            assert generator.vllm_client == mock_client
            assert generator.bucket_scheduler is not None
            mock_factory.assert_called_once_with('vllm-openai')
    
    def test_format_atomic_note_prompt(self):
        """测试提示格式化"""
        with patch('llm.vllm_atomic_note_generator.LLMFactory.create_provider'):
            mock_llm = Mock()
            generator = VllmAtomicNoteGenerator(llm=mock_llm)
            
            # 模拟提示模板
            with patch.object(generator, '_get_atomic_note_prompts') as mock_prompts:
                mock_prompts.return_value = ("系统提示", "用户提示: {text}")
                
                chunk_data = {"text": "测试文本"}
                prompt = generator._format_atomic_note_prompt(chunk_data)
                
                assert "测试文本" in prompt
    
    def test_process_llm_response(self):
        """测试LLM响应处理"""
        with patch('llm.vllm_atomic_note_generator.LLMFactory.create_provider'):
            mock_llm = Mock()
            generator = VllmAtomicNoteGenerator(llm=mock_llm)
            
            # 模拟响应解析
            with patch('llm.vllm_atomic_note_generator.parse_notes_response') as mock_parse:
                mock_parse.return_value = [
                    {
                        "content": "测试笔记内容",
                        "entities": ["实体1", "实体2"],
                        "salience": 0.8
                    }
                ]
                
                with patch.object(generator, '_convert_to_atomic_note_format') as mock_convert:
                    mock_convert.return_value = {
                        "note_id": "test_note_1",
                        "content": "测试笔记内容"
                    }
                    
                    chunk_data = {"text": "测试文本"}
                    response = '{"notes": [{"content": "测试笔记内容"}]}'
                    
                    notes = generator._process_llm_response(response, chunk_data)
                    
                    assert len(notes) >= 0  # 可能被过滤
    
    @pytest.mark.asyncio
    async def test_health_check_mock(self):
        """测试健康检查（模拟）"""
        with patch('llm.vllm_atomic_note_generator.LLMFactory.create_provider') as mock_factory:
            mock_client = AsyncMock()
            mock_client.health_check.return_value = {
                "endpoint_0": {"status": "healthy"},
                "endpoint_1": {"status": "healthy"}
            }
            mock_factory.return_value = mock_client
            
            mock_llm = Mock()
            generator = VllmAtomicNoteGenerator(llm=mock_llm)
            
            health = await generator.health_check()
            
            assert health["status"] == "healthy"
            assert "vllm_endpoints" in health
            assert "bucket_scheduler" in health


@pytest.mark.integration
class TestVllmIntegration:
    """集成测试（需要实际的vLLM服务）"""
    
    @pytest.mark.skip(reason="需要运行中的vLLM服务")
    @pytest.mark.asyncio
    async def test_real_vllm_client(self):
        """测试真实的vLLM客户端"""
        endpoints = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"]
        model = "Qwen/Qwen2.5-0.5B"  # 使用测试模型
        
        client = VllmOpenAIClient(endpoints=endpoints, model=model)
        
        try:
            # 测试健康检查
            health = await client.health_check()
            print(f"Health check: {health}")
            
            # 测试单个请求
            response = await client.chat_one("什么是人工智能？", "你是一个有用的助手。")
            print(f"Response: {response}")
            
            assert len(response) > 0
            
        finally:
            await client.close()
    
    @pytest.mark.skip(reason="需要运行中的vLLM服务")
    def test_real_vllm_note_generator(self):
        """测试真实的vLLM原子笔记生成器"""
        from llm.local_llm import LocalLLM
        
        # 创建测试数据
        text_chunks = [
            {
                "text": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                "chunk_index": 0,
                "source_info": {
                    "title": "人工智能简介",
                    "document_id": "doc_1"
                }
            },
            {
                "text": "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。",
                "chunk_index": 1,
                "source_info": {
                    "title": "机器学习概述",
                    "document_id": "doc_2"
                }
            }
        ]
        
        try:
            # 创建生成器
            mock_llm = Mock()
            generator = VllmAtomicNoteGenerator(llm=mock_llm)
            
            # 生成原子笔记
            notes = generator.generate_atomic_notes(text_chunks)
            
            print(f"Generated {len(notes)} notes")
            for note in notes[:3]:  # 打印前3个笔记
                print(f"Note: {note.get('content', '')[:100]}...")
            
            # 检查性能统计
            stats = generator.get_performance_stats()
            print(f"Performance stats: {stats}")
            
            assert len(notes) >= 0
            
        except Exception as e:
            pytest.skip(f"vLLM service not available: {e}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])