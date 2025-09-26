import os
import tempfile
import shutil
import sys
sys.path.append('.')
from main_musique import create_item_workdir

# 创建临时目录测试
temp_base = tempfile.mkdtemp()
print(f'临时工作目录: {temp_base}')

# 测试debug模式
debug_dir = create_item_workdir(temp_base, 'test_item', debug_mode=True)
print(f'Debug模式目录: {debug_dir}')

# 测试非debug模式  
normal_dir = create_item_workdir(temp_base, 'test_item', debug_mode=False)
print(f'普通模式目录: {normal_dir}')

# 检查路径是否正确
expected_debug = os.path.join(temp_base, 'debug', 'item_test_item')
expected_normal = os.path.join(temp_base, 'temp_item_test_item')

print(f'Debug路径正确: {debug_dir == expected_debug}')
print(f'普通路径正确: {normal_dir == expected_normal}')
print(f'Debug目录存在: {os.path.exists(debug_dir)}')
print(f'普通目录存在: {os.path.exists(normal_dir)}')

# 清理
shutil.rmtree(temp_base)
print('测试完成，临时目录已清理')
