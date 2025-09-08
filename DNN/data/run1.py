import time
import subprocess

# 定义要按指定顺序运行的.py文件列表
py_files = [
    'rep1.py',
    'rep2.py',
    'rep3.py',
    'rep4.py',
    'rep5.py'
]  # 将这里替换为你想要的顺序

# 逐个运行并记录时间
for py_file in py_files:
    print(f"runing: {py_file}")
    start_time = time.time()  # 记录开始时间
    subprocess.run(['python', py_file])  # 运行.py文件
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"{py_file} 的运行时间为: {elapsed_time} 秒")
