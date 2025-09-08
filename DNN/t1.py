import os
import subprocess


def run_script(script_path):
    folder = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)

    print(f"执行脚本: {script_name}")

    # 切换到脚本所在目录并执行
    current_dir = os.getcwd()
    os.chdir(folder)
    # 使用特定的Python解释器
    subprocess.run(["D:\\hongyouting\\py_code\\climate\\Scripts\\python.exe", script_name], check=True)
    os.chdir(current_dir)

# 按顺序列出要执行的脚本（使用完整路径）
scripts = [
    r"D:\hongyouting\py_code\DNN\1\cdr1\main.py",
    # r"D:\hongyouting\py_code\DNN\1\chirps1\main.py",
    # r"D:\hongyouting\py_code\DNN\1\cmorph1\main.py",
    # r"D:\hongyouting\py_code\DNN\1\gsmap_mvkg1\main.py",
    # r"D:\hongyouting\py_code\DNN\1\gsmap_nrt1\main.py",
    # r"D:\hongyouting\py_code\DNN\1\gsmap_nrtg1\main.py",
    # r"D:\hongyouting\py_code\DNN\1\imerg_e71\main.py",
    # r"D:\hongyouting\py_code\DNN\1\imerg_f71\main.py"
]

print("开始按顺序执行Python脚本...")

for script in scripts:
    try:
        run_script(script)
    except subprocess.CalledProcessError as e:
        print(f"脚本 {script} 执行失败: {e}")
        # 如果您希望一个脚本失败后继续执行下一个脚本，请取消下面的注释
        # continue
        # 如果您希望一个脚本失败后停止执行，请使用下面的 break
        break

print("所有脚本执行完毕。")