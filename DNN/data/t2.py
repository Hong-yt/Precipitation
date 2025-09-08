import os
import subprocess
import time


def run_script(script_path):
    """
    运行指定的Python脚本
    """
    print(f"\n开始运行脚本: {script_path}")
    start_time = time.time()

    try:
        # 使用指定的Python解释器运行脚本
        python_path = r"D:/hongyouting/py_code/climate/Scripts/python.exe"
        result = subprocess.run([python_path, script_path],
                                capture_output=True,
                                text=True)

        # 打印输出
        print(result.stdout)

        # 如果有错误，打印错误信息
        if result.stderr:
            print("错误信息:")
            print(result.stderr)

        # 检查返回码
        if result.returncode == 0:
            print(f"脚本 {script_path} 运行成功")
        else:
            print(f"脚本 {script_path} 运行失败，返回码: {result.returncode}")

    except Exception as e:
        print(f"运行脚本 {script_path} 时发生错误: {str(e)}")

    end_time = time.time()
    print(f"运行时间: {end_time - start_time:.2f} 秒")
    print("-" * 50)


def main():
    # 定义要运行的脚本列表
    scripts = [
        '50_clip.py',
        'world_clip1.py',
        '50-60-85_clip.py',
        'world_clip.py'
    ]

    print("开始批量运行脚本...")
    print("=" * 50)

    # 运行每个脚本
    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"脚本文件不存在: {script}")

    print("\n所有脚本运行完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()