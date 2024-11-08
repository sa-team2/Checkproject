import schedule
import time
import os
import subprocess
import sys

file_path = sys.argv[1]  # 第一个参数是脚本名，第二个是传入的文件路径
print(f"接收到的文件路径: {file_path}")

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 svm.py 的绝对路径
svm_script_path = os.path.join(current_dir, 'svm.py')

# 使用 subprocess 运行 svm.py
subprocess.run(['python', svm_script_path])

schedule.run_pending()
time.sleep(1)
