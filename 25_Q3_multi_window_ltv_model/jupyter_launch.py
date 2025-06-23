import os
import subprocess

# 切换工作目录
os.chdir(r"E:\git_repo\learning_step\25_Q3_multi_window_ltv_model")

# 启动 Jupyter Notebook
subprocess.run([
    "jupyter",
    "notebook",
    "--ip", "127.0.0.1"
])
