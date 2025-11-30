import subprocess
import sys
import os

# 2. path to python.exe
python_exe = os.path.join(sys.prefix,'bin', 'python.exe')
# 3. upgrade pip
subprocess.call([python_exe, "-m", "ensurepip"])
#subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
# 4. install required packages
subprocess.call([python_exe, "-m", "pip", "install", "opencv-python"])
#注：若没有换pip源可将此句改为
#subprocess.call([python_exe, "-m", "pip", "install", "package_name",  "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
