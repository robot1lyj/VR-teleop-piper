import pickle

# 替换为你的pkl文件路径
file_path = "/home/lyj/lerobot_gello/.cache/calibration.gello_piper_right.pkl"
try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print(data)
except FileNotFoundError:
    print(f"文件 {file_path} 不存在。")
except pickle.UnpicklingError as e:
    print(f"读取pkl文件时出错: {e}")
