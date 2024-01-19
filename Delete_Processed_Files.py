import os

# 路径1和路径2
path1 = "test3/data"
path2 = "test3/output"

# 获取路径2下文件名称前四个字母的集合
path2_filenames = set()
for filename in os.listdir(path2):
    if len(filename) >= 4:
        path2_filenames.add(filename[:4])

# 删除路径1下与路径2下文件名称前四个字母相同的文件
for filename in os.listdir(path1):
    if len(filename) >= 4 and filename[:4] in path2_filenames:
        file_path = os.path.join(path1, filename)
        os.remove(file_path)
        print("已删除文件:", file_path)