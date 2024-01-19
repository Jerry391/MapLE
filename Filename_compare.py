import os
import xlwt

# 要处理的文件夹路径
folder_path = "test2/output"

# 获取文件名列表
file_names = os.listdir(folder_path)

# 创建一个新的Excel工作簿
workbook = xlwt.Workbook()
sheet = workbook.add_sheet("Output_Names")

# 写入文件名的前四个字母到Excel表格中
for index, file_name in enumerate(file_names):
    if len(file_name) >= 4:
        short_name = file_name[:4]
        sheet.write(index, 0, short_name)

# 保存Excel文件
xls_file_path = "file_names.xls"
workbook.save(xls_file_path)

print("文件名已保存到", xls_file_path)

# 要处理的文件夹路径
folder_path2 = "test2/data"

# 获取文件名列表
file_names2 = os.listdir(folder_path2)


# 创建一个新的Excel工作簿
workbook = xlwt.Workbook()
sheet = workbook.add_sheet("Input_Names")

# 写入文件名的前四个字母到Excel表格中
for index, file_name in enumerate(file_names2):
    if len(file_name) >= 4:
        short_name = file_name[:4]
        sheet.write(index, 0, short_name)

# 保存Excel文件
xls_file_path2 = "file_names2.xls"
workbook.save(xls_file_path2)

print("文件名已保存到2", xls_file_path2)
