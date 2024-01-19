import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 指定字体文件和路径
font_path = "/home/liaochuyue/.fonts/times.ttf"

# 创建字体属性对象
custom_font = FontProperties(fname=font_path)

# 创建一个简单的图例
plt.plot([1, 2, 3, 4], label='Example Line')
legend = plt.legend()


# 创建一个简单的图例，同时设置字体大小
plt.plot([1, 2, 3, 4], label='Example Line')
plt.legend(prop=custom_font, fontsize=20)

# 在图形中显示绘图
plt.show()
