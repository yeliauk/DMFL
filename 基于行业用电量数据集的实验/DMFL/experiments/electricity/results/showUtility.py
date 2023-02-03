# 下标从1开始
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams

print(matplotlib.matplotlib_fname())
print(matplotlib.get_cachedir())


# 计算合同

num = 9  # 电 参与方数目
# num = 50  # 电 参与方数目
# num = 100  # 电 参与方数目

c = 50  # 成本

# 填写数据质量
theta = [0, 0.778, 0.786, 0.794, 0.806, 0.825, 0.876, 0.897, 0.93, 0.938]  # 电 9个参与方
'''theta = [0, 0.508, 0.508, 0.521, 0.699, 0.731, 0.755, 0.781, 0.871, 0.872, 0.875,  #电50个参与方
         0.877, 0.880, 0.883, 0.884, 0.885, 0.890, 0.894, 0.895, 0.896, 0.896,
         0.897, 0.897, 0.898, 0.900, 0.901, 0.903, 0.904, 0.905, 0.907, 0.908,
         0.909, 0.909, 0.910, 0.910, 0.911, 0.912, 0.919, 0.928, 0.930, 0.930,
         0.935, 0.943, 0.961, 0.962, 0.963, 0.971, 0.980, 0.986, 0.996, 0.996]'''
'''theta = [0, 0.546, 0.546, 0.642, 0.782, 0.788, 0.871, 0.875, 0.886, 0.897, 0.904,  # 电100个参与方
         0.909, 0.933, 0.934, 0.935, 0.936, 0.936, 0.938, 0.939, 0.939, 0.940, 
         0.940, 0.942, 0.944, 0.945, 0.945, 0.945, 0.945, 0.945, 0.945, 0.946, 
         0.946, 0.946, 0.947, 0.947, 0.948, 0.948, 0.948, 0.948, 0.949, 0.949, 
         0.950, 0.950, 0.950, 0.950, 0.951, 0.951, 0.951, 0.951, 0.952, 0.952, 
         0.952, 0.952, 0.952, 0.954, 0.956, 0.956, 0.956, 0.960, 0.960, 0.961, 
         0.961, 0.962, 0.962, 0.962, 0.963, 0.963, 0.964, 0.964, 0.964, 0.966, 
         0.967, 0.967, 0.969, 0.970, 0.972, 0.974, 0.974, 0.975, 0.975, 0.976, 
         0.976, 0.976, 0.977, 0.978, 0.979, 0.979, 0.980, 0.984, 0.984, 0.985, 
         0.986, 0.987, 0.987, 0.992, 0.994, 0.994, 0.997, 0.998, 0.999, 0.999]'''


print(f"数据质量theta：{theta}")
# 计算模型标准线
M = [0] * (num + 1)
for i in range(1, num+1):
    M[i] = theta[i] / 20
print(f"模型标准线M：{M}")

# 计算奖励
R = [0] * (num + 1)
for i in range(1, num+1):
    R[i] = M[i] * M[i] * 4000
print(f"合同奖励R:{R}")

# 计算注册费
f = [0] * (num + 1)
f[1] = (1 / (2 * c)) * pow((theta[1]*R[1]), 2)
for i in range(2, num+1):
    f[i] = (1 / (2 * c)) * pow((theta[i] * R[i]), 2) - (1 / (2 * c)) * pow((theta[i] * R[i - 1]), 2) + f[i - 1]
print(f"合同注册费f:{f}")


# 计算参与方效用
U = [0] * (num + 1)
for i in range(1, num+1):
    U[i] = [0] * (num + 1) # 每个参与方选择不同类型合同的效用列表
    for j in range(1, num+1):
        eij = (1 / c) * theta[i] * R[j]
        if(i == j):
            print(f"参与方{i}的训练意愿ei为:{eij}")
        U[i][j] = 100 * (theta[i] * eij * R[j] - f[j] - (c / 2) * pow(eij, 2))
    print(f"参与方{i}选择不同类型合同的效用为U{i}：{U[i]}")

# 计算全局的效用
Us = 0
for i in range(1, num+1):
    ei = (1 / c) * theta[i] * R[i]
    Us = Us + (f[i] + theta[i] * ei * (M[i] * M[i] * 4000 - R[i]))
print(f"全局的效用为Us：{Us}")


# -----------------------------------------------画图--------------------------------------------------------------------
# 格式设置
config = {
    "font.family": 'serif', # 衬线字体
    "font.size": 8, # 相当于6号字大小
    #"font.serif": ['SimSun'], # 宋体
    "font.serif": ['SimHei'], # 宋体
    "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    #"mathtext.fontset": ['Euclid'],
    'axes.unicode_minus': False, # 处理负号，即-号
    'figure.figsize': (4.0, 4.0),
}
rcParams.update(config)
# 设置坐标轴的粗细
ax=plt.gca()  #获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(0.5)  #设置底部坐标轴的粗细  # 单位是磅吗？
ax.spines['left'].set_linewidth(0.5)  #设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(0.5)  #设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(0.5)  #设置上部坐标轴的粗细

zhfont = matplotlib.font_manager.FontProperties(fname="D:\\pythonProjects\\venv\\Lib\\site-packages\\matplotlib\\mpl-data\\fonts\\ttf\\SimSun.ttf")  # 导入字体


#-----------------------每个参与方选择不同类型合同时的参与方效用展示图-----------------------------
contract_type = range(1, num+1)  # x坐标

plt.xlabel(u'合同类型')
plt.ylabel(u'参与方效用（×$ \mathit{10}^{-2}$）')
colorList = ["snow", "red", "darkorange", "gold", "forestgreen", "lightseagreen", "dodgerblue", "slateblue", "palevioletred", "sienna", "black"]
legendList = []
for i in range(1, num+1, 2):
    Ui = U[i][1:] # 第i个参与方选择不同类型合同时的效用 y坐标
    plt.plot(contract_type, Ui, color=colorList[i], linewidth=1, zorder=1)    # 绘制曲线图
    legendList.append(f'参与方{i}')
    plt.vlines([i], -7, U[i][i], colors='black', linestyles='dashed', linewidth=0.5)  # 画垂直线  用电量
    ax.scatter(i, U[i][i], s=7, color = 'black', zorder=2)  # 画散点图标记特定点
plt.legend(legendList, loc='upper left', edgecolor='none')  # 标识放在最佳位置不带边框,背景为白色仍保留  用电量
xTicks = np.arange(1, num+1, 1)
plt.xticks(xTicks, fontsize=8)
yTicks = np.arange(-8, 21, 1)  # 用电量
plt.yticks(yTicks, fontsize=8)
# 显示网格线
plt.grid(zorder=0, alpha=0.3)
plt.show()

#---------------------------参与方数目不同且各参与方选择为其设计的合同类型时的全局效用展示图---------------------------
#------------------------------------------------行业用电量-------------------------------------------------------------
elec_cnum = [9, 50, 100]  # 行业用电量实验参与方数目
elec_Us = [3.1010763056571418, 19.590226533414075, 53.854273163020544]  # 行业用电量实验参与方数目为9、50、100时，全局模型效用
x_data = [f"{elec_cnum[i]}" for i in range(0, len(elec_cnum))]

# 控制x轴为整数
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# 颜色
colorList = ["dodgerblue", "dodgerblue", "dodgerblue", "dodgerblue", "lightseagreen", "dodgerblue", "slateblue", "palevioletred", "sienna", "black"]
# 间距

# 画图，plt.bar()可以画柱状图
for i in range(len(elec_cnum)):
    plt.bar(x_data[i], elec_Us[i], width=0.4, color=colorList[i], linewidth = 0.5, zorder=10, edgecolor='black')
# 显示网格线
plt.grid(zorder=0, alpha = 0.3)
plt.grid(axis='x') # 设置 x 轴方向显示网格线
# 设置x轴标签名
plt.xlabel('参与方数目')
# 设置y轴标签名
plt.ylabel('全局效用')
plt.xticks(fontsize=8) #  刻度字体大小
yTicks = np.arange(0, 70, 10)
plt.yticks(yTicks, fontsize=8) #  刻度字体大小
# 显示
plt.show()