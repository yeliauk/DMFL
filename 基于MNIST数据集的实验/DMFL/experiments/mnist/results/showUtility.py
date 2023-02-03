# 下标从1开始
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams

print(matplotlib.matplotlib_fname())
print(matplotlib.get_cachedir())

# 计算合同
# num = 9 # 数字 参与方数目
num = 10 # 数字 参与方数目


c = 50  # 成本
# 填写数据质量
# theta = [0, 0.150, 0.200, 0.250, 0.350, 0.500, 0.700, 0.950, 1.300, 1.600]  # 数字 9个参与方
theta = [0, 0.100, 0.150, 0.200, 0.250, 0.350, 0.500, 0.650, 0.850, 1.200, 1.750]  # 数字 10个参与方



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
# 设置figure_size尺寸
# plt.rcParams['figure.figsize'] = (1.5748, 1.5748)
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
    plt.vlines([i], -2500, U[i][i], colors='black', linestyles='dashed', linewidth=0.5) # 画垂直线
    ax.scatter(i, U[i][i], s=7, color = 'black', zorder=2)  # 画散点图标记特定点
plt.legend(legendList, loc='lower left', edgecolor='none')  # 标识放在最佳位置不带边框,背景为白色仍保留
xTicks = np.arange(1, num+1, 1)
plt.xticks(xTicks, fontsize=8)
yTicks = np.arange(-2500, 200, 100)
plt.yticks(yTicks, fontsize=8)
# 显示网格线
plt.grid(zorder=0, alpha=0.3)
plt.show()

#---------------------------参与方数目不同且各参与方选择为其设计的合同类型时的全局效用展示图---------------------------
#------------------------------------------------MNIST-------------------------------------------------------------
MNIST_cnum = [8, 9, 10, 11]  # MNIST实验参与方数目 x坐标
MNIST_Us = [0, 18.338474984375004, 27.85065414062499, 0]  # MNIST实验参与方数目为9和10时，全局模型效用 y坐标
x_data = [" ", "9", "10", "  "]

# 控制x轴为整数
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# 颜色
colorList = ["dodgerblue", "dodgerblue", "dodgerblue", "dodgerblue", "lightseagreen", "dodgerblue", "slateblue", "palevioletred", "sienna", "black"]

# 画图，plt.bar()可以画柱状图
for i in range(len(MNIST_cnum)):
    plt.bar(x_data[i], MNIST_Us[i], width=0.5, color=colorList[i], linewidth = 0.5, zorder=2, edgecolor='black')
# 显示网格线
plt.grid(zorder=0, alpha = 0.3)
plt.grid(axis='x') # 设置 x 轴方向显示网格线
# 设置x轴标签名
plt.xlabel('参与方数目')
# 设置y轴标签名
plt.ylabel('全局效用')
plt.xticks(fontsize=8) #  刻度字体大小
yTicks = np.arange(0, 35, 5)
plt.yticks(yTicks, fontsize=8) #  刻度字体大小
# 显示
plt.show()