# -*- coding:utf-8 -*-
import csv
import numpy as np


# 传入文件编号，输出该文件用电量数据列的平均值
def meanData(num):
    filedir = 'E:\\elecData\\IC1811\\split1\\' # 用电量数据所在目录 E:\\elecData\\IC1811\\split1\\
                                                              # E:\\elecData\\IC1811\\split2\\
                                                              # E:\\elecData\\IC1811\\split3\\
    filename = filedir + str(num) +'.csv'
    with open(filename) as csv_file:
        row = csv.reader(csv_file, delimiter=',')  # 分隔符方式

        next(row)  # 读取首行
        elecData = []  # 创建一个数组来存储数据

        # 读取除首行以后每一行的第6列数据，并将其加入到数组elecData之中
        for r in row:
            elecData.append(float(r[5]))  # 将字符串数据转化为浮点型加入到数组之中
    meanValue = np.mean(elecData)
    return meanValue

# 传入各个参与方平均值列表，返回emd值列表
def emd(meanList):
    meanAll = np.mean(meanList) # 所有参与方的用电量数据平均 全局平均
    emdList = []
    for i in range(0, len(meanList)):
        emdList.append(abs(meanList[i] - meanAll))
    return emdList

# 传入emd值列表，对每个值进行归一化，返回归一化的emd值结果
def normalizationEmd(emdList):
    max_emdValue = max(emdList)  # 最大值
    min_emdValue = min(emdList)  # 最小值
    norEmdList = [] # 归一化结果
    for i in range(0, len(emdList)):
        norEmdList.append((emdList[i] - min_emdValue) / (max_emdValue - min_emdValue))
    return norEmdList

#计算各个参与方数据质量 1-emd（已归一化） 保留三位小数
def dataQuality(norEmdList):
    dataQualityList = []
    for i in range(0, len(norEmdList)):
        dataQualityList.append(round(1 - norEmdList[i], 3)) # 保留3位小数
        # dataQualityList.append(1 - norEmdList[i])
    return dataQualityList

clientNum = 9  # 参与方数量 9 50 100
meanList = [] # 各个参与方用电量数据列均值列表
for i in range(1,clientNum+1):
    meanList.append(meanData(i))
print('均值列表：', meanList)

emdList = emd(meanList)  # 计算得到emd值列表
print('emd值列表：', emdList)

norEmdList = normalizationEmd(emdList)  # 对emd值列表归一化
print('归一化emd值列表：', norEmdList)

dataQualityList = dataQuality(norEmdList) # 初始计算的数据质量（包含01）
sortdataQulityList = sorted(dataQualityList) # 初始数据质量的排序

# 对特殊值 0 1进行处理  处理得到最终的dataQualityList
maxDiff = 0
minDiff = 1
for i in range(1, len(sortdataQulityList)-3):
    if(sortdataQulityList[i+1] - sortdataQulityList[i] > maxDiff):
        maxDiff = sortdataQulityList[i+1] - sortdataQulityList[i]
    if (sortdataQulityList[i + 1] - sortdataQulityList[i] < minDiff):
        minDiff = sortdataQulityList[i + 1] - sortdataQulityList[i]
for i in range(len(dataQualityList)):
    if(dataQualityList[i] == 0):
        if(sortdataQulityList[1] - minDiff > 0):
            dataQualityList[i] = sortdataQulityList[1] - minDiff
        else:
            dataQualityList[i] = 0.001
    if(dataQualityList[i] == 1):
        if(sortdataQulityList[len(sortdataQulityList)-2] + minDiff < 1):
            dataQualityList[i] = sortdataQulityList[len(sortdataQulityList) - 2] + minDiff
        else:
            dataQualityList[i] = 0.999

print('数据质量：', dataQualityList)