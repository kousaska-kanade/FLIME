# Example4 -- Kendall correlation coefficient
from scipy.stats.stats import kendalltau
import numpy as np

# Tau_b
from scipy.stats import pearsonr

f = open(r"D:\PyCharm 2021.1.3\pycharmproject\tiny-imagenet-200\tiny-imagenet-200\val\stableresult\result1.txt","r")
lines = f.readlines()
s = []
count = 0
for line in lines:
    line=line.strip('\n')# 删除\n
    #print(line)
    #s.append([])
    #for i in range(5):
    result3 = line.split(' ')
    s.append(result3)
    # = count + 1
print(s)
print(len(s))
print(s[1][0])
sum = 0.0
count = 0
for i in range(99):
    for j in range(99):
        dat1 = np.array([int(s[i][0]),int(s[i][1]),int(s[i][2]),int(s[i][3]),int(s[i][4])])
        dat2 = np.array([int(s[j][0]),int(s[j][1]),int(s[j][2]),int(s[j][3]),int(s[j][4])])
        pccs = pearsonr(dat1, dat2)
        count = count + 1
        sum = sum + pccs[0]
print(sum)
print(count)
# dat1 = np.array([3,5,1,9,7,2,8,4,6])
# dat2 = np.array([5,3,2,6,8,1,7,9,4])

# c = 0
# d = 0
# t_x = 0
# t_y = 0
# for i in range(len(dat1)):
#     for j in range(i + 1, len(dat1)):
#         if (dat1[i] - dat1[j]) * (dat2[i] - dat2[j]) > 0:
#             c = c + 1
#         elif (dat1[i] - dat1[j]) * (dat2[i] - dat2[j]) < 0:
#             d = d + 1
#         else:
#             if (dat1[i] - dat1[j]) == 0 and (dat2[i] - dat2[j]) != 0:
#                 t_x = t_x + 1
#             elif (dat1[i] - dat1[j]) != 0 and (dat2[i] - dat2[j]) == 0:
#                 t_y = t_y + 1
#
# tau_b = (c - d) / np.sqrt((c + d + t_x) * (c + d + t_y))
#
# print(pccs = np.corrcoef(x, y))