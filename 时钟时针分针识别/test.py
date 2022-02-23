# -*- codeing= utf-8 -*-
# @Time : 2021/11/30 21:39
# @Author : 郑中天
# @File : test.py
# @Software : PyCharm
import numpy as np
ratio_test = np.array([-1,-2,1,1,1,3,4])
# Transform the ratio test result into all positive
rt_positive = np.where(ratio_test >= 0, ratio_test, np.inf)
pivot_row_index = np.where(rt_positive == rt_positive.min())[0][0]
print(pivot_row_index)
print(ratio_test[1:])