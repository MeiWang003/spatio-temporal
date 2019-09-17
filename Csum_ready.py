# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:07:01 2019
@author: Wangmei
"""

import pandas as pd
import numpy as np
import datetime

def C_zd(starttime, endtime, df2, df1):

    # Converts the string to date format
    starttime = datetime.datetime.strptime(starttime, "%Y/%m/%d %H:%M:%S") 
    endtime = datetime.datetime.strptime(endtime, "%Y/%m/%d %H:%M:%S") 
    # print(type(starttime))

    # under time limitation  pd.to_datetime -- Convert the string to date format
    df2 = df2[(pd.to_datetime(df2["date"]) >= starttime) & (
                pd.to_datetime(df2["date"]) <= endtime)]  

    #  dereplication,locations
    list1 = []
    d1 = list(df1["序号"])
    d1 = set(d1)  # set to repeat
    for c in d1:  # Save the deduplicated data set to the list1
        list1.append(c)
    list1.sort()  # sorted 
    print('list1',list1)

    #  dereplication,locations
    list2 = []
    d2 = list(df2["date"])
    # print('d2',d2)
    d2 = set(d2)  
    print('list2',d2)

    for c in d2:
        list2.append(c)

    def get_list(date): 
        return datetime.datetime.strptime(date, "%Y/%m/%d %H:%M")

    list2 = sorted(list2, key=lambda date: get_list(date))  # date 排序
    # print('list2',list2)

    # Two-dimensional array,initilazation
    arr = []
    for i in range(len(d1)):
        arr.append([])
        for j in range(len(d2)):
            arr[i].append(0)
    # print('arr',arr)

    df2 = df2.groupby(['z_id', 'date'], as_index=False).sum()

    # Assignment
    for i in range(len(d1)):
        for j in range(len(d2)):
            list3 = datetime.datetime.strptime(list2[j], "%Y/%m/%d %H:%M")  # 字符串转化为date形式
            # print('list3',list3)
            a = df2[(df2["z_id"] == list1[i]) & ((pd.to_datetime(df2["date"])) == list3)]["num"].tolist()  # 选取满足条件的值

            if a == []:  # if there is no data,fill it with 0
                arr[i][j] = 0
            else:
                arr[i][j] = int(a[0])  # else,fill it with a[0]
    # print('arr',arr)
    arr = np.array(arr)  #  Converts the list to array
    m, n = arr.shape  # m rows and n columns

    # the sum of observed cases
    C_sum = arr.sum()
    # print(C_sum)
    # C_z indicates the total of cases on the area z,1 indicates the row
    C_z = arr.sum(axis=1)

    # C_d indicates the total number of cases on the time interval d,0 indicates the columns
    C_d = arr.sum(axis=0)

    # Calculate the expected value
    u_zd = np.multiply(C_d, np.mat(C_z).T) / C_sum
    u_zd = np.array(u_zd)  

    return arr, C_sum, u_zd, C_z, C_d

