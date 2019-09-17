
"""
    approach：Output all likelihood and P value, as the input of NI-scan model

    caculate log-likelihood ratio :
    LGLR = C_zd * math.log(C_zd / u_zd) + (C_sum - C_zd) * math.log(
          (C_sum - C_zd) / (C_sum - u_zd))
	  
    caculate P value with Monte Carlo algorithm 

    @author: Wangmei
    @date: 2019/05/16
"""

import pandas as pd
import distance
import Csum_ready
import copy
import math
import operator
import time
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import gini_process_v3

np.set_printoptions(threshold=np.nan)
# Set the display style
pd.set_option('display.width', 1000)  
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  


def space_cluster(df1, R_max):
    d = []  # save distance data

    # Get the latitude and longitude that meets the conditions
    def JD(k):  # Define longitude
        a = df1[df1.序号 == k].经度
        return a.tolist()[0]

    def WD(k):  # Defining latitude
        b = df1[df1.序号 == k].纬度
        # print(b)
        return b.tolist()[0]

    m, n = df1.shape
    print('m', 'n', m, n)
    # Computing space aggregation area
    for i in range(m):
        d.append([])
        for j in range(m):
            # print(JD(i+1),WD(i+1),JD(j+1),WD(j+1))
            d[i].append(distance.getDistance(JD(i + 1), WD(i + 1), JD(j + 1), WD(j + 1)))  # Calculate the distance between the current location and another locations
        # print('i',i,','"d[i]",d )

        #print(d)

        for k in range(m):
            # print(d[i][k])
            if (d[i][k] <= R_max):  
                c = {}
                c["center_xbid"] = i + 1
                c["r"] = d[i][k]
                xbid = []  # save the array subscript

                #print('d[i][k]',d[i][k])
                for v in range(m):  
                    # print(d[i][v])
                    #xbid = list(set(xbid))

                    if (d[i][k] >= d[i][v]):
                        if (i != k):
                            xbid.append(v + 1)  
                        else:
                            xbid.append(i + 1)  
                    else:
                        continue

                #if i == 170:
                   # print('xbid-171', xbid)
                xbid = list(set(xbid))
                c["xbids"] = xbid
                circle.append(c)  # save form [{ }{ }{ }]
            else:
                continue
    # print(circle)
    # print('d', d)
    return circle

def scan(case, circle, zdsjc, U, day):  # Scan result calculation function

    C = copy.deepcopy(case)  # events matrix
    circle = copy.deepcopy(circle) 
    T_max = copy.deepcopy(zdsjc)  # the Maximum time cluster
    U = copy.deepcopy(U)  # expected matrix
    # print('c',C)
    # print('type(U)',type(U))
    # day = 7
    cir = []
    # cir = {}
    for T_cluster in range(0, T_max):  
        # print("T_cluster")
        # print(T_cluster)
        circle1 = copy.deepcopy(circle)
        for i in range(len(circle1)):
            circle1[i]["T_cluster"] = T_cluster + 1  # time cluster
            circle1[i]["C_zd"] = ""
            circle1[i]["u_zd"] = ""
            C_zd = 0
            u_zd = 0

            xb_zd = []
            y_zd = []

            for k in circle1[i]["xbids"]:
                k_zd = 0  # initilization
                yc_zd = 0  

                # print(i)
                k = k - 1
                j = day - 1
                while j >= day - 1 - T_cluster:
                    k_zd += C[k][j]
                    yc_zd += U[k][j]

                    C_zd += C[k][j]
                    u_zd += U[k][j]
                    j -= 1
                xb_zd.append(k_zd)
                y_zd.append(yc_zd)  # Add a subscript corresponding to the predicted value

            u_zd = float(u_zd)
            # print("C_zd:{}".format(C_zd))
            if C_zd <= u_zd:
                LGLR = 0  # Actual case is less than expected, normal no warning
            else:
                LGLR = C_zd * math.log(C_zd / u_zd) + (C_sum - C_zd) * math.log(
                    (C_sum - C_zd) / (C_sum - u_zd))  # lilelihood ratio
            circle1[i]["LGLR"] = LGLR  #in circle1[i],Add likelihood ratio dictionary
            circle1[i]["C_zd"] = C_zd
            circle1[i]["u_zd"] = u_zd
            circle1[i]["xb_zd"] = xb_zd
            circle1[i]["yc_zd"] = y_zd
            circle1[i]["Risk"] = C_zd / u_zd

        # a = "cir"+ str(T_cluster)
        # print(T_cluster)
        # cir[T_cluster] = circle1
        # print(cir)
        # print(type(circle))
        # print("-----------------------------------")
        # (T_cluster)
        # print('circle1',circle1)
        cir = cir + circle1
 
    # transfer df --》 -- 》 remove r = 0 ,LGLR = 0 --》 sort --》 same value,removing weight --》to list
    df_cir = pd.DataFrame(cir)

    # First, sorted by LGLR ，then sorted by r
    df_cir = df_cir.sort_index(axis=0, ascending=[False, True], by=['LGLR', 'r'])

    return df_cir



# Knuth-Durstenfeld Shuffle algorithm
def randpermBySYB(p1):
    p = copy.copy(p1)
    p = np.array(p)
    n = np.size(p)

    for i in range(n):
        # print('i',i)
        j = np.random.randint(n - i)  # 产生1到n-1范围内伪随机整数
        # print('j',j)
        tmp = p[i + j]
        p[i + j] = p[i]
        p[i] = tmp  # 交换数据
    return p


# Random rearrangement
def shuffle(v):
    v = np.array(v)
    # print(v[0])
    m, n = v.shape
    # print('m,',m)
    for i in range(m):
        # print('v[i]',v[i])
        v[i] = randpermBySYB(v[i])
    return v

if __name__ == '__main__':

 
    data_dz = r"data\city_jjs.csv"
    data_case = r"data\jjs_quan_data.csv"
    # read location file
    df1 = pd.DataFrame(pd.read_csv(data_dz))
    # read case occure file
    df2 = pd.DataFrame(pd.read_csv(data_case))

    # m the sum locations 
    m, n = df1.shape

    # initilization
    R_max = 5  # the maximum
    T_max = 3  # the maximum time cluster

    space_xbis = []  # save all space clusters

    circle = []  # save each scan windown center,radius and space subscript collections 

    starttime = "2018/1/8 0:00:00"  
    endtime = "2018/1/14 0:00:00"  

    starttime1 = time.mktime(time.strptime(starttime, '%Y/%m/%d %H:%M:%S'))  
    endtime1 = time.mktime(time.strptime(endtime,'%Y/%m/%d %H:%M:%S'))  
    work_days = int((endtime1 - starttime1) / (24 * 60 * 60))  
    day = work_days + 1  # Reference days
  
    C, C_sum, U, C_z, C_d = Csum_ready.C_zd(starttime, endtime, df2, df1)  # caculate C,C_sum, u

    # Distance space array after judgment
    circle = space_cluster(df1, R_max)

    # print('C', C)
    # print('C_sum',C_sum)
    # print('C_d',C_d)
    # print('U',U)
    # print('circle', circle)

    result_init = scan(C, circle, T_max, U, day)  # original date's result
    #print("result_init")
    #print(result_init)

    result_init = np.array(result_init)
    result_init = result_init.tolist()
    # print('result_init',result_init)

    fzcs = 999  # simulation number
    c_re = np.zeros((1, day))  # initilization
    c_re = c_re.astype(np.int)
    np.random.seed(0)  # seed

    # save the final result 
    zzjg = []

    count = 0
    fz = []
    # print('c[1]', c[1])

    for x in result_init:
        c = x
        for s in range(fzcs):
            c_re = shuffle(C)

            # caculate expected value 
            c_re = np.array(c_re)  # list turn to array
            c_re_sum = c_re.sum()  # C sum
            c_re_z = c_re.sum(axis=1)  
            c_re_d = c_re.sum(axis=0)  
            u_re_zd = np.multiply(c_re_d, np.mat(c_re_z).T) / c_re_sum  
            u_re_zd = np.array(u_re_zd)  

            # Monte Carlo algorithm
            df_cir = scan(c_re, circle, T_max, u_re_zd, day)
            df_cir = df_cir.iloc[0:1]
            df_cir = np.array(df_cir)
            df_cir = df_cir.tolist()

            if df_cir[0][1] > c[1]:
                count += 1
                # print(df_cir[0][1])
            if count > 50:
                break
            else:
                fz.append(df_cir)
        zh = [0] * (len(fz) + 1)
        zh[0] = x
        for i in range(1, len(zh)):
            zh[i] = fz[i - 1][0]
        zh = sorted(zh, key=lambda x: x[1], reverse=True)

        no = zh.index(x)
        x.append((no + 1) / len(zh))
        zzjg.append(x)


    # transfer to dataFrame
    zzjg = pd.DataFrame(zzjg, columns=['C_zd', 'LGLR','risk', 'T_cluster', 'center_xbid', 'r', 'u_zd', 'xb_zd','xbids','y_zd','P-value'])

    # zzjg.to_csv(r'.\temp\39\dup_data39(3.1-3.8_10_quan).csv')
    zzjg.to_csv(r'.\output\(1.8-1.14_5_quan).csv')
    print(zzjg)











