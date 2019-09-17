
"""
    model : IN-scan
    amount of information calculated combining the Gini coefficient which obtained
    by using the mean value of the observed and expected values respectively of 
    each cluster with the significant statistics  calculated based on the likelihood 
    ratio scan model

    Gini:  1 - area_2.polygon_area(np.array(xb_new))/(d*d/2)
    NI : -math.log(1-gini)-math.log(p)

    @author: Wangmei
    @date: 2019/05/22
"""

import pandas as pd
import ast
import copy
import numpy as np
import area_2
import operator
import math

np.set_printoptions(threshold=np.nan)
# Set the display style
pd.set_option('display.width', 1000)  
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None)  


from collections import Counter  # Statistics package

def gini_xb(dic_center,xb):

    d = 200
    xb_new = []
    for x_b in xb:    
            xb_new.append(dic_center[x_b])
    xb_new.append([0,0])
    xb_new.append([d,d])

    xb_new = [value for index, value in sorted(enumerate(xb_new), key=lambda d: d[1])]
    gini_area = 1 - area_2.polygon_area(np.array(xb_new))/(d*d/2)

#print('gini_area',gini_area)
    return gini_area

def gini_result(df2,arr2):
    result = []
    list_xb  =  df2['xbids'].tolist()
    list_czd = df2['xb_zd'].tolist()
    list_uzd = df2['y_zd'].tolist()
    p_verse  =  df2['P-value'].tolist()
    p_verse = [1-l for l in p_verse]     # q = 1-p

    # String style list convert to list style
    for i in range(len(list_xb)):
        list_xb[i] = ast.literal_eval(list_xb[i])
        list_czd[i] = ast.literal_eval(list_czd[i])
        list_uzd[i] = ast.literal_eval(list_uzd[i])

    xb_czd = []
    xb_uzd = []

    for k in range(len(p_verse)):
         # print(list_czd[k])
         # print(p_verse[k])

         # caculate the mean of each location observed cases n*p
         list_czd[k] = [m * p_verse[k] for m in  list_czd[k]]
         list_uzd[k] = [n * p_verse[k] for n in list_uzd[k]]

         # the dictionary corresponding value
         xb_czd.append(dict(zip(list_xb[k], list_czd[k])))
         xb_uzd.append(dict(zip(list_xb[k], list_uzd[k])))

    #Supplement  xb_czd and xb_uzd missing key，its value is 0
    for k1 in arr2:
          for x1 in xb_czd:
             if k1 not in x1:
                 x1[k1] = 0

    for k2 in arr2:
        for x2 in xb_uzd:
            if k2 not in x2:
                x2[k2] = 0

    # In the two lists, accumulated the same data of the dictionary.
    xb_czd_all = [(k, sum([x[k] for x in xb_czd])) for k in arr2]
    xb_uzd_all = [(k, sum([x[k] for x in xb_uzd])) for k in arr2]

    #  dic - the points included corresponding to（zd,ud）
    dic = {}
    for m1 in range(len(xb_czd_all)):

        xx = []
        for m2 in range(len(xb_uzd_all)):
                # print('xb_czd_all[m]', xb_czd_all[m1][1])
                # print('xb_uzd_all[m]', xb_uzd_all[m2][1])
            if xb_czd_all[m1][0] == xb_uzd_all[m2][0]:
                if xb_czd_all[m1][1] < xb_uzd_all[m2][1]:
                    s1 = 0
                    s2 = 0
                else:
                    s1 = xb_czd_all[m1][1]
                    s2 = xb_uzd_all[m2][1]

                xx.extend([s1,s2])
                  # x = x.extend(xb_uzd_all[m2][1])
                dic[xb_czd_all[m1][0]]=xx
            else :
                continue

    return dic
  
def gini_result_zz(dic_center,df_deep,arr1,arr2,R_max,distance,t):
    #print(dic_center)
    # the distance between the two locations is less than the maximum radius,draw the circle with i center and k radius
    for d1 in range(len(arr1)):
        for d2 in range(len(arr2)):
            if(distance[arr1[d1]-1,arr2[d2]-1]<= R_max):  
                c = {}
                c["center_xbid"] = arr1[d1]
                c["r"] = distance[arr1[d1]-1,arr2[d2]-1]
                c['T_cluster'] = t

                xbid = []  # save the index
                cc_zd = []
                uu_zd = []
            # print(d[i][k])
	    # when drawing the circle with d1 center and r radius,Determine the location within the circle, store the subscript collection
                for d3 in range(len(arr2)):  
                    if (distance[arr1[d1]-1][arr2[d2]-1] >= distance[arr1[d1]-1][arr2[d3]-1]):
                        if (arr1[d1] != arr2[d2]):
                            xbid.append(arr2[d3])  # sapce clusters coleection
                            cc_zd.append(dic_center[arr2[d3]][0])
                            uu_zd.append(dic_center[arr2[d3]][1])

                        else:
                            xbid.append(arr1[d1])  # just save center
                            cc_zd.append(dic_center[arr1[d1]][0])
                            uu_zd.append(dic_center[arr2[d1]][1])

                    else:
                        continue
                c["xbids"] = xbid
                c["cc_zd"] = cc_zd
                c["uu_zd"] = uu_zd

                gini = gini_xb(dic_center,xbid)
                c["gini"] = gini
 
                xbid = sorted(xbid)    # list sorted from small to large
                db_xbid = str(xbid)

                df_copy = copy.deepcopy(df_deep)

                if df_copy[df_copy['xbids'] == db_xbid].empty:
                    p =1
                else:
                    df_equal = df_copy[(df_copy['center_xbid'] == arr1[d1]) & (df_copy['xbids'] == db_xbid) & (df_copy['T_cluster'] == t)]
                    p = df_equal['P-value'].tolist()[0]

                    # minimum population
                    c_zd = df_equal['C_zd'].tolist()[0]
                    LR = df_equal['LGLR'].tolist()[0]


                c["p_value"] = p
                c["C_zd"] = c_zd
                c["LGLR"] = LR

                if c_zd >= 2:
                    c['NI'] = -math.log(1-gini)-math.log(p)
                else:
                    c['NI'] = 0
                circle.append(c)
            else:
                continue

    return circle

def  dot(df):
    arr = df['xbids'].tolist()
    #print('arr',arr)
    for i in range(len(arr)):
        arr[i] = ast.literal_eval(arr[i])  
    # print(arr)
    for i in range(len(arr)):
        for j in arr[i]:
            if j not in arr[0]:
                arr[0].append(j)
   # print(arr[0])
    return  arr[0]

def gini_ssjg(df_reduce,df,df_copy,R_max,distance):

    # df1.to_csv("./temp/df1.csv",encoding="utf-8")
    center = df_reduce.iloc[0]['center_xbid']   # get center
    #print('center',center)
    df2 = copy.deepcopy(df)
    df2 = df2[(df2['center_xbid'] == center)]  # Take the repeat centers, such as the center of all 12

    T = df2['T_cluster'].tolist()
    # print('T',T)
    T = set(T)
    #print('T', list(T))

    ok2 = []
    for t in T:
        print(t)
        df1 = df[(df['T_cluster'] == t)]
        #df2 = copy.deepcopy(df)
        df1 = df1[(df1['center_xbid'] == center)]
       # print(df1)
        # obtain the subscript collection
        arr1 = dot(df1)  # The center repeats all the list of the following table, as analysis center
        #print('arr1',arr1)

        df1 = df[(df['center_xbid'].isin(arr1))&(df['T_cluster'] == t)] # 显示以选取下标为中心的全部结果
        #print('df1',df1)

        #  Display the points inside and outside the circle
        # print(df1)
        arr2 = dot(df1)
       # print('arr2',arr2)
        #df1.to_csv("./temp/df1.csv",encoding="utf-8")
        dic_center = gini_result(df1,arr2)  # center points ,n*p accumulated ，then return（c_zd，u_zd）
      #  print('dic_center1111111111',dic_center)
        ok2 += gini_result_zz(dic_center,df_copy,arr1,arr2,R_max,distance,t)

    return ok2,arr1
 

if __name__ == '__main__':
    # obtain distance
    df3 =pd.DataFrame(pd.read_csv(r'.\data\distance_12.csv'))
    #df3 = pd.DataFrame(pd.read_csv(r'.\data\distance_171.csv'))
    # caculate two-dimensional array
    distance = df3.values
    # delete the one column of two-dimensional array,np.delete(dataset,  Row/Column , axis=1)  axis 1 indicate column ,otherwise 0 is raw
    distance = np.delete(distance, 0, 1)

    # read dataset
    #df =  pd.DataFrame(pd.read_csv(r'.\temp\171\dup_data171(5_quan).csv'))
    #df =  pd.DataFrame(pd.read_csv(r'.\temp\39\dup_data39(1.8-1.14_5_quan).csv'))
    df =  pd.DataFrame(pd.read_csv(r'.\output\1\(1.8-1.14_5_quan).csv'))

    # update df[xnids] sorted from small to large
    df_xbids = df['xbids'].tolist()
   # print('df_xbids',df_xbids)
    for i in range(len(df_xbids)):
        #print('ast',sorted(ast.literal_eval(df_xbids[i])))
        df_xbids[i] = str(sorted(ast.literal_eval(df_xbids[i])))

   # print('df_xbids23',df_xbids)

    df123 = pd.DataFrame([])
    df123['xb'] = df_xbids
    df['xbids'] = df123['xb']

    df_copy = copy.deepcopy(df)
    df_reduce = copy.deepcopy(df)

    df_result = []
    df_result= pd.DataFrame(df_result)  # initialization
    count= 0
    aaa = 1
    while df_reduce.empty == False:
        circle = []
        #print('df_reduce111111111')

        #print(df_reduce)
        gn,arr = gini_ssjg(df_reduce,df,df_copy,5,distance)
        pd_gn = pd.DataFrame(gn)
        pd_gn = pd_gn[~pd_gn['C_zd'].isin([1]) ]    #delete the case only with one occur

        # pd_gn = pd_gn.sort_index(axis = 0 ,ascending = [False,True],by = ['gini','r'])
        pd_gn = pd_gn.sort_index(axis = 0 ,ascending = [False,True],by = ['NI','r'])
       # pd_gn.to_csv(str(aaa)+'.csv')
        aaa += 1

        df_result = df_result.append(pd_gn.iloc[[0]])
        df_reduce = df_reduce[~df_reduce['center_xbid'].isin(arr)]  #Show all results centered on the selected subscript

        if (df_reduce['P-value'] == 1).all():
            break

    '''gini sorted from large to small, equal center_xbid, leaving the first one, deleting the rest'''
    df_result = df_result.sort_index(axis=0, ascending=[False], by='NI')
    df_result = df_result.drop_duplicates(subset=['center_xbid'], keep='first') 

    # Create an empty set to store the final result
    df_zzjg = []
    df_zzjg = pd.DataFrame(df_zzjg)

    # Delete the first occurrence of duplicates in the table below
    while df_result.empty == False:

        df_result_cir = df_result.iloc[[0]] # get the first raw

        if df_zzjg is None:
            df_zzjg =df_result_cir

        else:
           df_zzjg = df_zzjg.append(df_result_cir)

        list_xbids = df_result_cir['xbids'].tolist()[0]
      
        df_result = df_result[~df_result['center_xbid'].isin(list_xbids)]  

    print('df_zzjg')
    print(df_zzjg)


 
