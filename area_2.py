import numpy as np

"""

    Calculate the area of the polygon:

   （1）including non-convex polygons
     The vertices required to be input are sequentially arranged in a counterclockwise order;
     The area of the required polygons are formed by connecting the vertices one by one.
     E = {<v0, v1>, <v1, v2>, <v2, v3>,...,<vn-2, vn-1>, <vn-1, v0>}
   （2）Supporting convex polygons
    :param polygon: Constructed trapezoidal by polygon vertices,then accumulation the divided cover 
    :return:  the area of the polygon

    The experiment is the second condition

    @author: Wangmei
    @date: 2019/05/21
"""

def mutli_area(v1, v2):
    """
    Calculate the crosswise of two vectors
    :param v1:
    :param v2:
    :return:
    """
    return ((v1[1] + v2[1]) *(v2[0] - v1[0]))/2

def polygon_area(polygon):

    n = len(polygon)

    if n < 3:
        return 0

    area = 0
    for i in range(1, n):
        area = area + mutli_area(polygon[i-1],polygon[i])

    return area


if __name__ == "__main__":
  
    """test"""
    polygon1 = np.array([[0, 0],
                         [0.25, 0.75],
                         [10, 10]
                  ])
    print(polygon_area(polygon1))

    # test 171
    polygon2 = np.array([
        [0.0, 0.0],
        [8,2.948454],
        [10,10]
    ])
    polygon3 = np.array([[0, 0],
                         [2.687,0.36],
                         [3.009,0.403],
                         [3.2, 0.429],
                         [3.235, 0.434],
                         [10, 10]
                         ])
    polygon4 = np.array([
                         [ 1.84487291849255,0.938783434473426],
                         [ 2.26117440841367, 1.43827865851629],
                         [ 2.76774758983348,  1.17366511720786],
                         [ 16.4276950043821,  6.96617446388356],
                         [ 19.3742331288343,  8.21565582045508],
                         [ 100, 100]
                         ])
    # test 12
    # polygon2 = np.array([[60, 0],
    #                      [60, 60],
    #                      [39.7711288343558, 6.29290013201832],
    #                      [0, 0]
    #                      ])
    # polygon3 = np.array([[60, 0],
    #                      [60, 60],
    #                      [39.7711288343558, 6.29290013201832],
    #                      [0.764241893076249, 0.12092435017029192],
    #                      [0, 0]
    #                      ])
    # polygon4 =np.array([[60, 0],
    #                      [60, 60],
    #                      [39.7711288343558, 6.29290013201832],
    #                      [0.764241893076249, 0.12092435017029192],
    #                      [0.353198948290973, 0.223543638158843],
    #                      [0, 0]
    #                      ])
    print('polygon2',1 - polygon_area(polygon2)/50)
    print('polygon3',1 - polygon_area(polygon3)/50)
    print('polygon4',1 - polygon_area(polygon4)/5000)
#
# df_result
#       C_zd      LGLR        NI  T_cluster                                              cc_zd  center_xbid      gini   p_value         r                                              uu_zd                                              xbids
# 1845     4  3.845418  1.882915          3  [1.034782608695652, 2.4869565217391285, 2.3130...          171  0.656926  0.443478  2.907957  [0.15383236216943078, 0.8192738682205274, 0.95...      [159, 168, 169, 170, 171, 172, 174, 185, 187]
# 1282     4  3.247890  1.261202          3  [0.7999999999999987, 0.18260869565217397, 0.61...          172  0.629761  0.765217  2.873473  [0.191125056028686, 0.06642761093679976, 0.093...  [168, 169, 170, 171, 172, 173, 174, 185, 186, ...
# 128      2  3.164201  0.970918          1           [0.4521739130434783, 0.5565217391304342]          154  0.510623  0.773913  1.699562        [0.037292693859255946, 0.04589870013446881]                                         [154, 156]
# 0        2  3.164201  0.877591          1                               [1.0086956521739125]          156  0.462752  0.773913  0.000000                              [0.08319139399372476]                                              [156]
# 1587     8  3.001119  0.686527          3  [0.0, 0.15652173913043432, 0.5217391304347818,...          169  0.370849  0.800000  3.859306  [0.0, 0.048946660690273196, 0.0978933213805465...  [106, 163, 167, 168, 169, 170, 171, 172, 173, ...
# 1729     3  2.718078  0.380656          3  [0.26956521739130435, 0, 0.026086956521739202,...          168  0.229486  0.886957  2.889832  [0.036127297176154184, 0, 0.020977140295831462...           [151, 159, 168, 171, 172, 174, 185, 187]
# 618      2  2.137259  0.023681          2                             [0.017391304347825987]           75  0.014836  0.991304  0.000000                            [0.0025549081129538176]                                               [75]
# df_zzjg
#       C_zd      LGLR        NI  T_cluster                                              cc_zd  center_xbid      gini   p_value         r                                              uu_zd                                          xbids
# 1845     4  3.845418  1.882915          3  [1.034782608695652, 2.4869565217391285, 2.3130...          171  0.656926  0.443478  2.907957  [0.15383236216943078, 0.8192738682205274, 0.95...  [159, 168, 169, 170, 171, 172, 174, 185, 187]
# 128      2  3.164201  0.970918          1           [0.4521739130434783, 0.5565217391304342]          154  0.510623  0.773913  1.699562        [0.037292693859255946, 0.04589870013446881]                                     [154, 156]
# 618      2  2.137259  0.023681          2                             [0.017391304347825987]           75  0.014836  0.991304  0.000000                            [0.0025549081129538176]                                           [75]
