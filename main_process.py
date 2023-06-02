# 根据用户提供的数据路径检测数据是否正常
# 若数据正常则根据process_funcs.py中的run_matlab_function()函数处理数据
# 等待数据处理完成后，根据数据路径读取数据
# 若文件存在则使用rocess_funcs.py中的get_somfcn()函数处理数据 得到的是处理后的somfcn数据
# 使用process_funcs.py中的get_somafcn()函数处理数据
# 使用process_funcs.py中的process_extract_upmatirx_feature_one()函数打平数据
# selected_feature_indices = [3020, 285, 1310, 1276, 2045, 2661, 754, 2068, 2877, 
#                             2048, 3420, 67, 2078, 2690, 3318, 2866, 3340, 320, 
#                             355, 472, 247, 1805, 3556, 936, 1227, 3531, 2808, 
#                             1520, 371, 2153, 3623, 3683, 1988, 2089, 3091, 1596,
#                               39, 1891, 417, 1711, 2293, 259, 584, 326, 21, 48, 
#                               3092, 2033, 2506, 2070] # 二分类
# 选择特征
# 加载svm模型，预测数据
import numpy as np
import pandas as pd
from process_funcs import *
from joblib import dump, load  # 添加joblib库
import joblib

def main_process(data_path):
    if contains_dcm_files(data_path) == 0:
        print("数据路径错误,未检测到dcm文件")
        print(data_path)
        return 0,0
    else:
        result = run_matlab_function(data_path)
        if result == 0:# 继续处理数据
            filename = "/home/kangxy/processdata/Results/ROISignals_FunImgARWSCF/ROISignals_sub.mat"
            if os.path.exists(filename):
                somfcn = get_somfcn(filename)
                # print("somfcn")
                somafcn = get_somafcn(somfcn)
                print('somafcn')
                upmatrix = process_extract_upmatirx_feature_one(somafcn)
                upmatrix = np.reshape(upmatrix, (1, -1))  # 将 upmatrix 转换为一个二维数组
                selected_feature_indices = [3020, 285, 1310, 1276, 2045, 2661, 754, 2068, 2877, 2048, 3420, 67, 2078, 2690, 3318, 2866, 3340, 320, 355, 472, 247, 1805, 3556, 936, 1227, 3531, 2808, 1520, 371, 2153, 3623, 3683, 1988, 2089, 3091, 1596, 39, 1891, 417, 1711, 2293, 259, 584, 326, 21, 48, 3092, 2033, 2506, 2070]
                selected_data = upmatrix[:, selected_feature_indices]
                # 加载模型
                model = joblib.load("/home/kangxy/model/Best_svm_model.pkl")
                # 预测数据
                result = model.predict(selected_data)
                print(result)
                result_char = ''
                # result=0,result_char='NC',result=3,result_char='AD',result=1,result_char='EMCI',result=2,result_char='LMCI'
                for i in range(len(result)):
                    if result[i] == 0:
                        result_char = result_char + 'NC'
                    elif result[i] == 3:
                        result_char = result_char + 'AD'
                    elif result[i] == 1:
                        result_char = result_char + 'EMCI'
                    elif result[i] == 2:
                        result_char = result_char + 'LMCI'
                
                delete_files_and_create_subdir('/home/kangxy/processdata', 'FunRaw/sub')
                return result_char,'97.8%'
            else:
                print("数据处理失败")
                return 0,0
        else:
            delete_files_and_create_subdir('/home/kangxy/processdata', 'FunRaw/sub')


if __name__ == "__main__":
    result,acc = main_process('/home/kangxy/testdata/FunRaw/Sub_001')# 返回分类结果及准确率
    print(result)
    print(acc)
