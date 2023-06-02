import numpy as np
import subprocess
import pway_funcs as fn2
import GRAB as grab
from sklearn import preprocessing
import scipy.io as scio
import pandas as pd
from mrmr import mrmr_classif
import os
import shutil


def contains_dcm_files(data_path):
    """
    检测给定文件夹中是否包含有扩展名为 .dcm 的文件。

    参数：
    data_path (str): 文件夹路径。

    返回：
    bool: 如果文件夹中包含 .dcm 文件，则返回 True，否则返回 False。
    """
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                return True
    return False


# 运行matlab函数处理数据获得roisignals
# data_dir为用户提供的数据路径
def run_matlab_function(data_dir):
    command = f'matlab -nodisplay -nojvm -nosplash -r "dpabi_auto(\'/home/kangxy/codes/pra.mat\',\'{data_dir}\', \'aal.nii\'); exit"'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print("Error occurred while running Matlab command:")
        print(stderr.decode('utf-8'))
    else:
        print("Matlab function executed successfully.")
        print(stdout.decode('utf-8'))

    return process.returncode
# 用法示例
# data_dir = "/home/kangxy/testdata/FunRaw/Sub_001"  # 这里可以更改为用户提供的data_dir
#     result = run_matlab_function(data_dir)
#     if result == 0:
#         # 在这里继续处理数据


# 获得somfcn 
def get_somfcn(filename): # filename = "/home/kangxy/processdata/Results/ROISignals_FunImgARWSCF/ROISignals_sub.mat"
    data_mat = scio.loadmat(filename)
    data = data_mat['ROISignals'][:, :90]  # 获得ROIsignals
    data, _ = sparse_SMO(data)
    return data

def sparse_SMO(train):
    max_iter = 50
    tol = 1e-4
    dual_max_iter = 600
    dual_tol = 1e-5
    # train = scipy.io.loadmat('data/simulation_data.mat')['Simulation_data_p']

    train1 = fn2.standardize(train)
    data_1 = train1.T
    S = np.cov(data_1)  # 协方差矩阵
    data = train
    node_num = 90
    (Theta, blocks) = grab.BCD_modified(Xtrain=data, Ytrain=data, S=S, lambda_1=16, lambda_2=8, K=5, max_iter=max_iter,
                                        tol=tol, dual_max_iter=dual_max_iter, dual_tol=dual_tol)
    # print("Theta: ", Theta)

    # print("Overlapping Blocks: ", blocks)
    Theta = -Theta
    # print(np.array(Theta).shape)

    # 此函数为取对角线 Thea - [[x,0,0],[0,y,0],[0,0,z]]
    Theta = Theta - np.diag(np.diag(Theta))
    Theta = Theta.reshape(((node_num * node_num), 1))

    # 数据归一化到[-1,1]，MinAbsScaler为归一化到[0,1]
    max_abs_scaler = preprocessing.MaxAbsScaler()
    # 先拟合数据，再标准化
    Theta = np.array(Theta)
    Theta = max_abs_scaler.fit_transform(Theta)
    Theta = Theta.reshape((node_num, node_num))

    # # 二值化 非0即1
    # Theta[Theta != 0] = 1

    #
    # plotting.plot_matrix(Theta, figure=(9, 7), vmax=1, vmin=-1, )
    # plt.title("SOM", fontsize='40')
    # plt.show()
    # # plt.savefig("../fig/som/som_test.png")
    return Theta, blocks


def getPerson(x_simple, y_simple):
    # 与dataframe的皮尔逊相关系数操作为是否转置的区别,转置之后结果相等
    return np.corrcoef(x_simple, y_simple)

# 获取高阶FCN 输入为som_fcn的数据
def get_somafcn(data):
    # (30*116 -> 30*30)
    dd = np.zeros_like(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if i < j:
                np.seterr(divide='ignore', invalid='ignore')
                # 计算皮尔逊相关系数
                tmp = getPerson(data[i], data[j])[0][1]
                dd[i, j] = 0 if np.isnan(tmp) else tmp  # 之前是dd[i,j]=-1
            else:
                dd[i, j] = dd[j, i]
    return dd


# 取上三角，打平后返回
def process_extract_upmatirx_feature_one(Matrix):
    a = np.array([])
    for i in range(Matrix.shape[0] - 1):
        a = np.append(a, Matrix[i + 1, : i + 1])
    a.astype(int)
    return a



def getmRMR(data_x, data_y, feature_count=50):
    X = pd.DataFrame(data_x).fillna(0)
    y = pd.Series(data_y)
    #
    # use mrmr classification  调用库函数 不论重复几次，结果不会改变
    # 输出资后的
    selected_features = mrmr_classif(X, y, K=feature_count)  # 索引从0到n-1
    # a = (np.array(selected_features[:21]) - 1) % 90  # -1是因为局部效率和全局效率
    # a = sorted(a)
    # print(a)
    return selected_features


def delete_files_and_create_subdir(base_path, sub_path):
    # 检查路径是否存在
    if os.path.exists(base_path):
        # 删除目录下的所有文件和子目录
        for filename in os.listdir(base_path):
            file_path = os.path.join(base_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

    else:
        # 如果路径不存在，则创建路径
        os.makedirs(base_path)

    # 创建子目录
    sub_dir_path = os.path.join(base_path, sub_path)
    os.makedirs(sub_dir_path, exist_ok=True)

