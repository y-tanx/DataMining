import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import heapq

als = 1
has_error = False
user_id_map = {}
artist_id_map = {}

def build_matrixs(input_file, d=10):
    # prob = 1
    global user_id_map, artist_id_map
    data = pd.read_csv(input_file)
    # data = data.head(int(len(data) * prob))
    user_ids = data['user_id'].unique()
    artist_ids = data['artist_id'].unique()
    
    # 将id映射为索引
    user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    artist_id_map = {artist_id: idx for idx, artist_id in enumerate(artist_ids)}
    
    # 1.创建效用矩阵
    users = data['user_id'].map(user_id_map).values
    artists = data['artist_id'].map(artist_id_map).values
    values = data['count'].values
    M = sp.csr_matrix((values, (users, artists)), shape=(len(user_ids), len(artist_ids)))
    
    # 2. 效用矩阵标准化，对每个用户的count进行标准化
    # 计算每个用户的count的均值和标准差
    # 按行求和
    row_sum = np.array(M.sum(1), dtype=float).flatten()
    # 计算每一行的归一化因子，即每行和的倒数
    row_inv = np.power(row_sum, -1)
    row_inv[np.isinf(row_inv)] = 0  # 对行号为0，倒数为INF的值设置为0
    # 构造对角矩阵
    row_mat_inv = sp.diags(row_inv)
    # 行归一化
    M = row_mat_inv.dot(M)
    
    # 3. 初始化矩阵U、V
    num_users = len(data['user_id'].unique())
    num_artists = len(data['artist_id'].unique())
    # U = np.random.normal(0, 0.1, (num_users, d))
    # V = np.random.normal(0, 0.1, (d, num_artists))
    U = np.ones((num_users, d))
    V = np.ones((d, num_artists))
    
    return M, U, V

def draw_errors(errors, step):
    plt.plot(range(len(errors)), errors)
    plt.xlabel('Iteration')
    plt.ylabel('Squared Error')
    plt.title('Error vs. Iteration')
    plt.xticks(range(0, len(errors), step))
    plt.show()

def ALS(U, V, M, max_iter=20, lambda_reg=0.1, tolerance=1e-3):
    errors = []
    num_users, num_items = M.shape
    num_factors = U.shape[1]
    
    M_csc = M.tocsc()

    for current_iter in range(max_iter):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iteration {current_iter} started")
        # 固定V,更新U
        for u in range(num_users):
            row = M.getrow(u)
            item_indices = row.indices
            if len(item_indices) == 0:
                continue

            V_sub = V[:, item_indices]
            M_u = row.data

            A = V_sub @ V_sub.T + lambda_reg * np.eye(num_factors)
            b = V_sub @ M_u
            U[u, :] = np.linalg.solve(A, b)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] U updated")
        # 固定U,更新V
        for i in range(num_items):
            col = M_csc.getcol(i)
            user_indices = col.indices
            if len(user_indices) == 0:
                continue

            U_sub = U[user_indices, :]
            M_i = col.data

            A = U_sub.T @ U_sub + lambda_reg * np.eye(num_factors)
            b = U_sub.T @ M_i
            V[:, i] = np.linalg.solve(A, b)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] V updated")

        total_error = 0.0
        for u in range(num_users):
            row = M.getrow(u)
            for i, m_ui in zip(row.indices, row.data):
                prediction = np.dot(U[u], V[:, i])
                error = m_ui - prediction
                total_error += error ** 2
        errors.append(total_error)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: {total_error:.4f}")

        if current_iter >= 2 and abs(errors[-1] - errors[-2]) < tolerance:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Early stopped due to small delta error.")
            break

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished training!")
    
    return U, V, errors


def gradient_descent(U, V, M, max_iter=30, eta=1e-2, beta=0.1, tolerance=1e-3):
    errors = []
    num_users, num_items = M.shape

    # 将警告转化为异常
    warnings.simplefilter("error", RuntimeWarning)

    for current_iter in range(max_iter):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iteration {current_iter} started")

        try:
            # 更新 U 和 V
            for i in range(num_users):
                row = M[i, :]
                
                for j, m_ij in zip(row.indices, row.data):
                    pred = np.dot(U[i], V[:, j])
                    eij = m_ij - pred

                    # # 保存当前 U[i]，防止更新后影响 V 的梯度
                    U_i_old = U[i].copy()

                    U[i] += eta * (2 * eij * V[:, j] - beta * U[i])
                    V[:, j] += eta * (2 * eij * U_i_old - beta * V[:, j])

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] U, V updated")

            # 计算误差
            total_error = 0
            for i in range(num_users):
                row = M[i, :]
                for j, m_ij in zip(row.indices, row.data):
                    pred = np.dot(U[i], V[:, j])
                    error = m_ij - pred
                    total_error += error ** 2
                    total_error += (beta / 2) * (np.linalg.norm(U[i]) ** 2 + np.linalg.norm(V[:, j]) ** 2)

            errors.append(total_error)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: {total_error:.4f}")
            
            if current_iter >= 2:
                if errors[-1] >= errors[-2]:
                    has_error = True
                    break
                if current_iter >= 2 and abs(errors[-1] - errors[-2]) < tolerance :
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Early stopped at iter {current_iter}")
                    break

        except RuntimeWarning as e:
            # 如果捕获到 RuntimeWarning，则停止迭代并绘制误差图
            has_error = True
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Warning encountered: {e}. Stopping iteration.")
            break

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished Traning!")

    return U, V, errors

def write_result(U, V, output_file, top_k=100):
    buffer=[]
    buffer_size = 100_000
    reverse_artist_map = {v: k for k, v in artist_id_map.items()}
    user_ids = list(user_id_map.keys())
    
    with open(output_file, 'w') as f:
        f.write("user_id,artist_id,count\n")    # 标题
        
        for user_idx, user_id in enumerate(user_ids):
            
            # 获取用户矩阵的行向量
            user_row = U[user_idx, :]
            
            # 预测用户听歌次数,1*artist_num
            predicts = np.dot(user_row, V)
            
            # 获取top-k artist的艺术家索引
            topk_artists = heapq.nlargest(top_k, range(len(predicts)), key=lambda i: predicts[i])
            
            # 从索引转换为艺术家id
            for artist_idx in topk_artists:
                artist_id = reverse_artist_map.get(artist_idx)
                prediction = round(predicts[artist_idx], 4)
                buffer.append(f"{user_id},{artist_id},{prediction}\n")
            
            if len(buffer) >= buffer_size:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Write!")
                f.writelines(buffer)
                buffer.clear()
        if buffer:
            f.writelines(buffer)
            
    print("Finish Writing!")
                
    
if __name__ == "__main__":
    # 1. 获得效用矩阵、用户矩阵、项矩阵
    if als == 1:
        # 使用ALS算法实现矩阵分解
        M, U, V = build_matrixs('data/clean/user_artist_data.csv')
        U, V, errors = ALS(U, V, M)
        draw_errors(errors, 2)
    else:
        # etas = [1e-3, 1e-4, 1e-5]
        # etas = [1e-1, 1e-2, 8e-3, 5e-3, 2e-3]
        etas = [1e-3]
        for eta in etas:
            print(f"\n\n学习率: {eta}\n")
            has_error = False
            M, U, V = build_matrixs('data/clean/user_artist_data.csv')
            # M, U, V = build_matrixs('data/clean/1_large.csv')
            U, V, errors = gradient_descent(U, V, M, eta=eta)
            if(has_error):
                # 绘制误差图
                draw_errors(errors[:-2], 2)
            else:
                draw_errors(errors, 2)
    # 3. 将结果写到CSV文件中
    write_result(U, V, 'data/predict/user_artist_data_als.csv')
