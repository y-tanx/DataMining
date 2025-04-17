import pandas as pd
import numpy as np
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import Isomap

def draw_result(df, algorithm_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['F1'], df['F2'], c = df['profit_level'], cmap='viridis', alpha=0.7)
    
    plt.title(algorithm_name + ' Result')
    plt.xlabel('F1')
    plt.ylabel('F2')
    
    plt.colorbar(label='Profit Level')
    plt.show()

def clustering(df, title):
    # 聚类
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['F1', 'F2']])

    # 绘制聚类结果的散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(df['F1'], df['F2'], c=df['cluster'], cmap='viridis')
    plt.title(title)
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.colorbar(label='Cluster')
    plt.show()
    
def draw_profit_level(df, title):
    # 统计各类中 profit_level 属性值的分布
    # 创建 3 个子图来显示不同聚类的 profit_level 分布
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # 统计每个聚类的 profit_level 分布
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        
        # 绘制直方图
        axes[cluster].hist(cluster_data['profit_level'], bins=np.arange(0.5, 4, 1), edgecolor='black', rwidth=0.8)
        axes[cluster].set_title(title + f'Cluster {cluster}')
        axes[cluster].set_xlabel('Profit Level')
        axes[cluster].set_ylabel('Frequency')
        axes[cluster].set_xticks([1, 2, 3])
        axes[cluster].set_xticklabels([1, 2, 3])
    
    plt.tight_layout()
    plt.show()

#%%
if __name__ == "__main__":
    
    # 读取数据预处理结果data_result
    df = pd.read_csv("data_result.csv")
    
    # 1. 使用目标编码处理original_language，目标变量为profit_level
    encoder = TargetEncoder(cols=['original_language'], smoothing=10)
    df = encoder.fit_transform(df, df['profit_level'])    
    df.to_csv("encoded_data.csv", index=False)
    
    # 2. 对数据进行标准化处理
    features = [col for col in df.columns if col != 'profit_level'] # 要降维的属性
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features]) # 标准化数据
#%%
    # 3. PCA降维
    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(df_scaled), columns=['F1', 'F2'])
    df_pca = pd.concat([df_pca, df['profit_level']], axis=1)
    df_pca.to_csv("pca_data.csv", index=False)
    # 结果可视化
    draw_result(df_pca, 'PCA')
    
    # KMeans 聚类
    clustering(df_pca, 'PCA-KMeans Cluster Result')
    
    # 绘制profit_level在每个类中的分布情况
    draw_profit_level(df_pca, 'PCA-KMeans')
#%%
    # 4. Isomap降维
    isomap = Isomap(n_components=2)
    df_isomap = pd.DataFrame(isomap.fit_transform(df_scaled), columns=['F1', 'F2'])
    df_isomap = pd.concat([df_isomap, df['profit_level']], axis=1)
    df_isomap.to_csv("isomap_data.csv", index=False)
    #结果可视化
    draw_result(df_isomap, 'Isomap')
    
    # KMeans聚类
    clustering(df_isomap, 'Isomap-KMeans Cluster Result')
    
    # 绘制profit_level在每个类中的分布情况
    draw_profit_level(df_isomap, 'Isomap-KMeans')
    
    
    