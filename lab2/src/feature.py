import pandas as pd
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.decomposition import PCA
from src.gloabl_vars import *

src_dict = '/clean'
dst_dict_feature = '/feature'
dst_dict_predict = '/predict'

team_elo = {}
team_bmi = {}
team_age = {}
team_rank = {}
team_streak_win = {}    
team_streak_lose = {}
last_race_time = {}
team_names = set()
base_elo = 1600

test_idx = 0    # 训练集 验证集
predict_idx = 0 # 验证集 测试集

def get_elo(team_name):
    try:
        return team_elo[team_name]
    except:
        team_elo[team_name] = base_elo
        return team_elo[team_name]

def calculate_elo(winner, loser):
    winner_elo = get_elo(winner)    
    loser_elo = get_elo(loser)
    
    exp = (loser_elo - winner_elo) / 400
    winner_win_exception = 1 / (1 + math.pow(10, exp))
    
    # 调整系数K，用于控制分数波动幅度
    K = 24
        
    new_winner_elo = round(winner_elo + (K * (1 - winner_win_exception)))
    elo_change = new_winner_elo - winner_elo
    new_loser_elo = loser_elo - elo_change
    
    return new_winner_elo, new_loser_elo

def add_elo_feature(df):
    train_df = df.iloc[:test_idx].copy()
    test_predict_df = df.iloc[test_idx:].copy()
    
    # 为训练集添加elo特征，注意使用累计的等级分，不能包含当前比赛的信息
    for index, row in train_df.iterrows():
        home_name = row['home']
        home_elo = get_elo(home_name)
        
        visitor_name = row['visitor']
        visitor_elo = get_elo(visitor_name)
        
        train_df.loc[index, 'elo'] = home_elo - visitor_elo
        
        # 更新elo等级分
        if row['home_win'] == 1:
            team_elo[home_name], team_elo[visitor_name] = calculate_elo(home_name, visitor_name)
        else:
            team_elo[visitor_name], team_elo[home_name] = calculate_elo(visitor_name, home_name)
        
    # 为测试集添加elo特征
    test_predict_df.loc[:, 'elo'] = test_predict_df['home'].map(get_elo) - test_predict_df['visitor'].map(get_elo)
    
    # 拼接回df，且确保数据仍然按照比赛时间排序
    df = pd.concat([train_df, test_predict_df]).reset_index(drop=True)
    return df

def calculate_bmi(row):
    height_m = row['player_height_inches'] * 0.0254
    weight_kg = row['player_weight'] * 0.453592
    bmi = round(weight_kg / (height_m ** 2), 2)
    return bmi    

def add_bio_feature(df):
    df['bmi'] = (df['home'].map(team_bmi) - df['visitor'].map(team_bmi)).round(2)
    df['age'] = (df['home'].map(team_age) - df['visitor'].map(team_age)).round(2)
    return df

def add_rk_feature(df):
    df['rk'] = df['home'].map(team_rank) - df['visitor'].map(team_rank)
    return df

def calcuate_streak(row):
    home_streak_win, home_streak_lose, visitor_streak_win, visitor_streak_lose = get_streak_info(row)
    
    streak = 0
    if (home_streak_win > 3 and visitor_streak_win > 3) or (home_streak_lose > 3 and visitor_streak_lose > 3):
        # 主队和客队都连胜/连负
        streak = 0
    elif home_streak_win > 3 and visitor_streak_lose > 3:
        # 主队连胜, 客队连负
        streak = 2  
    elif home_streak_lose > 3 and visitor_streak_win > 3:
        # 主队连负，客队连胜
        streak = -2
    elif home_streak_win > 3 or visitor_streak_lose > 3:
        # 主队连胜 或 客队连负
        streak = 1
    elif home_streak_lose > 3 or visitor_streak_win > 3:
        # 主队连负 或 客队连胜
        streak = -1
        
    return streak

def get_streak_info(row):
    home = row['home']
    visitor = row['visitor']
    home_streak_win = team_streak_win.get(home, 0)
    home_streak_lose = team_streak_lose.get(home, 0)
    visitor_streak_win = team_streak_win.get(visitor, 0)
    visitor_streak_lose = team_streak_lose.get(visitor, 0)
    
    return [home_streak_win, home_streak_lose, visitor_streak_win, visitor_streak_lose]

def update_streak_info(row):
    home = row['home']
    visitor = row['visitor']
    home_streak_win, home_streak_lose, visitor_streak_win, visitor_streak_lose = get_streak_info(row)
   
    if row['home_win'] == 1:
        team_streak_win[home] = home_streak_win + 1
        team_streak_lose[home] = 0
        team_streak_win[visitor] = 0
        team_streak_lose[visitor] = visitor_streak_lose + 1
    else:
        team_streak_win[visitor] = visitor_streak_win + 1
        team_streak_lose[visitor] = 0
        team_streak_win[home] = 0
        team_streak_lose[home] = home_streak_lose + 1

# 增加 连败特征
def add_streak_feature(df):
    train_df = df.iloc[:test_idx].copy()
    test_predict_df = df.iloc[test_idx:].copy()
    
    # 训练集添加streak
    for index, row in train_df.iterrows():
        # 设置streak属性，如果x连胜，则x_streak = 1；若x连负，则x_streak = -1. 
        train_df.loc[index, 'streak'] = calcuate_streak(row)
        # 更新连胜/连负记录
        update_streak_info(row)
     
    # 测试集添加streak
    for index, row in test_predict_df.iterrows():
        test_predict_df.loc[index, 'streak'] = calcuate_streak(row)
    
    df = pd.concat([train_df, test_predict_df]).reset_index(drop=True)
    return df

def get_time_info(row):
    home = row['home']
    visitor = row['visitor']
    current = row['date']
    home_last = last_race_time.get(home, current)
    visitor_last = last_race_time.get(visitor, current)
    return [current, home_last, visitor_last]
    
def calculate_time_diff(row):
    date_format = "%a %b %d %Y"
    current, home_last, visitor_last = get_time_info(row)
    home_last_race_time = datetime.strptime(home_last, date_format)
    visitor_last_race_time = datetime.strptime(visitor_last, date_format)
    delta = home_last_race_time - visitor_last_race_time
    return delta.days

def update_time_info(row):
    current, home_last, visitor_last = get_time_info(row)
    last_race_time[row['home']] = current
    last_race_time[row['visitor']] = current
    
def add_time_feature(df):
    for index, row in df.iterrows():
        df.loc[index, 'time_diff'] = calculate_time_diff(row)
        update_time_info(row)
    
    return df
        
def logistics_model(df):
    X = df1[['elo', 'bmi', 'age', 'rk', 'streak', 'time_diff']]
    y = df1['home_win']
    
    # 划分训练集,测试集和验证集，需要按照时间顺序划分
    X_train, X_test, X_predict = X[:test_idx], X[test_idx:predict_idx], X[predict_idx:]
    y_train, y_test = y[:test_idx], y[test_idx:predict_idx]
    
    # 标准化处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)   # 测试集的均值和标准差应该来源于训练集，避免数据泄露
    X_predict = scaler.transform(X_predict)
    
    pca = PCA(n_components=5)  # 设置主成分的数量，例如降到3个主成分
    X_train_pca = pca.fit_transform(X_train)  # 对训练集进行PCA降维
    X_test_pca = pca.transform(X_test)  # 对测试集进行PCA降维
    X_predict_pca = pca.transform(X_predict)     
    
    # 训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X_train_pca, y_train)
    
    # 预测测试集
    y_pred_test = model.predict(X_test_pca)
    print("测试集准确率:", accuracy_score(y_test, y_pred_test))
    print("分类报告:\n", classification_report(y_test, y_pred_test))
    
    # 预测预测集
    y_pred_predict = model.predict(X_predict_pca)
    df_predict_result = df.iloc[predict_idx:].drop(columns=['home_win']).copy()
    df_predict_result['home_win_predict'] = y_pred_predict
    return df_predict_result
#%%
if __name__ == "__main__":
    df1 = pd.read_csv(current + src_dict + schedule)
    df2 = pd.read_csv(current + src_dict + player_bio)
    df3 = pd.read_csv(prev + src_dict + prev_stats)
    
    # 获得所有队伍名称的集合
    # 每个赛季有30个固定的NBA球队参加比赛
    team_names = set(df1['visitor'])  | set(df1['home'])
    
    # 将df1划分为训练集、测试集和预测集
    # 由于数据之间有时序关系，因此需要一起构造特征，但为了避免数据泄露，构造特征的方法有所不同
    predict_idx = (df1['date'] == predict_start_date).idxmax()
    test_idx = int(predict_idx * 0.8)
    
    # 1. 计算基础准确率：默认主队获胜
    accuracy = df1.loc[:predict_idx, "home_win"].mean() * 100
    print(f"预测准确率: {accuracy:.1f}%")
    
    # 2. 添加累积等级分属性
    # 首先将df1划分为训练集和测试集,这是因为elo特征有时序依赖关系，预测集中elo只能使用训练集中的整体elo，而不能逐个计算
    df1 = add_elo_feature(df1)

    # 3. 添加球员生物特征
    df2['bmi'] = df2.apply(calculate_bmi, axis=1)
    team_bmi = df2.groupby('team')['bmi'].mean()
    team_age = df2.groupby('team')['age'].mean()
    
    df1 = add_bio_feature(df1)
    # logistics_model(df1)    
    
    # 4. 添加上个赛季的两个队伍的相对排名
    team_rank = dict(zip(df3['team'], df3['rk']))
    df1 = add_rk_feature(df1)
    # logistics_model(df1)
    
    # 5. 是否连胜/连负(最近3场)
    df1 = add_streak_feature(df1)
    # logistics_model(df1)
    
    # 6. 添加上次比赛距离本场比赛的时间
    df1 = add_time_feature(df1)
    df_predict_result = logistics_model(df1)
    df_predict_result = df_predict_result[['date', 'visitor', 'home', 'home_win_predict']]
    
    # 将结果写到feature目录中
    df1.to_csv(current + dst_dict_feature + schedule, index=False)
    df_predict_result.to_csv(current + dst_dict_predict + predict, index=False)





