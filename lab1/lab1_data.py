import pandas as pd
import json
import ast


# 计算profit_level
def caculate_profit_level(row):
    profit_level = 1
    if row['budget'] != 0 and row['revenue'] != 0:
        profit_ration = (row['revenue'] - row['budget']) / row['budget']
        if profit_ration >= 1:
            profit_level = 3
        elif profit_ration >= 0.5:
            profit_level = 2
    return profit_level

# 添加profit_level属性
def add_profit_level(df):
    df['profit_level'] = df.apply(caculate_profit_level, axis=1)
    return df

# 添加genres.names属性
def add_genres_columns(df):
    # 提取所有 genres
    all_genres = set()
    for genres in df['genres']:
        all_genres.update(genres)
    # print(all_genres)
    # 为每个 genre 创建一列并设置 0/1 值
    for genre in all_genres:
        df[genre] = df['genres'].apply(lambda x: 1 if genre in x else 0)
    
    return df

# 添加total_directed属性
def add_total_directed(df):
    director_movie_count = {}
    # 统计每个导演执导电影的次数
    for directors in df['director']:
        for director in directors:
            if director in director_movie_count:
                director_movie_count[director] += 1
            else:
                director_movie_count[director] = 1
    # 添加total_directed属性
    def cacluate_total_directed(directors):
        total_movies = sum(director_movie_count[director] for director in directors)
        return total_movies / len(directors) if directors else 0
    
    df['total_directed'] = df['director'].apply(cacluate_total_directed)
    
    return df

#%%
if __name__ == "__main__":
    
    # 1. 删除无关属性
    df = pd.read_csv('movies.csv')
    column_reserved = [0, 1, 5, 12, 13, 17, 22]
    df = df.iloc[:, column_reserved]  # 保留指定列
    
    # 2. 处理 genres、crew 属性，创建director属性
    for index, row in df.iterrows():
        if row['genres']:  # 如果 genres 字段不为空
            data = json.loads(row['genres'])
            names = [item["name"] for item in data]
            df.at[index, 'genres'] = names  # 更新 genres 字段
        if row['crew']:
            data = json.loads(row['crew'])
            directors = [item["name"] for item in data if item['job'] == "Director"]
            df.at[index, 'director'] = directors
    df = df.drop('crew', axis=1)
    
    df.to_csv("init.csv", index=False)
    #%%
    # 3. 填充缺失值
    
    # 4. 增加分组标签属性profit_level
    df = pd.read_csv('movie_fill.csv')
    df['genres'] = df['genres'].apply(ast.literal_eval)
    df['director'] = df['director'].apply(ast.literal_eval)
    df = add_profit_level(df)
    
    # 5. 新增属性genres.names
    df = add_genres_columns(df)
    
    # 6. 新增属性total_directed
    df = add_total_directed(df)
    
    # 输出中间结果
    df.to_csv("movie_add.csv", index=False)
    
    # 7. 删除director、 genres、 crew、 revenue、 title
    columns_removed = ['director', 'genres', 'revenue', 'title']
    df_result = df.drop(columns_removed, axis=1)
    
    # 输出预处理结果
    df_result.to_csv("data_result.csv", index=False)
    

    
    
    
    
    
    