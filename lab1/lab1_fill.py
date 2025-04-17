import pandas as pd
from imdb import Cinemagoer
import ast
import re
import time
import math

def get_number(value):
    if isinstance(value, str):
        match = re.search(r'(\d[\d,]*)', value)
        if match:
            return float(match.group(1).replace(',', ''))  # 转换为float()
    elif isinstance(value, (float, int)) and not math.isnan(value):  # 允许 float 或 int 直接返回
        return float(value)
    return 0.0

def handle_missing_values(df):
    columns_to_be_filled = ['budget', 'revenue', 'runtime', 'genres']
    ia = Cinemagoer()

    def fill_columns(row):
        need_fill = False
        if any(
                (pd.isnull(row[col]) or row[col] == 0) if col != 'genres' else len(row[col]) == 0
                for col in columns_to_be_filled):
            need_fill = True
                
        if need_fill:
            while True:
                try:
                    movies = ia.search_movie(row['title'])
                    if movies:
                        movie = ia.get_movie(movies[0].movieID)
                        
                        box_office = movie.get('box office', {})
                        row['budget'] = get_number(box_office.get('Budget', row['budget']))
                        revenue_value = box_office.get('Cumulative Worldwide Gross', box_office.get('Opening Weekend United States', None))
                        row['revenue'] = get_number(revenue_value if revenue_value is not None else row['revenue'])
                        row['runtime'] = get_number(movie.get('runtimes', [row['runtime']])[0])
                        
                        # 由于IMDb API中的gernes与当前数据集中的gernes有很多不同，所以尽量避免替换genres属性
                        if len(row['genres']) == 0:
                            row['genres'] = movie.get('genres', row['genres'])
                    break
                except Exception as e:
                    print(f"捕获异常:{e}")
                    time.sleep(10)
                    print("重试")
        return row        

    return df.apply(fill_columns, axis=1)

if __name__ == "__main__":
    df = pd.read_csv("init.csv")
    df['genres'] = df['genres'].apply(ast.literal_eval)
    df = handle_missing_values(df)
    
    df.to_csv("movie_fill.csv", index=False)

