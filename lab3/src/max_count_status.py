import pandas as pd

def get_top_30_artists(input_file):
    df = pd.read_csv(input_file)
    artist_count = df.groupby('artist_id')['count'].sum()
    top_30_artists = artist_count.sort_values(ascending=False).head(30)
    top_30_artists_df = top_30_artists.reset_index()
    top_30_artists_df.columns = ['artist_id', 'count']
    return top_30_artists_df

def get_artist_info(input_file, top_30_artists_df):
    df = pd.read_csv(input_file, dtype={'artist_id': str})
    top_30_artists_df['artist_id'] = top_30_artists_df['artist_id'].astype('str')
    top_30_artists_info = df[df['artist_id'].isin(top_30_artists_df['artist_id'])]
    result_df = pd.merge(top_30_artists_info, top_30_artists_df, on='artist_id', how='left')
    return result_df

# 读取听歌数据和艺术家信息
top_30_artists_df = get_top_30_artists('data/clean/user_artist_data.csv')
# print(top_30_artists)
result_df = get_artist_info('data/clean/artist_data.csv', top_30_artists_df)
result_df = result_df.sort_values(by='count', ascending=False)
result_df.to_csv('data/status/top_30_artists.csv', index=False)