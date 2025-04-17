import requests
import pandas as pd

# 获得球员生物信息数据nba_player_bio
url = "https://stats.nba.com/stats/leaguedashplayerbiostats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&ISTRound=&LastNGames=0&LeagueID=00&Location=&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PerMode=Totals&Period=0&PlayerExperience=&PlayerPosition=&Season=2024-25&SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange=&StarterBench=&TeamID=0&VsConference=&VsDivision=&Weight="
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com'
}

response = requests.get(url, headers=headers)
data = response.json()

# 转换为DataFrame
columns = data['resultSets'][0]['headers']
rows = data['resultSets'][0]['rowSet']
df = pd.DataFrame(rows, columns=columns)

# 保存为CSV
df.to_csv('player_bio.csv', index=False)
#%%
import pandas as pd
from src.gloabl_vars import *

src_dict = '/raw'
dst_dict = '/clean'

# 读取当前赛季的比赛信息、球员的生物指标信息，上个赛季的整体统计信息
df1 = pd.read_csv(current + src_dict + schedule)
df2 = pd.read_csv(prev + src_dict + prev_stats)
df3 = pd.read_csv(current + src_dict + player_bio)

# 筛选列
column1_reserve = [0, 2, 3, 4, 5]
column3_reserve = [3, 4, 6, 7]

df1 = df1.iloc[:, column1_reserve]
df2 = df2.dropna(axis=1, how="all").iloc[:, :-3]
df3 = df3.iloc[:, column3_reserve]

# 预测集的行索引
predict_idx = (df1['Date'] == predict_start_date).idxmax()

# 在schedule的训练集和测试集中增加label: home_win
df1['home_win'] = (df1['PTS'] < df1['PTS.1']).astype(int)
df1.loc[predict_idx:, 'home_win'] = None
df1 = df1.drop(['PTS', 'PTS.1'], axis=1)

# 删除total_stats中球队名称的*号
df2['Team'] = df2['Team'].apply(lambda x: x.replace("*", ""))

# 将nba_player_bio中的队伍缩写名替换为全称
df = pd.read_json('team_name.json', orient='index')
team_dict = df[0].to_dict()
df3['team'] = df3['TEAM_ABBREVIATION'].apply(lambda x: team_dict.get(x))
df3 = df3.drop('TEAM_ABBREVIATION', axis=1)

# 将属性名转换为小写
df1.columns = ['date', 'visitor', 'home', 'home_win']
df2.columns = df2.columns.str.lower()
df3.columns = df3.columns.str.lower()

df1.to_csv(current + dst_dict + schedule, index=False)
df2.to_csv(prev + dst_dict + prev_stats, index=False)
df3.to_csv(current + dst_dict + player_bio, index=False)
