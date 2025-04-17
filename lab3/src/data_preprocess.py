import csv
import pandas as pd

# 读取alias,创建键值对为 错误ID和正确ID的字典
def load_artist_alias(filename):
    alias_map = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("\t")
            # 存在只有一个ID的行，因此要筛选len(parts)==2
            if len(parts) == 2: 
                wrong_id, right_id = parts
                alias_map[wrong_id] = right_id
    return alias_map

# 处理 artist_data.txt
def process_artist_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
          open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["artist_id", "artist_name"])  # 属性名
        
        for line in infile:
            parts = line.strip().split(None, 1)  # 只分割第一个空白符
            if len(parts) == 2 and parts[0].isdigit():
                artist_id, artist_name = parts
                writer.writerow([artist_id, artist_name])   # 将一行数据写入csv文件

def max_count_status(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["artist_id", "artist_name"])  # 属性名
        
        for line in infile:
            parts = line.strip().split(None, 1)  # 只分割第一个空白符
            if len(parts) == 2:
                artist_id, artist_name = parts
                writer.writerow([artist_id, artist_name])   # 将一行数据写入csv文件
                
# 处理 user_artist_data.txt
def process_user_artist_data(input_file, output_file, alias_map):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["user_id", "artist_id", "count"])  # 属性名

        for line in infile:
            parts = line.strip().split()
            if len(parts) == 3:  # 过滤错误行
                user_id, artist_id, count = parts
                # 根据artist_alias.txt的映射关系，将user_artist_data.txt进行更新
                artist_id = alias_map.get(artist_id, artist_id)
                writer.writerow([user_id, artist_id, count])

# 提取前10%的艺术家，实现抽样，它是ALS的数据集
def sample_user_artist_data(input_file, output_file):
    sample_prob = 0.2
    artist_df = pd.read_csv('data/clean/artist_data.csv', dtype={'artist_id': 'int64'})
    
    # 提取前sample_prob%个艺术家
    sample_artist_df = artist_df.head(int(len(artist_df) * sample_prob))
    sample_artist_ids = sample_artist_df['artist_id'].tolist()
    
    # 在user_artist_data中筛选出artist_id在sample_prob中的所有样本
    user_artist_df = pd.read_csv(input_file)
    filtered_user_artist_df = user_artist_df[user_artist_df['artist_id'].isin(sample_artist_ids)]
    
    # 保存结果
    filtered_user_artist_df.to_csv(output_file, index=False)
    print(f"保留的样本数量: {len(filtered_user_artist_df)}")
    

if __name__ == '__main__':
    # 根据artist_alias.txt的映射关系，将user_artist_data.txt进行更新
    artist_alias_map = load_artist_alias("data/raw/artist_alias.txt")
    process_artist_data("data/raw/artist_data.txt", "data/clean/artist_data.csv")
    process_user_artist_data("data/raw/user_artist_data.txt", "data/clean/user_artist_data.csv", artist_alias_map)
    sample_user_artist_data("data/clean/user_artist_data.csv", "data/clean/user_artist_data_filtered.csv")

