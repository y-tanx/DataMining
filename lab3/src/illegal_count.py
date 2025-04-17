import codecs
import chardet

def is_garbled(text):
    result = chardet.detect(text.encode())
    encoding = result['encoding']
    confidence = result['confidence']
    if confidence < 0.5 or encoding is None:
        return True
    return False
    
def artist_data_illegal_status(input_file):
    missing_artist_id = 0
    missing_artist_name = 0
    illegal_artist_name = 0

    with codecs.open(input_file, 'r', encoding = 'utf-8') as infile:
        for line in infile:
            parts = line.strip().split(None, 1) # 选择第一个\t作为分隔符
            if len(parts) == 1:
                if parts[0].isdigit():
                    missing_artist_name += 1
                else: 
                    missing_artist_id += 1
                continue
            
            artist_id, artist_name = parts
            if is_garbled(artist_name):
                illegal_artist_name += 1
    return {
            'missing_artist_id': missing_artist_id, 
            'missing_artist_name': missing_artist_name, 
            'illegal_artist_name': illegal_artist_name
            }

def artist_alias_illegal_status(input_file):
    missing_id = 0
    
    with codecs.open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                missing_id += 1
    return {'missing_id': missing_id}

if __name__ == "__main__":
    data_status = artist_data_illegal_status('data/raw/artist_data.txt')
    alias_status = artist_alias_illegal_status('data/raw/artist_alias.txt')
    print(data_status)
    print(alias_status)