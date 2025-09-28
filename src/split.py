import pandas as pd
from sklearn.model_selection import train_test_split


file_path = './data/labeled_data.csv'

def parse_csv(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split(',')
            
                        
            if i == 0:
                data.append(parts)
                continue

            if len(parts) == 7:   
                if parts[5] not in ['0', '1', '2', '3']:
                    continue
                parts = parts[:6] + [','.join(parts[6:])]
                data.append(parts)
    return pd.DataFrame(data[1:], columns=data[0])


df = parse_csv(file_path)

train, val = train_test_split(df, test_size=0.1, random_state=42)

train.to_csv('./data/train_labeled_data.csv', index=False)
val.to_csv('./data/val_labeled_data.csv', index=False)