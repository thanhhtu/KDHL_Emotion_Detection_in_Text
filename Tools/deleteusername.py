import pandas as pd
import re

def remove_usernames(text):
    if isinstance(text, str):
        cleaned = re.sub(r'@\w[\w\-]*', '', text)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        return cleaned.strip()
    return text

# Đọc file CSV có sẵn header
df = pd.read_csv('file_v1.csv', on_bad_lines='skip')

# Kiểm tra tên cột
print("Cột trong file:", df.columns)

# Đảm bảo tên cột là 'comment' và 'label'
df['Comment'] = df['Comment'].apply(remove_usernames)

# Ép kiểu label về số nguyên (nếu cần)
df['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(-1).astype(int)

# Ghi ra file mới
df.to_csv('file_v2.csv', index=False)
print("✅ Done!")
