import csv
import re

def clean_text(text):
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF"
        "\U0000200D"
        "\U0001FA70-\U0001FAFF"
        "\U00002300-\U000023FF"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    text = re.sub(r'([:=]=*\)+|:v|:3|=3|:D)+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(haha|lmao|kkk+)\b', '', text, flags=re.IGNORECASE)
    return text.strip()

with open('all.csv', 'r', encoding='utf-8') as infile, \
     open('all1.csv', 'w', encoding='utf-8', newline='') as outfile:

    reader = csv.DictReader(infile)
    fieldnames = [fn for fn in reader.fieldnames if fn is not None]  # loại bỏ None
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        # Bỏ qua dòng nếu có key None (do lỗi dòng trống)
        if None in row:
            continue
        
        # Lọc row chỉ giữ các key hợp lệ
        clean_row = {k: row[k] for k in fieldnames if k in row}

        # Làm sạch cột 'text' nếu có
        if 'text' in clean_row:
            clean_row['text'] = clean_text(clean_row['text'])

        writer.writerow(clean_row)

print("✅ Đã làm sạch dữ liệu và lưu vào 'tichcuc_cleaned.csv'")
