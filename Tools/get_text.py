import json
import csv

texts = []

with open('2.json', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            item = json.loads(line)
            if 'text' in item:
                texts.append(item['text'])
        except json.JSONDecodeError:
            continue

with open('./2.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['text'])  # Tiêu đề cột

    for text in texts:
        writer.writerow([text])

print(f"Đã trích xuất {len(texts)} dòng bình luận.")
