import csv
from collections import Counter
import matplotlib.pyplot as plt

# Đọc file all.csv và đếm số lượng mỗi label
label_counter = Counter()
with open("all.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) > 1:
            label = row[-1].strip()
            label_counter[label] += 1

# Gán nhãn cảm xúc
label_names = {
    "0": "Giận dữ",
    "1": "Vui vẻ",
    "2": "Buồn bã",
    "3": "Bình thường",
    "4": "Ngạc nhiên"
}

labels = [label_names.get(k, k) for k in label_counter.keys()]
counts = list(label_counter.values())

# Vẽ biểu đồ
plt.figure(figsize=(8,5))
plt.bar(labels, counts, color='skyblue')
plt.xlabel("Cảm xúc")
plt.ylabel("Số lượng")
plt.title("Phân bố cảm xúc trong all.csv")
plt.tight_layout()
plt.show()