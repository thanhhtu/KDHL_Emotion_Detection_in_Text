import pandas as pd

# Đọc CSV, thêm option: quotechar='"' để giữ nguyên phần có dấu phẩy trong dấu ngoặc kép
df = pd.read_csv("./all.csv", header=None, names=["text", "label"], quotechar='"')

# Đổi các label từ 2 thành 0
df["label"] = df["label"].replace(2, 0)

# (Tùy chọn) Lưu lại
df.to_csv("file_da_doi_label.csv", index=False, header=False)

# In kiểm tra
print(df[df["label"] == 0])
