import pandas as pd

# Đọc file CSV
df = pd.read_csv('manual_label.csv')  # Thay bằng tên thật nếu cần

# Nếu Label chưa tồn tại hoặc bạn muốn gán lại từ đầu
if 'Label' not in df.columns:
    df['Label'] = None

# Lặp qua từng dòng để nhập Label thủ công
for idx, row in df.iterrows():
    comment = row['Comment']
    print(f"\n[{idx}] {comment}")
    
    try:
        label = input("→ Nhập Label (ví dụ: 0, 1, 2, 3): ").strip()
        df.at[idx, 'Label'] = int(label)
        df.to_csv('labeled_output.csv', index=False)
    except ValueError:
        print("❌ Không hợp lệ, bỏ qua dòng này.")
        continue

# Lưu kết quả vào file mới (tránh ghi đè nếu không chắc)
df.to_csv('labeled_output.csv', index=False)
print("\n✅ Đã lưu kết quả vào labeled_output.csv")
