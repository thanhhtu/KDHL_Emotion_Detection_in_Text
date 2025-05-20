import csv

input_file = 'tichcuc_cleand11.csv'
output_file = 'tichcuc.csv'
label = '2'  # Gán nhãn cố định

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)

    for row in reader:
        if not row:
            continue
        # Gộp toàn bộ nội dung dòng lại (trong trường hợp có nhiều cột do dấu ,)
        text = ",".join(row).strip()
        writer.writerow([text, label])

print(f"\n✅ Đã gán nhãn {label} cho toàn bộ và lưu vào: {output_file}")
