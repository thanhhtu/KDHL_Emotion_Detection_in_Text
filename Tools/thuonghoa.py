with open("all1.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()

with open("all.csv", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line.lower())