import random

with open("all_finally.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

with open("all1.csv", "w", encoding="utf-8") as f:
    f.writelines(lines)