import pandas as pd
import re

df = pd.read_csv("all1.csv", on_bad_lines='skip')  # pandas >= 1.3

print(df.head())

abbreviations = {
    "ko": "không",
    "k": "không",
    "khong": "không",
    "mk": "mình",
    "nt": "nhắn tin",
    "dc": "được",
    "đc": "được",
    "bt": "bình thường",
    "bth": "bình thường",
    "r ": "rồi ",
    "vs": "với",
    "cx": "cũng",
    "bn": "bạn",
    "mik": "mình",
    "j": "gì",
    "hok": "không",
    "kh": "không",
    "ak": "á",
    "dkh": "được không",
    "dk": "được",
    "dki": "đăng kí",
    "hk": "không",
    "thik": "thích",
    "lm": "làm",
    "s": "sao",
}

def clean_text(text):
    if pd.isnull(text):
        return ""

    
    for abbr, full in abbreviations.items():
        text = re.sub(rf"\b{abbr}\b", full, text)
    
    
    return text

df["Comment"] = df["Comment"].apply(clean_text)

df.to_csv("all.csv", index=False, encoding="utf-8-sig")