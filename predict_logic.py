import joblib
import re
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ĐẾN CÁC FILE ĐÃ LƯU ---
try:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
except NameError:
    BASE_DIR = os.getcwd()
    print(f"CẢNH BÁO: Biến __file__ không được định nghĩa. Sử dụng thư mục làm việc hiện tại: {BASE_DIR}")

MODEL_DIR = os.path.join(BASE_DIR, 'saved_models') 

LOAD_MODEL_SVC_PATH = os.path.join(MODEL_DIR, 'best_linear_svc_model.joblib')
LOAD_MODEL_LR_PATH = os.path.join(MODEL_DIR, 'best_logistic_regression_model.joblib')
LOAD_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
LOAD_PCA_MODEL_PATH = os.path.join(MODEL_DIR, 'pca_transformer.joblib')

# ... (phần còn lại của code như cũ) ...
PCA_WAS_USED_IN_TRAINING = False # <<< THAY ĐỔI GIÁ TRỊ NÀY CHO PHÙ HỢP

# --- SAO CHÉP CHÍNH XÁC CÁC HẰNG SỐ VÀ HÀM LÀM SẠCH ---
ABBREVIATIONS_FOR_PREDICTION = {
    "ko": "không", "k": "không", "kg": "không", "khg": "không", "kh": "không", "kô": "không", "hok": "không", "hk": "không",
    "dc": "được", "đc": "được", "dk": "được", "đk": "được", "đx": "được", "dx": "được",
    "bt": "biết", "bik": "biết", "bit": "biết",
    "ntn": "như thế nào",
    "vs": "với", "dzới": "với", "zới": "với",
    "mn": "mọi người", "mng": "mọi người",
    # ... (TOÀN BỘ DANH SÁCH ABBREVIATIONS CỦA BẠN) ...
    "huhu": "", "hehe": "", "haha": "", "hihi": "", "hoho": "", "kkk": "", "hichic": "",
    "contane": "container",
    "kmh": "km/h",
}
SORTED_ABBREVIATIONS_FOR_PREDICTION = dict(sorted(ABBREVIATIONS_FOR_PREDICTION.items(), key=lambda item: len(item[0]), reverse=True))
VIETNAMESE_CHARS_FOR_PREDICTION = "a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ"
ALLOWED_PUNCTUATION_FOR_PREDICTION = r".,?!:;()\[\]{}"

def clean_text_for_prediction_advanced(raw_text):
    # ... (Hàm làm sạch giữ nguyên như bạn đã định nghĩa ở trên) ...
    if not isinstance(raw_text, str):
        try: return str(raw_text)
        except: return ""
    text = raw_text.lower()
    for abb, full_form in SORTED_ABBREVIATIONS_FOR_PREDICTION.items():
        pattern = r'\b' + re.escape(abb) + r'\b'
        text = re.sub(pattern, full_form, text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*(\.\.\.)\s*', r' ... ', text) # Thêm khoảng trắng để ... có thể là token
    text = re.sub(r'\s+([.,?!:;])', r'\1', text)
    text = re.sub(r'([.,?!:;])([' + VIETNAMESE_CHARS_FOR_PREDICTION + r'])', r'\1 \2', text)
    text = re.sub(r'([(\[{])\s+', r'\1', text)
    text = re.sub(r'\s+([)\]}])', r'\1', text)
    text = re.sub(r'(['+ VIETNAMESE_CHARS_FOR_PREDICTION + r'])([(\[{])', r'\1 \2', text)
    text = re.sub(r'([)\]}])(['+ VIETNAMESE_CHARS_FOR_PREDICTION + r'])', r'\1 \2', text)
    pattern_to_keep = r'[^' + VIETNAMESE_CHARS_FOR_PREDICTION + r'\s' + re.escape(ALLOWED_PUNCTUATION_FOR_PREDICTION) + r']'
    text = re.sub(pattern_to_keep, '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- TẢI CÁC ĐỐI TƯỢNG ---
loaded_svc_model = None
loaded_lr_model = None
loaded_vectorizer = None
loaded_pca_transformer = None
models_loaded = False

def initialize_models():
    global loaded_svc_model, loaded_lr_model, loaded_vectorizer, loaded_pca_transformer, models_loaded
    if models_loaded:
        print("Các mô hình đã được tải trước đó.")
        return True
    try:
        loaded_svc_model = joblib.load(LOAD_MODEL_SVC_PATH)
        print(f"Đã tải model LinearSVC từ: {LOAD_MODEL_SVC_PATH}")

        loaded_lr_model = joblib.load(LOAD_MODEL_LR_PATH)
        print(f"Đã tải model LogisticRegression từ: {LOAD_MODEL_LR_PATH}")
        
        loaded_vectorizer = joblib.load(LOAD_VECTORIZER_PATH)
        print(f"Đã tải vectorizer từ: {LOAD_VECTORIZER_PATH}")

        if PCA_WAS_USED_IN_TRAINING:
            try:
                loaded_pca_transformer = joblib.load(LOAD_PCA_MODEL_PATH)
                print(f"Đã tải PCA transformer từ: {LOAD_PCA_MODEL_PATH}")
            except FileNotFoundError:
                print(f"CẢNH BÁO: PCA_WAS_USED_IN_TRAINING là True nhưng file PCA '{LOAD_PCA_MODEL_PATH}' không tồn tại. Bỏ qua PCA.")
                loaded_pca_transformer = None
        models_loaded = True
        return True
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi tải model hoặc vectorizer: {e}")
        models_loaded = False
        return False

# Gọi hàm tải mô hình ngay khi script được import hoặc chạy
initialize_models()


# --- HÀM DỰ ĐOÁN ---
def predict_sentiment(text_list_input, model_type='svc'): # Thêm model_type, mặc định là 'svc'
    """
    Dự đoán cảm xúc cho một danh sách các văn bản thô.
    Sử dụng các model và vectorizer đã được tải toàn cục.
    model_type: 'svc' hoặc 'lr' để chọn mô hình.
    """
    if not models_loaded:
        print("Lỗi: Các thành phần mô hình chưa được tải thành công. Gọi initialize_models() trước.")
        return [{"original_text": text, "error": "Models not loaded"} for text in text_list_input]
        
    if model_type == 'svc' and loaded_svc_model:
        active_model = loaded_svc_model
    elif model_type == 'lr' and loaded_lr_model:
        active_model = loaded_lr_model
    else:
        print(f"Lỗi: Loại mô hình '{model_type}' không hợp lệ hoặc mô hình chưa được tải. Sử dụng LinearSVC mặc định (nếu có).")
        if loaded_svc_model:
            active_model = loaded_svc_model
        elif loaded_lr_model: # Fallback to LR if SVC not loaded but LR is
            print("LinearSVC không khả dụng, sử dụng Logistic Regression.")
            active_model = loaded_lr_model
        else:
            return [{"original_text": text, "error": f"Model type '{model_type}' not available or no models loaded."} for text in text_list_input]


    if not isinstance(text_list_input, list):
        text_list_input = [text_list_input]

    cleaned_text_list = [clean_text_for_prediction_advanced(text) for text in text_list_input]
    text_tfidf_vectors = loaded_vectorizer.transform(cleaned_text_list)
    
    input_features = text_tfidf_vectors
    if PCA_WAS_USED_IN_TRAINING and loaded_pca_transformer:
        # ... (logic xử lý PCA như cũ) ...
        if text_tfidf_vectors.shape[1] == loaded_pca_transformer.n_features_in_:
            try:
                input_features = loaded_pca_transformer.transform(text_tfidf_vectors.toarray())
            except Exception as e:
                print(f"Lỗi khi áp dụng PCA: {e}. Dùng TF-IDF gốc.")
                input_features = text_tfidf_vectors
        else:
            print(f"CẢNH BÁO PCA: Kích thước TF-IDF không khớp PCA. Dùng TF-IDF gốc.")
            input_features = text_tfidf_vectors
            
    predictions = active_model.predict(input_features)
    
    # Giả sử 3 nhãn: 0 (tiêu cực), 1 (trung tính), 2 (tích cực)
    label_map = {
        0: "Tiêu cực",
        1: "Trung tính",
        2: "Tích cực"
    }
    
    results = []
    for i, pred_label in enumerate(predictions):
        sentiment = label_map.get(pred_label, f"Không xác định (Nhãn {pred_label})")
        results.append({
            "original_text": text_list_input[i],
            "cleaned_text_for_model": cleaned_text_list[i],
            "predicted_label": pred_label,
            "sentiment": sentiment,
            "model_used": model_type if active_model else "N/A"
        })
    return results

# # --- SỬ DỤNG TRONG FILE KHÁC ---
# if __name__ == "__main__":
#     if models_loaded:
#         print("\n--- Thử nghiệm dự đoán với các mô hình đã tải ---")
        
#         sample_comments = [
#            "bài hát này quá dở"
#         ]

#         print("\n**Dự đoán bằng LinearSVC:**")
#         predictions_s = predict_sentiment(sample_comments, model_type='svc')
#         for item in predictions_s:
#             print(f"Văn bản: '{item['original_text']}'")
#             print(f"  -> Dự đoán (SVC): {item['sentiment']} (Nhãn: {item['predicted_label']})\n")
            
#         print("\n**Dự đoán bằng Logistic Regression:**")
#         predictions_l = predict_sentiment(sample_comments, model_type='lr')
#         for item in predictions_l:
#             print(f"Văn bản: '{item['original_text']}'")
#             print(f"  -> Dự đoán (LR): {item['sentiment']} (Nhãn: {item['predicted_label']})\n")

#         # Ví dụ dự đoán một câu đơn với mô hình mặc định (SVC)
#         single_comment = "Hôm nay trời đẹp tuyệt vời luôn đó mn"
#         single_prediction = predict_sentiment(single_comment) 
#         if single_prediction:
#             item = single_prediction[0]
#             print(f"Dự đoán câu đơn (mặc định SVC): '{item['original_text']}' -> {item['sentiment']}")

#     else:
#         print("Không thể chạy thử nghiệm dự đoán do lỗi tải model/vectorizer.")