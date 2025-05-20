from flask import Flask, render_template, request
import predict_logic # This will run initialize_models() from predict_logic.py
import os

app = Flask(__name__)

# Tải mô hình một lần khi server khởi động
try:
    predict_logic.initialize_models()
    print("Tất cả các mô hình đã được tải thành công cho server.")
except Exception as e:
    print(f"LỖI NGHIÊM TRỌNG: Không thể khởi tạo mô hình khi server bắt đầu: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    raw_text_input = ""
    selected_model = 'svc' # Mặc định

    if request.method == 'POST':
        raw_text_input = request.form.get('text_input', '')
        selected_model = request.form.get('model_type', 'svc')
        
        if not raw_text_input.strip():
            predictions = [] 
        else:
            text_list = [line.strip() for line in raw_text_input.splitlines() if line.strip()]
            if text_list:
                predictions = predict_logic.predict_sentiment(text_list, model_type=selected_model)
            else:
                predictions = [] 

    return render_template('index.html', predictions=predictions, raw_text_input=raw_text_input, selected_model=selected_model)

if __name__ == '__main__':
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')):
        print("LỖI: Thư mục 'templates' không được tìm thấy. Hãy tạo nó và đặt 'index.html' vào trong.")
    elif not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predict_logic.py')):
        print("LỖI: File 'predict_logic.py' không được tìm thấy trong cùng thư mục với 'app.py'.")
    else:
        # Chạy server ở chế độ debug để dễ dàng theo dõi lỗi
        # Chỉ sử dụng host='0.0.0.0' nếu bạn muốn truy cập từ máy khác trong mạng
        # app.run(debug=True, host='0.0.0.0', port=5000) 
        app.run(debug=True, port=5000)