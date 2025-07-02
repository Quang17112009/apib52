from flask import Flask, jsonify, request
import requests
import os
import collections
import copy
import random

app = Flask(__name__)

# --- Cấu hình API bên ngoài ---
EXTERNAL_API_URL = "https://apib52.up.railway.app/api/taixiumd5" 

# --- Mẫu dữ liệu ban đầu và trạng thái toàn cục ---
initial_api_data_template = {
    "Phien_moi": None,
    "pattern_length": 8,
    "pattern": "xxxxxxxx",
    "matches": ["x"],
    "pattern_tai": 0,
    "pattern_xiu": 0,
    "pattern_percent_tai": 0,
    "pattern_percent_xiu": 0,
    "phan_tram_tai": 50,
    "phan_tram_xiu": 50,
    "tong_tai": 0.0,
    "tong_xiu": 0.0,
    "du_doan": "Không có",
    "ly_do": "Chưa có dữ liệu dự đoán.",
    "phien_du_doan": None,
    "admin_info": "@heheviptool",
    "prediction_performance": {
        "total_predictions_evaluated": 0,
        "correct_predictions": 0,
        "accuracy_percentage": 0.0
    },
    "current_prediction_score": { # Thêm trường này để hiển thị điểm số
        "Tài": 0,
        "Xỉu": 0
    }
}

# Lịch sử các kết quả thực tế (t='Tài', x='Xỉu')
history_results = collections.deque(maxlen=100) 

# Lưu trữ trạng thái dự đoán gần nhất và bộ đếm hiệu suất
last_prediction_info = {
    "predicted_expect": None,
    "predicted_result": None,
    "consecutive_losses": 0,
    "last_actual_result": None,
    "predictions_evaluated": 0,
    "correct_predictions": 0
}

# --- Hàm hỗ trợ ---
def calculate_tai_xiu(total_sum):
    """
    Xác định Tài/Xỉu từ tổng xúc xắc.
    Trả về ('Tài'/'Xỉu')
    """
    if total_sum >= 4 and total_sum <= 10:
        return "Xỉu"
    elif total_sum >= 11 and total_sum <= 17:
        return "Tài"
    else: 
        if total_sum == 3: return "Xỉu"
        if total_sum == 18: return "Tài"
        return "Không xác định"

def get_next_expect_code(current_expect_code):
    """
    Tính toán Expect code của phiên tiếp theo bằng cách tăng phần số cuối cùng.
    Ở đây, Expect code (Phien) là một số nguyên, nên chỉ cần tăng lên 1.
    """
    try:
        return current_expect_code + 1
    except TypeError:
        print(f"Warning: Expect code '{current_expect_code}' is not an integer.")
        return None

def update_history_and_state(new_session_data):
    """
    Cập nhật lịch sử và trạng thái dự đoán toàn cục dựa trên dữ liệu phiên mới.
    """
    global history_results, initial_api_data_template, last_prediction_info

    current_id = new_session_data['id']
    current_phien = new_session_data['Phien']
    current_tong = new_session_data['Tong']
    actual_result_type = new_session_data['Ket_qua']
    actual_result_char = "t" if "Tài" in actual_result_type else "x"

    if not any(entry['Phien'] == current_phien for entry in history_results):
        history_results.append({
            "ID": current_id,
            "Phien": current_phien,
            "Tong": current_tong,
            "Result": actual_result_char
        })
        print(f"Added new session to history: Phien {current_phien}, ID {current_id} - Result: {actual_result_type}")

        # --- Cập nhật Consecutive Losses và tổng số dự đoán ---
        if last_prediction_info["predicted_expect"] is not None and \
           last_prediction_info["predicted_expect"] == current_phien and \
           last_prediction_info["predicted_result"] is not None:
            
            predicted_res_char = "t" if last_prediction_info["predicted_result"] == "Tài" else "x"
            
            last_prediction_info["predictions_evaluated"] += 1

            if predicted_res_char != actual_result_char:
                last_prediction_info["consecutive_losses"] += 1
                print(f"Prediction '{last_prediction_info['predicted_result']}' for session Phien {current_phien} MISSED. Consecutive losses: {last_prediction_info['consecutive_losses']}")
            else:
                last_prediction_info["consecutive_losses"] = 0
                last_prediction_info["correct_predictions"] += 1
                print(f"Prediction '{last_prediction_info['predicted_result']}' for session Phien {current_phien} CORRECT. Resetting losses.")
        else:
            last_prediction_info["consecutive_losses"] = 0
            print("No matching previous prediction to evaluate or app restarted. Resetting losses.")
        
        last_prediction_info["last_actual_result"] = actual_result_char

    initial_api_data_template["Phien_moi"] = current_phien
    
    next_phien = get_next_expect_code(current_phien)
    initial_api_data_template["phien_du_doan"] = next_phien if next_phien is not None else "Không xác định"

    current_pattern_chars = "".join([entry['Result'] for entry in history_results])
    initial_api_data_template['pattern'] = current_pattern_chars[-initial_api_data_template['pattern_length']:]
    
    tai_count = initial_api_data_template['pattern'].count('t')
    xiu_count = initial_api_data_template['pattern'].count('x')
    
    initial_api_data_template['pattern_tai'] = tai_count
    initial_api_data_template['pattern_xiu'] = xiu_count

    total_pattern_chars = len(initial_api_data_template['pattern'])
    if total_pattern_chars > 0:
        initial_api_data_template['pattern_percent_tai'] = round((tai_count / total_pattern_chars) * 100, 2)
        initial_api_data_template['pattern_percent_xiu'] = round((xiu_count / total_pattern_chars) * 100, 2)
    else:
        initial_api_data_template['pattern_percent_tai'] = 0
        initial_api_data_template['pattern_percent_xiu'] = 0

    if history_results:
        initial_api_data_template['matches'] = [history_results[-1]['Result']]
    else:
        initial_api_data_template['matches'] = []

    initial_api_data_template['phan_tram_tai'] = initial_api_data_template['pattern_percent_tai']
    initial_api_data_template['phan_tram_xiu'] = initial_api_data_template['pattern_percent_xiu']
    
    initial_api_data_template['tong_tai'] = round(initial_api_data_template['phan_tram_tai'] * 1000 / 100, 2)
    initial_api_data_template['tong_xiu'] = round(initial_api_data_template['phan_tram_xiu'] * 1000 / 100, 2)

    # Cập nhật các trường hiệu suất dự đoán trong dictionary con
    initial_api_data_template['prediction_performance']['total_predictions_evaluated'] = last_prediction_info["predictions_evaluated"]
    initial_api_data_template['prediction_performance']['correct_predictions'] = last_prediction_info["correct_predictions"]
    
    if last_prediction_info["predictions_evaluated"] > 0:
        initial_api_data_template['prediction_performance']['accuracy_percentage'] = round(
            (last_prediction_info["correct_predictions"] / last_prediction_info["predictions_evaluated"]) * 100, 2
        )
    else:
        initial_api_data_template['prediction_performance']['accuracy_percentage'] = 0.0


# --- Logic Dự Đoán Thông Minh Hơn (Đã Nâng Cấp với Hệ Thống Tính Điểm) ---
def analyze_streaks(history_deque):
    """Phân tích các chuỗi (streaks) Tài/Xỉu trong lịch sử gần đây."""
    if not history_deque:
        return 0, None

    current_streak_length = 0
    current_streak_type = None

    for i in range(len(history_deque) - 1, -1, -1):
        result = history_deque[i]['Result']
        if current_streak_type is None:
            current_streak_type = result
            current_streak_length = 1
        elif result == current_streak_type:
            current_streak_length += 1
        else:
            break

    return current_streak_length, current_streak_type

def calculate_conditional_probability(history_deque, lookback_length=3):
    """
    Tính xác suất có điều kiện của 't' hoặc 'x' dựa trên 'lookback_length' kết quả trước đó.
    Trả về dict: { 'prefix': {'t': probability_of_next_is_t, 'x': probability_of_next_is_x} }
    """
    if len(history_deque) < lookback_length + 1:
        return {}

    probabilities = {}
    
    results_chars = "".join([entry['Result'] for entry in history_deque])

    for i in range(len(results_chars) - lookback_length):
        prefix = results_chars[i : i + lookback_length]
        next_char = results_chars[i + lookback_length]

        if prefix not in probabilities:
            probabilities[prefix] = {'t': 0, 'x': 0, 'total': 0}
        
        probabilities[prefix][next_char] += 1
        probabilities[prefix]['total'] += 1
    
    final_probs = {}
    for prefix, counts in probabilities.items():
        if counts['total'] > 0:
            final_probs[prefix] = {
                't': counts['t'] / counts['total'],
                'x': counts['x'] / counts['total']
            }
        else:
            final_probs[prefix] = {'t': 0, 'x': 0}

    return final_probs


def perform_prediction_logic():
    """
    Thực hiện logic dự đoán thông minh cho phiên tiếp theo và cập nhật 'du_doan', 'ly_do'.
    Sử dụng hệ thống tính điểm để kết hợp các tín hiệu.
    """
    global initial_api_data_template, last_prediction_info, history_results

    # Khởi tạo điểm số cho Tài và Xỉu
    prediction_scores = {"Tài": 0, "Xỉu": 0}
    reasons = [] # Danh sách các lý do để tạo chuỗi 'ly_do' chi tiết hơn

    # --- 1. Tín hiệu Phân tích cầu (Streaks) ---
    min_streak_for_prediction = 3 # Cầu tối thiểu để bắt đầu dự đoán theo/bẻ
    break_streak_threshold = 5   # Ngưỡng bẻ cầu
    
    current_streak_length, current_streak_type = analyze_streaks(history_results)

    if current_streak_type:
        if current_streak_length >= min_streak_for_prediction:
            if current_streak_length < break_streak_threshold:
                # Theo cầu: Cộng điểm cho hướng cầu
                if current_streak_type == 't':
                    prediction_scores["Tài"] += 2
                    reasons.append(f"Theo cầu Tài dài ({current_streak_length} lần).")
                else:
                    prediction_scores["Xỉu"] += 2
                    reasons.append(f"Theo cầu Xỉu dài ({current_streak_length} lần).")
            else:
                # Bẻ cầu: Cộng điểm cao hơn cho hướng ngược lại
                if current_streak_type == 't':
                    prediction_scores["Xỉu"] += 3 # Trọng số cao hơn cho bẻ cầu dài
                    reasons.append(f"Bẻ cầu Tài dài ({current_streak_length} lần) có khả năng đảo chiều.")
                else:
                    prediction_scores["Tài"] += 3
                    reasons.append(f"Bẻ cầu Xỉu dài ({current_streak_length} lần) có khả năng đảo chiều.")
        else:
            reasons.append("Không có cầu rõ ràng để áp dụng luật theo/bẻ cầu.")
    else:
        reasons.append("Chưa đủ dữ liệu để phân tích cầu.")

    # --- 2. Tín hiệu Xác suất có điều kiện (Conditional Probability) ---
    lookback_prob = 3 # Nhìn vào N phiên trước đó để tính xác suất
    
    if len(history_results) >= lookback_prob:
        recent_prefix_chars = "".join([entry['Result'] for entry in history_results])[-lookback_prob:]
        conditional_probs = calculate_conditional_probability(history_results, lookback_prob)

        if recent_prefix_chars in conditional_probs:
            prob_t = conditional_probs[recent_prefix_chars]['t']
            prob_x = conditional_probs[recent_prefix_chars]['x']

            prob_threshold_strong = 0.6          # Ngưỡng xác suất đủ mạnh
            prob_threshold_very_strong = 0.75    # Ngưỡng xác suất rất mạnh, cho điểm cao hơn
            
            if prob_t > prob_x and prob_t >= prob_threshold_strong:
                score_gain = 2
                if prob_t >= prob_threshold_very_strong: 
                    score_gain = 4 # Điểm cao hơn cho xác suất cực mạnh
                prediction_scores["Tài"] += score_gain
                reasons.append(f"Xác suất Tài cao ({round(prob_t*100, 2)}%) sau '{recent_prefix_chars}' (Điểm: +{score_gain}).")
            elif prob_x > prob_t and prob_x >= prob_threshold_strong:
                score_gain = 2
                if prob_x >= prob_threshold_very_strong: 
                    score_gain = 4
                prediction_scores["Xỉu"] += score_gain
                reasons.append(f"Xác suất Xỉu cao ({round(prob_x*100, 2)}%) sau '{recent_prefix_chars}' (Điểm: +{score_gain}).")
        else:
            reasons.append(f"Không tìm thấy mẫu '{recent_prefix_chars}' trong dữ liệu lịch sử để tính xác suất có điều kiện.")
    else:
        reasons.append("Chưa đủ dữ liệu để tính xác suất có điều kiện.")

    # --- 3. Tín hiệu Logic "Đang trật X lần → Auto đảo ngược" ---
    # Cơ chế này ưu tiên cao, nó sẽ cố gắng "bẻ" mọi dự đoán trước đó nếu đang thua liên tiếp
    reverse_threshold = 3 # Ngưỡng đảo ngược
    if last_prediction_info["consecutive_losses"] >= reverse_threshold:
        score_gain_base = 5 # Điểm cơ bản cao để ưu tiên cơ chế này
        score_gain = score_gain_base + (last_prediction_info["consecutive_losses"] - reverse_threshold) * 2 # Tăng điểm nếu trật nhiều hơn

        # Dự đoán ngược lại với kết quả thực tế gần nhất, hoặc ưu tiên bên có điểm thấp hơn để "đảo"
        # Điều chỉnh điểm số để đảm bảo bên đảo ngược có lợi thế
        if last_prediction_info["last_actual_result"] == 't': # Nếu phiên trước là Tài, giờ dự đoán Xỉu
             prediction_scores["Xỉu"] += score_gain
             prediction_scores["Tài"] = max(0, prediction_scores["Tài"] - score_gain_base) # Giảm điểm của bên kia để tạo sự chênh lệch rõ ràng
             reasons.append(f"Đang trật {last_prediction_info['consecutive_losses']} lần → Auto đảo ngược sang Xỉu (Điểm: +{score_gain}).")
        elif last_prediction_info["last_actual_result"] == 'x': # Nếu phiên trước là Xỉu, giờ dự đoán Tài
             prediction_scores["Tài"] += score_gain
             prediction_scores["Xỉu"] = max(0, prediction_scores["Xỉu"] - score_gain_base) # Giảm điểm của bên kia
             reasons.append(f"Đang trật {last_prediction_info['consecutive_losses']} lần → Auto đảo ngược sang Tài (Điểm: +{score_gain}).")
        else: # Trường hợp chưa có last_actual_result (ví dụ khởi động lại)
            if prediction_scores["Tài"] >= prediction_scores["Xỉu"]: # Nếu Tài có điểm cao hơn hoặc bằng
                prediction_scores["Xỉu"] += score_gain # Đảo sang Xỉu
                reasons.append(f"Đang trật {last_prediction_info['consecutive_losses']} lần → Auto đảo ngược sang Xỉu (Điểm: +{score_gain}).")
            else: # Nếu Xỉu có điểm cao hơn
                prediction_scores["Tài"] += score_gain # Đảo sang Tài
                reasons.append(f"Đang trật {last_prediction_info['consecutive_losses']} lần → Auto đảo ngược sang Tài (Điểm: +{score_gain}).")


    # --- Quyết định cuối cùng dựa trên điểm số ---
    du_doan_ket_qua = "Không có"
    ly_do_du_doan = ""

    # Lưu điểm số hiện tại vào template để hiển thị
    initial_api_data_template['current_prediction_score']['Tài'] = prediction_scores["Tài"]
    initial_api_data_template['current_prediction_score']['Xỉu'] = prediction_scores["Xỉu"]


    if prediction_scores["Tài"] > prediction_scores["Xỉu"]:
        du_doan_ket_qua = "Tài"
        ly_do_du_doan = f"Dự đoán Tài (Điểm: Tài={prediction_scores['Tài']}, Xỉu={prediction_scores['Xỉu']}). Chi tiết: {' | '.join(reasons)}"
    elif prediction_scores["Xỉu"] > prediction_scores["Tài"]:
        du_doan_ket_qua = "Xỉu"
        ly_do_du_doan = f"Dự đoán Xỉu (Điểm: Tài={prediction_scores['Tài']}, Xỉu={prediction_scores['Xỉu']}). Chi tiết: {' | '.join(reasons)}"
    else:
        # Nếu điểm bằng nhau, dùng tỷ lệ pattern chung làm yếu tố quyết định cuối cùng
        if initial_api_data_template['pattern_percent_tai'] > initial_api_data_template['pattern_percent_xiu']:
            du_doan_ket_qua = "Tài"
            ly_do_du_doan = f"Dự đoán Tài (Điểm bằng nhau: Tài={prediction_scores['Tài']}, Xỉu={prediction_scores['Xỉu']}). Ưu tiên Tài theo tỷ lệ pattern chung. Chi tiết: {' | '.join(reasons)}"
        elif initial_api_data_template['pattern_percent_xiu'] > initial_api_data_template['pattern_percent_tai']:
            du_doan_ket_qua = "Xỉu"
            ly_do_du_doan = f"Dự đoán Xỉu (Điểm bằng nhau: Tài={prediction_scores['Tài']}, Xỉu={prediction_scores['Xỉu']}). Ưu tiên Xỉu theo tỷ lệ pattern chung. Chi tiết: {' | '.join(reasons)}"
        else:
            # Nếu tất cả các tín hiệu đều cân bằng, dự đoán ngẫu nhiên
            du_doan_ket_qua = random.choice(["Tài", "Xỉu"])
            ly_do_du_doan = f"Dự đoán Ngẫu nhiên (Tất cả tín hiệu cân bằng. Điểm: Tài={prediction_scores['Tài']}, Xỉu={prediction_scores['Xỉu']}). Chi tiết: {' | '.join(reasons)}"


    initial_api_data_template['du_doan'] = du_doan_ket_qua
    initial_api_data_template['ly_do'] = ly_do_du_doan

    # Lưu dự đoán này để kiểm tra ở phiên tiếp theo
    last_prediction_info["predicted_expect"] = initial_api_data_template["phien_du_doan"]
    last_prediction_info["predicted_result"] = du_doan_ket_qua


@app.route('/')
def home():
    return "Chào mừng đến với API dự đoán Tài Xỉu trên Render! Truy cập /predict để xem dự đoán."

@app.route('/predict', methods=['GET'])
def get_prediction():
    """
    Endpoint chính để lấy dữ liệu mới nhất từ API bên ngoài, cập nhật trạng thái
    và trả về dự đoán cho phiên tiếp theo theo định dạng JSON mẫu.
    """
    global initial_api_data_template, last_prediction_info

    try:
        print(f"Calling external API: {EXTERNAL_API_URL}")
        response = requests.get(EXTERNAL_API_URL)
        response.raise_for_status()
        external_data = response.json()
        print(f"Data received from external API: {external_data}")

        if "Phien" in external_data and "Ket_qua" in external_data and "Tong" in external_data:
            new_session_data = external_data

            update_history_and_state(new_session_data)
            perform_prediction_logic()

            return jsonify(copy.deepcopy(initial_api_data_template)), 200
        else:
            error_message = "Invalid data format from external API. Missing 'Phien', 'Ket_qua', or 'Tong'."
            print(f"Error: {error_message} - Raw response: {external_data}")
            return jsonify({"error": error_message, "raw_response": external_data}), 500

    except requests.exceptions.RequestException as e:
        error_message = f"Error connecting to external API: {e}. Vui lòng kiểm tra URL và kết nối."
        print(f"Error: {error_message}")
        return jsonify({"error": error_message}), 500
    except Exception as e:
        error_message = f"Internal server error: {e}"
        print(f"Error: {error_message}")
        return jsonify({"error": error_message}), 500

@app.route('/status', methods=['GET'])
def get_current_status():
    """
    Endpoint để lấy trạng thái dự đoán hiện tại mà không gọi API bên ngoài.
    """
    return jsonify(copy.deepcopy(initial_api_data_template)), 200

@app.route('/history', methods=['GET'])
def get_history():
    """
    Endpoint để xem lịch sử các phiên đã được xử lý (trong bộ nhớ).
    """
    return jsonify(list(history_results)), 200

@app.route('/last_prediction_info', methods=['GET'])
def get_last_prediction_info_route():
    """
    Endpoint để xem thông tin về dự đoán gần nhất và số lần trật liên tiếp.
    """
    return jsonify(last_prediction_info), 200

# --- Chạy ứng dụng Flask ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

