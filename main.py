import os
import json
import time
import math
import random
import threading
import logging
from collections import defaultdict, deque
from flask import Flask, jsonify
from flask_cors import CORS
import requests # Import the requests library for HTTP requests

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Helper Functions ---
def _get_history_strings(history_list):
    """Hàm trợ giúp để lấy danh sách chuỗi 'Tài'/'Xỉu' từ danh sách dict."""
    return [item['ket_qua'] for item in history_list]


# 1. Định nghĩa các Patterns (Mở rộng với các pattern phức tạp và siêu phức tạp)
def define_patterns():
    """
    Định nghĩa một bộ sưu tập lớn các patterns từ đơn giản đến siêu phức tạp.
    Mỗi pattern là một hàm lambda nhận lịch sử (dạng chuỗi) và trả về True nếu khớp.
    """
    patterns = {
        # --- Cầu Bệt (Streaks) ---
        "Bệt": lambda h: len(h) >= 3 and h[-1] == h[-2] == h[-3],
        "Bệt siêu dài": lambda h: len(h) >= 5 and all(x == h[-1] for x in h[-5:]),
        "Bệt gãy nhẹ": lambda h: len(h) >= 4 and h[-1] != h[-2] and h[-2] == h[-3] == h[-4],
        "Bệt gãy sâu": lambda h: len(h) >= 5 and h[-1] != h[-2] and all(x == h[-2] for x in h[-5:-1]),
        "Bệt xen kẽ ngắn": lambda h: len(h) >= 4 and h[-4:-2] == [h[-4]]*2 and h[-2:] == [h[-2]]*2 and h[-4] != h[-2],
        "Bệt ngược": lambda h: len(h) >= 4 and h[-1] == h[-2] and h[-3] == h[-4] and h[-1] != h[-3],
        "Xỉu kép": lambda h: len(h) >= 2 and h[-1] == 'Xỉu' and h[-2] == 'Xỉu',
        "Tài kép": lambda h: len(h) >= 2 and h[-1] == 'Tài' and h[-2] == 'Tài',
        "Ngẫu nhiên bệt": lambda h: len(h) > 8 and 0.4 < (h[-8:].count('Tài') / 8) < 0.6 and h[-1] == h[-2],

        # --- Cầu Đảo (Alternating) ---
        "Đảo 1-1": lambda h: len(h) >= 4 and h[-1] != h[-2] and h[-2] != h[-3] and h[-3] != h[-4],
        "Xen kẽ dài": lambda h: len(h) >= 5 and all(h[i] != h[i+1] for i in range(-5, -1)),
        "Xen kẽ": lambda h: len(h) >= 3 and h[-1] != h[-2] and h[-2] != h[-3],
        "Xỉu lắc": lambda h: len(h) >= 4 and h[-4:] == ['Xỉu', 'Tài', 'Xỉu', 'Tài'],
        "Tài lắc": lambda h: len(h) >= 4 and h[-4:] == ['Tài', 'Xỉu', 'Tài', 'Xỉu'],
        
        # --- Cầu theo nhịp (Rhythmic) ---
        "Kép 2-2": lambda h: len(h) >= 4 and h[-4:] == [h[-4], h[-4], h[-2], h[-2]] and h[-4] != h[-2],
        "Nhịp 3-3": lambda h: len(h) >= 6 and all(x == h[-6] for x in h[-6:-3]) and all(x == h[-3] for x in h[-3:]),
        "Nhịp 4-4": lambda h: len(h) >= 8 and h[-8:-4] == [h[-8]]*4 and h[-4:] == [h[-4]]*4 and h[-8] != h[-4],
        "Lặp 2-1": lambda h: len(h) >= 3 and h[-3:-1] == [h[-3], h[-3]] and h[-1] != h[-3],
        "Lặp 3-2": lambda h: len(h) >= 5 and h[-5:-2] == [h[-5]]*3 and h[-2:] == [h[-2]]*2 and h[-5] != h[-2],
        "Cầu 3-1": lambda h: len(h) >= 4 and all(x == h[-4] for x in h[-4:-1]) and h[-1] != h[-4],
        "Cầu 4-1": lambda h: len(h) >= 5 and h[-5:-1] == [h[-5]]*4 and h[-1] != h[-5],
        "Cầu 1-2-1": lambda h: len(h) >= 4 and h[-4] != h[-3] and h[-3]==h[-2] and h[-2] != h[-1] and h[-4]==h[-1],
        "Cầu 2-1-2": lambda h: len(h) >= 5 and h[-5:-3] == [h[-5]]*2 and h[-2] != h[-5] and h[-1] == h[-5],
        "Cầu 3-1-2": lambda h: len(h) >= 6 and h[-6:-3]==[h[-6]]*3 and h[-3]!=h[-2] and h[-2:]==[h[-2]]*2 and len(set(h[-6:])) == 2,
        "Cầu 1-2-3": lambda h: len(h) >= 6 and h[-6:-5]==[h[-6]] and h[-5:-3]==[h[-5]]*2 and h[-3:]==[h[-3]]*3 and len(set(h[-6:])) == 2,
        "Dài ngắn đảo": lambda h: len(h) >= 5 and h[-5:-2] == [h[-5]] * 3 and h[-2] != h[-1] and h[-2] != h[-5],

        # --- Cầu Chu Kỳ & Đối Xứng (Cyclic & Symmetric) ---
        "Chu kỳ 2": lambda h: len(h) >= 4 and h[-1] == h[-3] and h[-2] == h[-4],
        "Chu kỳ 3": lambda h: len(h) >= 6 and h[-1] == h[-4] and h[-2] == h[-5] and h[-3] == h[-6],
        "Chu kỳ 4": lambda h: len(h) >= 8 and h[-8:-4] == h[-4:],
        "Đối xứng (Gương)": lambda h: len(h) >= 5 and h[-1] == h[-5] and h[-2] == h[-4],
        "Bán đối xứng": lambda h: len(h) >= 5 and h[-1] == h[-4] and h[-2] == h[-5],
        "Ngược chu kỳ": lambda h: len(h) >= 4 and h[-1] == h[-4] and h[-2] == h[-3] and h[-1] != h[-2],
        "Chu kỳ biến đổi": lambda h: len(h) >= 5 and h[-5:] == [h[-5], h[-4], h[-5], h[-4], h[-5]],
        "Cầu linh hoạt": lambda h: len(h) >= 6 and h[-1]==h[-3]==h[-5] and h[-2]==h[-4]==h[-6],
        "Chu kỳ tăng": lambda h: len(h) >= 6 and h[-6:] == [h[-6], h[-5], h[-6], h[-5], h[-6], h[-5]] and h[-6] != h[-5],
        "Chu kỳ giảm": lambda h: len(h) >= 6 and h[-6:] == [h[-6], h[-6], h[-5], h[-5], h[-4], h[-4]] and len(set(h[-6:])) == 3,
        "Cầu lặp": lambda h: len(h) >= 6 and h[-6:-3] == h[-3:],
        "Gãy ngang": lambda h: len(h) >= 4 and h[-1] == h[-3] and h[-2] == h[-4] and h[-1] != h[-2],

        # --- Cầu Phức Tạp & Tổng Hợp ---
        "Gập ghềnh": lambda h: len(h) >= 5 and h[-5:] == [h[-5], h[-5], h[-3], h[-3], h[-5]],
        "Bậc thang": lambda h: len(h) >= 3 and h[-3:] == [h[-3], h[-3], h[-1]] and h[-3] != h[-1],
        "Cầu đôi": lambda h: len(h) >= 4 and h[-1] == h[-2] and h[-3] != h[-4] and h[-3] != h[-1],
        "Đối ngược": lambda h: len(h) >= 4 and h[-1] == ('Xỉu' if h[-2]=='Tài' else 'Tài') and h[-3] == ('Xỉu' if h[-4]=='Tài' else 'Tài'),
        "Cầu gập": lambda h: len(h) >= 5 and h[-5:] == [h[-5], h[-4], h[-4], h[-2], h[-2]],
        "Phối hợp 1": lambda h: len(h) >= 5 and h[-1] == h[-2] and h[-3] != h[-4],
        "Phối hợp 2": lambda h: len(h) >= 4 and h[-4:] == ['Tài', 'Tài', 'Xỉu', 'Tài'],
        "Phối hợp 3": lambda h: len(h) >= 4 and h[-4:] == ['Xỉu', 'Xỉu', 'Tài', 'Xỉu'],
        "Chẵn lẻ lặp": lambda h: len(h) >= 4 and len(set(h[-4:-2])) == 1 and len(set(h[-2:])) == 1 and h[-1] != h[-3],
        "Cầu dài ngẫu": lambda h: len(h) >= 7 and all(x == h[-7] for x in h[-7:-3]) and len(set(h[-3:])) > 1,
        
        # --- Cầu Dựa Trên Phân Bố (Statistical) ---
        "Ngẫu nhiên": lambda h: len(h) > 10 and 0.4 < (h[-10:].count('Tài') / 10) < 0.6,
        "Đa dạng": lambda h: len(h) >= 5 and len(set(h[-5:])) == 2,
        "Phân cụm": lambda h: len(h) >= 6 and (all(x == 'Tài' for x in h[-6:-3]) or all(x == 'Xỉu' for x in h[-6:-3])),
        "Lệch ngẫu nhiên": lambda h: len(h) > 10 and (h[-10:].count('Tài') / 10 > 0.7 or h[-10:].count('Xỉu') / 10 > 0.7),

        # --- Siêu Cầu (Super Patterns) ---
        "Cầu Tiến 1-1-2-2": lambda h: len(h) >= 6 and h[-6:] == [h[-6], h[-5], h[-4], h[-4], h[-2], h[-2]] and len(set(h[-6:])) == 2,
        "Cầu Lùi 3-2-1": lambda h: len(h) >= 6 and h[-6:-3]==[h[-6]]*3 and h[-3:-1]==[h[-3]]*2 and h[-1]!=h[-3] and len(set(h[-6:])) == 2,
        "Cầu Sandwich": lambda h: len(h) >= 5 and h[-1] == h[-5] and h[-2] == h[-3] == h[-4] and h[-1] != h[-2],
        "Cầu Thang máy": lambda h: len(h) >= 7 and h[-7:] == [h[-7],h[-7],h[-5],h[-5],h[-3],h[-3],h[-1]] and len(set(h[-7:]))==4, # T-T-X-X-T-T-X
        "Cầu Sóng vỗ": lambda h: len(h) >= 8 and h[-8:] == [h[-8],h[-8],h[-6],h[-8],h[-8],h[-6],h[-8],h[-8]],
    }
    return patterns

# 2. Các hàm cập nhật và huấn luyện mô hình (IMPROVED)
def update_transition_matrix(app, prev_result, current_result):
    if not prev_result: return
    prev_idx = 0 if prev_result == 'Tài' else 1
    curr_idx = 0 if current_result == 'Tài' else 1
    app.transition_counts[prev_idx][curr_idx] += 1
    total_transitions = sum(app.transition_counts[prev_idx])
    alpha = 1 # Laplace smoothing
    num_outcomes = 2
    app.transition_matrix[prev_idx][0] = (app.transition_counts[prev_idx][0] + alpha) / (total_transitions + alpha * num_outcomes)
    app.transition_matrix[prev_idx][1] = (app.transition_counts[prev_idx][1] + alpha) / (total_transitions + alpha * num_outcomes)

def update_pattern_accuracy(app, predicted_pattern_name, prediction, actual_result):
    if not predicted_pattern_name: return
    stats = app.pattern_accuracy[predicted_pattern_name]
    stats['total'] += 1
    if prediction == actual_result:
        stats['success'] += 1

def train_logistic_regression(app, features, actual_result):
    y = 1.0 if actual_result == 'Tài' else 0.0
    z = app.logistic_bias + sum(w * f for w, f in zip(app.logistic_weights, features))
    try:
        p = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        p = 0.0 if z < 0 else 1.0
        
    error = y - p
    app.logistic_bias += app.learning_rate * error
    for i in range(len(app.logistic_weights)):
        gradient = error * features[i]
        regularization_term = app.regularization * app.logistic_weights[i]
        app.logistic_weights[i] += app.learning_rate * (gradient - regularization_term)

def update_model_weights(app):
    """Cập nhật trọng số của các mô hình trong ensemble dựa trên hiệu suất."""
    total_accuracy = 0
    accuracies = {}
    for name, perf in app.model_performance.items():
        if perf['total'] > 5: # Chỉ cập nhật nếu có đủ dữ liệu
            accuracy = perf['success'] / perf['total']
            accuracies[name] = accuracy
            total_accuracy += accuracy
        else: # Giữ trọng số mặc định nếu chưa đủ dữ liệu
            accuracies[name] = app.default_model_weights[name] * 2 # Tạm thời
            total_accuracy += app.default_model_weights[name] * 2

    if total_accuracy > 0:
        for name in app.model_weights:
            app.model_weights[name] = accuracies[name] / total_accuracy
    # Normalize lại để tổng bằng 1
    total_weight = sum(app.model_weights.values())
    if total_weight > 0:
        for name in app.model_weights:
            app.model_weights[name] /= total_weight
    logging.info(f"Updated model weights: {app.model_weights}")


# 3. Các hàm dự đoán cốt lõi (IMPROVED)
def detect_pattern(app, history_str):
    detected_patterns = []
    if len(history_str) < 2: return None
    
    total_occurrences = max(1, sum(s['total'] for s in app.pattern_accuracy.values()))

    for name, func in app.patterns.items():
        try:
            if func(history_str):
                stats = app.pattern_accuracy[name]
                accuracy = (stats['success'] / stats['total']) if stats['total'] > 10 else 0.55 # Ưu tiên nhẹ nếu chưa đủ dữ liệu
                recency_score = stats['total'] / total_occurrences
                
                # Trọng số kết hợp độ chính xác lịch sử (70%) và tần suất xuất hiện (30%)
                weight = 0.7 * accuracy + 0.3 * recency_score
                detected_patterns.append({'name': name, 'weight': weight})
        except IndexError:
            continue
    if not detected_patterns:
        return None
    return max(detected_patterns, key=lambda x: x['weight'])

def predict_with_pattern(app, history_str, detected_pattern_info):
    if not detected_pattern_info or len(history_str) < 2:
        return 'Tài', 0.5
    
    name = detected_pattern_info['name']
    last = history_str[-1]
    prev = history_str[-2]
    anti_last = 'Xỉu' if last == 'Tài' else 'Tài'

    # Logic dự đoán chi tiết hơn dựa trên loại pattern
    if any(p in name for p in ["Bệt", "kép", "2-2", "3-3", "4-4", "Nhịp", "Sóng vỗ"]):
        prediction = last # Theo cầu
    elif any(p in name for p in ["Đảo 1-1", "Xen kẽ", "lắc", "Đối ngược", "gãy", "Bậc thang"]):
        prediction = anti_last # Bẻ cầu
    elif any(p in name for p in ["Chu kỳ 2", "Gãy ngang"]):
        prediction = prev
    elif 'Chu kỳ 3' in name:
        prediction = history_str[-3]
    elif 'Chu kỳ 4' in name:
        prediction = history_str[-4]
    elif name == "Cầu 2-1-2":
        prediction = history_str[-5]
    elif name == "Cầu 1-2-1":
        prediction = anti_last
    elif name == "Đối xứng (Gương)":
        prediction = history_str[-3] # Dự đoán phần tử tiếp theo trong chuỗi đối xứng
    elif name == "Cầu lặp":
        prediction = history_str[-3]
    elif name == "Cầu 3-1" or name == "Cầu 4-1":
        prediction = last # Bẻ xong thường sẽ quay lại cầu cũ
    else: # Mặc định cho các cầu phức tạp khác là bẻ cầu
        prediction = anti_last
        
    return prediction, detected_pattern_info['weight']

def get_logistic_features(history_str):
    if not history_str: return [0.0] * 6
        
    # Feature 1: Current streak length
    current_streak = 0
    if len(history_str) > 0:
        last = history_str[-1]
        current_streak = 1
        for i in range(len(history_str) - 2, -1, -1):
            if history_str[i] == last: current_streak += 1
            else: break
    
    # Feature 2: Previous streak length
    previous_streak_len = 0
    if len(history_str) > current_streak:
        prev_streak_start_idx = len(history_str) - current_streak -1
        prev_streak_val = history_str[prev_streak_start_idx]
        previous_streak_len = 1
        for i in range(prev_streak_start_idx -1, -1, -1):
            if history_str[i] == prev_streak_val: previous_streak_len += 1
            else: break

    # Feature 3 & 4: Balance (Tài-Xỉu) short-term and long-term
    recent_history = history_str[-20:]
    balance_short = (recent_history.count('Tài') - recent_history.count('Xỉu')) / max(1, len(recent_history))
    
    long_history = history_str[-100:]
    balance_long = (long_history.count('Tài') - long_history.count('Xỉu')) / max(1, len(long_history))
    
    # Feature 5: Volatility (tần suất thay đổi)
    changes = sum(1 for i in range(len(recent_history)-1) if recent_history[i] != recent_history[i+1])
    volatility = changes / max(1, len(recent_history) -1) if len(recent_history) > 1 else 0.0

    # Feature 6: Alternation count in last 10 results
    last_10 = history_str[-10:]
    alternations = sum(1 for i in range(len(last_10) - 1) if last_10[i] != last_10[i+1])
    
    return [float(current_streak), float(previous_streak_len), balance_short, balance_long, volatility, float(alternations)]

def apply_meta_logic(prediction, confidence, history_str):
    """
    Áp dụng logic cấp cao để điều chỉnh dự đoán cuối cùng.
    Ví dụ: Logic "bẻ cầu" khi cầu quá dài.
    """
    final_prediction, final_confidence, reason = prediction, confidence, ""

    # Logic 1: Bẻ cầu khi cầu bệt quá dài (Anti-Streak)
    streak_len = 0
    if len(history_str) > 1:
        last = history_str[-1]
        for x in reversed(history_str):
            if x == last: streak_len += 1
            else: break
    
    if streak_len >= 9 and prediction == history_str[-1]:
        final_prediction = 'Xỉu' if history_str[-1] == 'Tài' else 'Tài'
        final_confidence = 78.0 # Gán một độ tin cậy khá cao cho việc bẻ cầu
        reason = f"Bẻ cầu bệt siêu dài ({streak_len})"
        logging.warning(f"META-LOGIC: Activated Anti-Streak. Streak of {streak_len} detected. Forcing prediction to {final_prediction}.")
    elif streak_len >= 7 and prediction == history_str[-1]:
        final_confidence = max(50.0, confidence - 15) # Giảm độ tin cậy
        reason = f"Cầu bệt dài ({streak_len}), giảm độ tin cậy"
        logging.info(f"META-LOGIC: Long streak of {streak_len} detected. Reducing confidence.")
        
    return final_prediction, final_confidence, reason


def predict_advanced(app, history_str):
    """Hàm điều phối dự đoán nâng cao, kết hợp nhiều mô hình với trọng số động."""
    if len(history_str) < 5:
        return "Chờ dữ liệu", "Phân tích", 50.0, {}

    last_result = history_str[-1]

    # --- Model 1: Pattern Matching ---
    detected_pattern_info = detect_pattern(app, history_str)
    patt_pred, patt_conf = predict_with_pattern(app, history_str, detected_pattern_info)

    # --- Model 2: Markov Chain ---
    last_result_idx = 0 if last_result == 'Tài' else 1
    prob_tai_markov = app.transition_matrix[last_result_idx][0]
    markov_pred = 'Tài' if prob_tai_markov > 0.5 else 'Xỉu'
    markov_conf = max(prob_tai_markov, 1 - prob_tai_markov)

    # --- Model 3: Logistic Regression ---
    features = get_logistic_features(history_str)
    z = app.logistic_bias + sum(w * f for w, f in zip(app.logistic_weights, features))
    try:
        prob_tai_logistic = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        prob_tai_logistic = 0.0 if z < 0 else 1.0
        
    logistic_pred = 'Tài' if prob_tai_logistic > 0.5 else 'Xỉu'
    logistic_conf = max(prob_tai_logistic, 1 - prob_tai_logistic)
    
    # Lưu lại dự đoán của từng mô hình để học
    individual_predictions = {
        'pattern': patt_pred,
        'markov': markov_pred,
        'logistic': logistic_pred
    }

    # --- Ensemble Prediction (Kết hợp các mô hình với trọng số động) ---
    predictions = {
        'pattern': {'pred': patt_pred, 'conf': patt_conf, 'weight': app.model_weights['pattern']},
        'markov': {'pred': markov_pred, 'conf': markov_conf, 'weight': app.model_weights['markov']},
        'logistic': {'pred': logistic_pred, 'conf': logistic_conf, 'weight': app.model_weights['logistic']},
    }
    
    tai_score, xiu_score = 0.0, 0.0
    for model in predictions.values():
        score = model['conf'] * model['weight']
        if model['pred'] == 'Tài': tai_score += score
        else: xiu_score += score

    final_prediction = 'Tài' if tai_score > xiu_score else 'Xỉu'
    total_score = tai_score + xiu_score
    final_confidence = (max(tai_score, xiu_score) / total_score * 100) if total_score > 0 else 50.0
    
    # Tăng độ tin cậy nếu pattern mạnh nhất trùng với dự đoán cuối cùng
    if detected_pattern_info and detected_pattern_info['weight'] > 0.6 and patt_pred == final_prediction:
        final_confidence = min(98.0, final_confidence + patt_conf * 10)

    # Áp dụng logic meta cuối cùng
    final_prediction, final_confidence, meta_reason = apply_meta_logic(final_prediction, final_confidence, history_str)

    used_pattern_name = detected_pattern_info['name'] if detected_pattern_info else "Ensemble"
    if meta_reason:
        used_pattern_name = meta_reason

    return final_prediction, used_pattern_name, final_confidence, individual_predictions

# --- END: Improved Logic ---


# --- Flask App Factory ---
def create_app():
    app = Flask(__name__)
    CORS(app)

    # --- Khởi tạo State ---
    app.lock = threading.Lock()
    app.MAX_HISTORY_LEN = 200
    
    app.history = deque(maxlen=app.MAX_HISTORY_LEN)
    app.session_ids = deque(maxlen=app.MAX_HISTORY_LEN)
    
    # State cho các thuật toán
    app.patterns = define_patterns()
    app.transition_matrix = [[0.5, 0.5], [0.5, 0.5]]
    app.transition_counts = [[0, 0], [0, 0]]
    app.logistic_weights = [0.0] * 6 # Mở rộng cho 6 features
    app.logistic_bias = 0.0
    app.learning_rate = 0.01
    app.regularization = 0.01
    
    # State cho ensemble model động
    app.default_model_weights = {'pattern': 0.5, 'markov': 0.2, 'logistic': 0.3}
    app.model_weights = app.default_model_weights.copy()
    app.model_performance = {name: {"success": 0, "total": 0} for name in app.model_weights}

    app.last_prediction = None
    app.pattern_accuracy = defaultdict(lambda: {"success": 0, "total": 0})
    
    # --- Xử lý API HTTP ---
    app.API_URL = os.getenv("API_URL", "https://apib52.up.railway.app/api/taixiumd5")

    def fetch_data_from_api():
        try:
            response = requests.get(app.API_URL, timeout=5)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data from API: {e}")
            return None

    def data_fetch_loop():
        last_processed_session = None
        while True:
            data = fetch_data_from_api()
            if data:
                try:
                    phien = data.get("Phien")
                    ket_qua = data.get("Ket_qua")

                    if phien is None or ket_qua not in ["Tài", "Xỉu"]:
                        logging.warning(f"Invalid data received from API: {data}")
                        time.sleep(1) # Wait a bit before retrying
                        continue

                    with app.lock:
                        # Only add new data if the session is newer than the last processed one
                        if not app.session_ids or phien > app.session_ids[-1]:
                            app.session_ids.append(phien)
                            app.history.append({'ket_qua': ket_qua, 'phien': phien})
                            logging.info(f"New result for session {phien}: {ket_qua}")
                            last_processed_session = phien
                        else:
                            # If the session is not newer, log that we've already processed it
                            if phien == app.session_ids[-1]:
                                logging.debug(f"Session {phien} already processed.")
                            else:
                                logging.debug(f"Received older session {phien}. Current latest is {app.session_ids[-1]}.")
                except (json.JSONDecodeError, TypeError) as e:
                    logging.error(f"Error decoding or processing API data: {e} - Data: {data}")
                except Exception as e:
                    logging.error(f"Unexpected error in data_fetch_loop: {e}")
            
            time.sleep(1) # Poll the API every 1 second

    # --- API Endpoints ---
    @app.route("/api/taixiu_ws", methods=["GET"]) # Renamed from taixiu_ws to avoid confusion, but kept for compatibility
    def get_taixiu_prediction():
        with app.lock:
            if len(app.history) < 2:
                return jsonify({"error": "Chưa có đủ dữ liệu"}), 500
            
            history_copy = list(app.history)
            session_ids_copy = list(app.session_ids)
            last_prediction_copy = app.last_prediction
        
        # --- Bước học Online (Online Learning) ---
        # Only learn if a new result is available for the session that was predicted
        # Ensure we have at least one past result to compare with for learning
        if last_prediction_copy and session_ids_copy and last_prediction_copy['session'] == session_ids_copy[-1]:
            prev_history_str = _get_history_strings(history_copy[:-1])
            actual_result = history_copy[-1]['ket_qua']
            
            # Cập nhật các mô hình (phải khóa vì thay đổi state chung)
            with app.lock:
                # Học cho Logistic Regression
                train_logistic_regression(app, last_prediction_copy['features'], actual_result)
                # Học cho Markov Chain
                if len(prev_history_str) > 0:
                     update_transition_matrix(app, prev_history_str[-1], actual_result)
                # Học cho Pattern
                update_pattern_accuracy(app, last_prediction_copy['pattern'], last_prediction_copy['prediction'], actual_result)
                
                # Cập nhật hiệu suất của từng mô hình con để điều chỉnh trọng số
                for model_name, model_pred in last_prediction_copy['individual_predictions'].items():
                    app.model_performance[model_name]['total'] += 1
                    if model_pred == actual_result:
                        app.model_performance[model_name]['success'] += 1
                
                # Cập nhật lại trọng số của ensemble model
                update_model_weights(app)

            logging.info(f"Learned from session {session_ids_copy[-1]}. Prediction was {last_prediction_copy['prediction']}, actual was {actual_result}. Pattern: {last_prediction_copy['pattern']}")

        # --- Bước Dự đoán (Prediction) ---
        history_str_for_prediction = _get_history_strings(history_copy)
        prediction_str, pattern_str, confidence, individual_preds = predict_advanced(app, history_str_for_prediction)
        
        # Lưu lại thông tin dự đoán để học ở lần tiếp theo
        with app.lock:
            current_session = session_ids_copy[-1] if session_ids_copy else None # Handle case where history is empty
            if current_session is not None:
                app.last_prediction = {
                    'session': current_session + 1,
                    'prediction': prediction_str,
                    'pattern': pattern_str,
                    'features': get_logistic_features(history_str_for_prediction),
                    'individual_predictions': individual_preds,
                }
            current_result = history_copy[-1]['ket_qua'] if history_copy else None
        
        # Tinh chỉnh hiển thị
        prediction_display = prediction_str
        if confidence < 75.0 and "Bẻ cầu" not in pattern_str:
            prediction_display = f"Đang phân tích"
            final_confidence_display = round(confidence, 1)
        else:
            final_confidence_display = round(confidence, 1)

        return jsonify({
            "current_session": current_session,
            "current_result": current_result,
            "next_session": current_session + 1 if current_session is not None else None,
            "prediction": prediction_display,
            "confidence_percent": final_confidence_display,
            "suggested_pattern": pattern_str,
        })

    @app.route("/api/history", methods=["GET"])
    def get_history_api():
        with app.lock:
            hist_copy = list(app.history)
        return jsonify({"history": hist_copy, "length": len(hist_copy)})

    @app.route("/api/performance", methods=["GET"])
    def get_performance():
        with app.lock:
            # Sắp xếp pattern theo tổng số lần xuất hiện và độ chính xác
            seen_patterns = {k: v for k, v in app.pattern_accuracy.items() if v['total'] > 0}
            sorted_patterns = sorted(
                seen_patterns.items(), 
                key=lambda item: (item[1]['total'], (item[1]['success'] / item[1]['total'] if item[1]['total'] > 0 else 0)),
                reverse=True
            )
            pattern_result = {}
            for p_type, data in sorted_patterns[:30]: # Lấy 30 pattern hàng đầu
                accuracy = round(data["success"] / data["total"] * 100, 2) if data["total"] > 0 else 0
                pattern_result[p_type] = { "total": data["total"], "success": data["success"], "accuracy_percent": accuracy }
            
            # Lấy hiệu suất của các mô hình con
            model_perf_result = {}
            for name, perf in app.model_performance.items():
                 accuracy = round(perf["success"] / perf["total"] * 100, 2) if perf["total"] > 0 else 0
                 model_perf_result[name] = {**perf, "accuracy_percent": accuracy}

        return jsonify({
            "pattern_performance": pattern_result,
            "model_performance": model_perf_result,
            "model_weights": app.model_weights
        })

    # Start the HTTP data fetching thread
    api_fetch_thread = threading.Thread(target=data_fetch_loop, daemon=True)
    api_fetch_thread.start()
    return app

# --- Thực thi chính ---
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"Flask app ready. Serving on http://0.0.0.0:{port}")
    from waitress import serve
    serve(app, host="0.0.0.0", port=port, threads=8)
