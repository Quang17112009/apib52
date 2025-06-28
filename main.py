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
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np # Thêm numpy để xử lý mảng

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Helper Functions ---
def _get_history_strings(history_list):
    """Hàm trợ giúp để lấy danh sách chuỗi 'Tài'/'Xỉu' từ danh sách dict."""
    return [item['ket_qua'] for item in history_list]

# --- Model Persistence Paths ---
MODEL_STATE_FILE = "model_state.json"
RF_MODEL_FILE = "random_forest_model.joblib"
META_MODEL_FILE = "meta_model.joblib"

# --- Model Persistence Functions ---
def save_model_state(app):
    state = {
        'transition_counts': app.transition_counts,
        'logistic_weights': app.logistic_weights.tolist() if isinstance(app.logistic_weights, np.ndarray) else app.logistic_weights, # Convert numpy array to list
        'logistic_bias': app.logistic_bias,
        'pattern_accuracy': dict(app.pattern_accuracy),
        'model_performance': app.model_performance,
        'model_weights': app.model_weights,
        'rf_meta_training_data': app.rf_meta_training_data # Save training data for RF/Meta
    }
    with open(MODEL_STATE_FILE, 'w') as f:
        json.dump(state, f)
    
    if hasattr(app, 'random_forest_model') and app.random_forest_model is not None and hasattr(app.random_forest_model, 'classes_'):
        joblib.dump(app.random_forest_model, RF_MODEL_FILE)
    else:
        # Nếu mô hình chưa được huấn luyện, xóa file cũ nếu có
        if os.path.exists(RF_MODEL_FILE):
            os.remove(RF_MODEL_FILE)

    if hasattr(app, 'meta_model') and app.meta_model is not None and hasattr(app.meta_model, 'classes_'):
        joblib.dump(app.meta_model, META_MODEL_FILE)
    else:
        # Nếu mô hình chưa được huấn luyện, xóa file cũ nếu có
        if os.path.exists(META_MODEL_FILE):
            os.remove(META_MODEL_FILE)
    logging.info("Model state saved.")

def load_model_state(app):
    try:
        with open(MODEL_STATE_FILE, 'r') as f:
            state = json.load(f)
            app.transition_counts = state.get('transition_counts', [[0, 0], [0, 0]])
            app.logistic_weights = np.array(state.get('logistic_weights', [0.0] * 12)) # Convert list to numpy array
            app.logistic_bias = state.get('logistic_bias', 0.0)
            app.pattern_accuracy = defaultdict(lambda: {"success": 0, "total": 0}, state.get('pattern_accuracy', {}))
            app.model_performance = state.get('model_performance', {name: {"success": 0, "total": 0} for name in app.default_model_weights})
            app.model_weights = state.get('model_weights', app.default_model_weights.copy())
            app.rf_meta_training_data = state.get('rf_meta_training_data', []) # Load training data for RF/Meta

        if os.path.exists(RF_MODEL_FILE):
            app.random_forest_model = joblib.load(RF_MODEL_FILE)
            logging.info("Random Forest model loaded.")
        else:
            app.random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            logging.info("Random Forest model initialized (no saved model found).")

        if os.path.exists(META_MODEL_FILE):
            app.meta_model = joblib.load(META_MODEL_FILE)
            logging.info("Meta-model loaded.")
        else:
            app.meta_model = LogisticRegression(random_state=42, solver='liblinear')
            logging.info("Meta-model initialized (no saved model found).")

        logging.info("Model state loaded.")
    except FileNotFoundError:
        logging.warning("No saved model state found. Initializing new models.")
        _initialize_new_models(app) # Use helper for fresh initialization
    except Exception as e:
        logging.error(f"Error loading model state: {e}. Initializing new models.")
        _initialize_new_models(app) # Use helper for fresh initialization

def _initialize_new_models(app):
    app.transition_counts = [[0, 0], [0, 0]]
    app.logistic_weights = np.array([0.0] * 12)
    app.logistic_bias = 0.0
    app.pattern_accuracy = defaultdict(lambda: {"success": 0, "total": 0})
    app.model_performance = {name: {"success": 0, "total": 0} for name in app.default_model_weights}
    app.model_weights = app.default_model_weights.copy()
    
    app.random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    app.meta_model = LogisticRegression(random_state=42, solver='liblinear')
    app.rf_meta_training_data = [] # Reset training data

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
    # Ensure features is a numpy array for dot product
    features_np = np.array(features)
    y = 1.0 if actual_result == 'Tài' else 0.0
    z = app.logistic_bias + np.dot(app.logistic_weights, features_np)
    try:
        p = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        p = 0.0 if z < 0 else 1.0
        
    error = y - p
    app.logistic_bias += app.learning_rate * error
    app.logistic_weights += app.learning_rate * (error * features_np - app.regularization * app.logistic_weights)

def update_model_weights(app):
    """Cập nhật trọng số của các mô hình trong ensemble dựa trên hiệu suất."""
    total_accuracy = 0
    accuracies = {}
    for name, perf in app.model_performance.items():
        if perf['total'] > 5: # Chỉ cập nhật nếu có đủ dữ liệu huấn luyện
            accuracy = perf['success'] / perf['total']
            accuracies[name] = accuracy
            total_accuracy += accuracy
        else: # Giữ trọng số mặc định nếu chưa đủ dữ liệu
            accuracies[name] = app.default_model_weights[name] * 2 # Tạm thời tăng để có trọng số ban đầu
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
                # Ưu tiên nhẹ hơn nếu chưa đủ dữ liệu hoặc độ chính xác thấp (tránh pattern ngẫu nhiên)
                accuracy = (stats['success'] / stats['total']) if stats['total'] > 10 else 0.55 
                recency_score = (stats['total'] / total_occurrences) if total_occurrences > 0 else 0
                
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
    if any(p in name for p in ["Bệt", "kép", "2-2", "3-3", "4-4", "Nhịp", "Sóng vỗ", "Cầu 3-1", "Cầu 4-1", "Dài ngắn đảo"]):
        prediction = last # Theo cầu
    elif any(p in name for p in ["Đảo 1-1", "Xen kẽ", "lắc", "Đối ngược", "gãy", "Bậc thang", "Cầu đôi"]):
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
        prediction = history_str[-3]
    elif name == "Cầu lặp":
        prediction = history_str[-3]
    elif name == "Gập ghềnh":
        prediction = history_str[-1]
    elif name == "Phối hợp 2":
        prediction = 'Xỉu' # Tài Tài Xỉu Tài -> Xỉu
    elif name == "Phối hợp 3":
        prediction = 'Tài' # Xỉu Xỉu Tài Xỉu -> Tài
    else: # Mặc định cho các cầu phức tạp khác là bẻ cầu
        prediction = anti_last
        
    return prediction, detected_pattern_info['weight']

def get_advanced_features(history_str):
    if not history_str or len(history_str) < 5: 
        # Cần ít nhất 5 kết quả để có các features này
        # Trả về một danh sách các số 0 nếu không đủ lịch sử
        return np.array([0.0] * 12) # Đảm bảo trả về numpy array với 12 features

    last = history_str[-1]
    anti_last = 'Xỉu' if last == 'Tài' else 'Tài'
    
    # Feature Group 1: Streak features
    current_streak = 0
    for i in range(len(history_str) - 1, -1, -1):
        if history_str[i] == last:
            current_streak += 1
        else:
            break
    
    previous_streak_len = 0
    if len(history_str) > current_streak:
        prev_streak_val = history_str[len(history_str) - current_streak - 1]
        for i in range(len(history_str) - current_streak - 1, -1, -1):
            if history_str[i] == prev_streak_val:
                previous_streak_len += 1
            else:
                break

    # Feature Group 2: Balance and Volatility
    recent_history_10 = history_str[-10:]
    recent_history_20 = history_str[-20:]
    long_history_50 = history_str[-50:]

    balance_10 = (recent_history_10.count('Tài') - recent_history_10.count('Xỉu')) / max(1, len(recent_history_10))
    balance_20 = (recent_history_20.count('Tài') - recent_history_20.count('Xỉu')) / max(1, len(recent_history_20))
    balance_50 = (long_history_50.count('Tài') - long_history_50.count('Xỉu')) / max(1, len(long_history_50))

    changes_10 = sum(1 for i in range(len(recent_history_10)-1) if recent_history_10[i] != recent_history_10[i+1])
    volatility_10 = changes_10 / max(1, len(recent_history_10) - 1) if len(recent_history_10) > 1 else 0.0

    changes_20 = sum(1 for i in range(len(recent_history_20)-1) if recent_history_20[i] != recent_history_20[i+1])
    volatility_20 = changes_20 / max(1, len(recent_history_20) - 1) if len(recent_history_20) > 1 else 0.0

    # Feature Group 3: Specific Pattern Indicators (simplified)
    is_alternating_4 = 1.0 if len(history_str) >= 4 and all(history_str[i] != history_str[i+1] for i in range(-4, -1)) else 0.0
    is_alternating_6 = 1.0 if len(history_str) >= 6 and all(history_str[i] != history_str[i+1] for i in range(-6, -1)) else 0.0
    
    # Feature 4: Time since last appearance of 'Tài' or 'Xỉu' (conceptual)
    count_anti_last_10 = recent_history_10.count('Xỉu' if last == 'Tài' else 'Tài')

    # Feature 5: Check for specific mini-patterns
    is_triple_same_last = 1.0 if len(history_str) >= 3 and history_str[-3:] == [last, last, last] else 0.0
    is_double_broken = 1.0 if len(history_str) >= 4 and history_str[-4:] == [last, last, anti_last, anti_last] else 0.0

    return np.array([ # Trả về numpy array
        float(current_streak),
        float(previous_streak_len),
        balance_10,
        balance_20,
        balance_50,
        volatility_10,
        volatility_20,
        is_alternating_4,
        is_alternating_6,
        count_anti_last_10,
        is_triple_same_last,
        is_double_broken
    ])

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
    
    if streak_len >= 9 and prediction == history_str[-1]: # Ngưỡng bẻ cầu
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
    """Hàm điều phối dự đoán nâng cao, kết hợp nhiều mô hình với trọng số động và Stacking."""
    if len(history_str) < 15: # Cần nhiều lịch sử hơn cho các features nâng cao và RF
        return "Chờ dữ liệu", "Phân tích", 50.0, {}

    last_result = history_str[-1]
    features = get_advanced_features(history_str)

    # --- Model 1: Pattern Matching ---
    detected_pattern_info = detect_pattern(app, history_str)
    patt_pred, patt_conf = predict_with_pattern(app, history_str, detected_pattern_info)
    patt_pred_val = 1 if patt_pred == 'Tài' else 0 # Chuyển đổi sang giá trị số

    # --- Model 2: Markov Chain ---
    last_result_idx = 0 if last_result == 'Tài' else 1
    prob_tai_markov = app.transition_matrix[last_result_idx][0]
    markov_pred = 'Tài' if prob_tai_markov > 0.5 else 'Xỉu'
    markov_conf = abs(prob_tai_markov - 0.5) * 2 # Chuyển đổi xác suất thành độ tin cậy [0,1]
    markov_pred_val = 1 if markov_pred == 'Tài' else 0

    # --- Model 3: Logistic Regression ---
    z = app.logistic_bias + np.dot(app.logistic_weights, features)
    try:
        prob_tai_logistic = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        prob_tai_logistic = 0.0 if z < 0 else 1.0
        
    logistic_pred = 'Tài' if prob_tai_logistic > 0.5 else 'Xỉu'
    logistic_conf = abs(prob_tai_logistic - 0.5) * 2
    logistic_pred_val = 1 if logistic_pred == 'Tài' else 0

    # --- Model 4: Random Forest ---
    rf_pred_val = 0 # Mặc định là Xỉu
    rf_conf = 0.5
    rf_pred = 'Xỉu'
    if hasattr(app, 'random_forest_model') and app.random_forest_model is not None:
        try:
            if hasattr(app.random_forest_model, 'classes_'): # Chỉ dự đoán nếu mô hình đã được huấn luyện (fit)
                rf_prob_tai = app.random_forest_model.predict_proba([features])[0][1]
                rf_pred = 'Tài' if rf_prob_tai > 0.5 else 'Xỉu'
                rf_conf = abs(rf_prob_tai - 0.5) * 2
                rf_pred_val = 1 if rf_pred == 'Tài' else 0
            else:
                logging.debug("Random Forest model not yet fitted.")
        except Exception as e:
            logging.error(f"Error predicting with Random Forest: {e}")

    # Thu thập dự đoán (dạng số) từ các mô hình cấp 1 (Base models)
    base_predictions = np.array([patt_pred_val, markov_pred_val, logistic_pred_val, rf_pred_val])

    # --- Model 5: Meta-model (Stacking) ---
    meta_pred = 'Xỉu' # Mặc định
    meta_conf = 0.5
    if hasattr(app, 'meta_model') and app.meta_model is not None:
        try:
            if hasattr(app.meta_model, 'classes_'): # Chỉ dự đoán nếu mô hình đã được huấn luyện
                meta_prob_tai = app.meta_model.predict_proba([base_predictions])[0][1]
                meta_pred = 'Tài' if meta_prob_tai > 0.5 else 'Xỉu'
                meta_conf = abs(meta_prob_tai - 0.5) * 2
            else:
                logging.debug("Meta-model not yet fitted.")
        except Exception as e:
            logging.error(f"Error predicting with Meta-model: {e}")
            
    # Lưu lại dự đoán của từng mô hình để học
    individual_predictions = {
        'pattern': patt_pred,
        'markov': markov_pred,
        'logistic': logistic_pred,
        'random_forest': rf_pred,
        'meta_model': meta_pred
    }

    # --- Ensemble Prediction (Kết hợp tất cả các mô hình) ---
    # Ưu tiên dự đoán của meta-model nếu độ tin cậy cao
    if meta_conf > 0.65: # Ngưỡng tin cậy của meta-model
        final_prediction = meta_pred
        final_confidence = min(99.0, meta_conf * 100 * 1.1) # Tăng nhẹ độ tin cậy
        used_pattern_name = "Meta-model (Stacking)"
    else:
        # Nếu meta-model không đủ tin cậy, quay lại ensemble với trọng số động
        predictions_for_ensemble = {
            'pattern': {'pred': patt_pred, 'conf': patt_conf, 'weight': app.model_weights.get('pattern', app.default_model_weights['pattern'])},
            'markov': {'pred': markov_pred, 'conf': markov_conf, 'weight': app.model_weights.get('markov', app.default_model_weights['markov'])},
            'logistic': {'pred': logistic_pred, 'conf': logistic_conf, 'weight': app.model_weights.get('logistic', app.default_model_weights['logistic'])},
            'random_forest': {'pred': rf_pred, 'conf': rf_conf, 'weight': app.model_weights.get('random_forest', app.default_model_weights['random_forest'])},
        }
        
        tai_score, xiu_score = 0.0, 0.0
        for model_name, model_info in predictions_for_ensemble.items():
            score = model_info['conf'] * model_info['weight']
            if model_info['pred'] == 'Tài': tai_score += score
            else: xiu_score += score

        final_prediction = 'Tài' if tai_score > xiu_score else 'Xỉu'
        total_score = tai_score + xiu_score
        final_confidence = (max(tai_score, xiu_score) / total_score * 100) if total_score > 0 else 50.0
        used_pattern_name = detected_pattern_info['name'] if detected_pattern_info else "Ensemble Weighted"

    # Tăng độ tin cậy nếu pattern mạnh nhất trùng với dự đoán cuối cùng (nếu ensemble thường)
    if detected_pattern_info and detected_pattern_info['weight'] > 0.6 and patt_pred == final_prediction:
        final_confidence = min(98.0, final_confidence + patt_conf * 10)

    # Áp dụng logic meta cuối cùng
    final_prediction, final_confidence, meta_reason = apply_meta_logic(final_prediction, final_confidence, history_str)

    if meta_reason:
        used_pattern_name = meta_reason # Ưu tiên hiển thị lý do meta

    return final_prediction, used_pattern_name, final_confidence, individual_predictions

# --- Flask App Factory ---
def create_app():
    app = Flask(__name__)
    CORS(app)

    # --- Khởi tạo State ---
    app.lock = threading.Lock()
    app.MAX_HISTORY_LEN = 500 # Tăng độ dài lịch sử để huấn luyện tốt hơn
    
    app.history = deque(maxlen=app.MAX_HISTORY_LEN)
    app.session_ids = deque(maxlen=app.MAX_HISTORY_LEN)
    
    app.patterns = define_patterns()
    app.transition_matrix = [[0.5, 0.5], [0.5, 0.5]]
    app.transition_counts = [[0, 0], [0, 0]]
    app.logistic_weights = np.array([0.0] * 12) # Cập nhật cho 12 features
    app.logistic_bias = 0.0
    app.learning_rate = 0.01
    app.regularization = 0.001 # Giảm regularization một chút

    # Thêm trọng số cho các mô hình mới
    app.default_model_weights = {
        'pattern': 0.3,
        'markov': 0.1,
        'logistic': 0.2,
        'random_forest': 0.3,
        'meta_model': 0.1
    }
    app.model_weights = app.default_model_weights.copy()
    app.model_performance = {name: {"success": 0, "total": 0} for name in app.model_weights}

    app.last_prediction = None
    app.pattern_accuracy = defaultdict(lambda: {"success": 0, "total": 0})
    
    # Khởi tạo dữ liệu huấn luyện batch cho RF và Meta-model
    app.rf_meta_training_data = []
    app.TRAIN_BATCH_SIZE = 20 # Huấn luyện lại sau mỗi 20 mẫu (có thể điều chỉnh)

    # Load model state khi khởi động ứng dụng
    load_model_state(app) # Gọi hàm tải trạng thái ở đây

    app.API_URL = os.getenv("API_URL", "https://apib52.up.railway.app/api/taixiumd5")

    def fetch_data_from_api():
        while True:
            try:
                response = requests.get(app.API_URL, timeout=10)
                response.raise_for_status() # Ném lỗi cho mã trạng thái HTTP xấu (4xx hoặc 5xx)
                data = response.json()
                
                ket_qua = data.get("Ket_qua")
                phien = data.get("Phien")

                # Chuyển đổi "X\u1ec9u" thành "Xỉu" nếu cần (requests.json() thường tự xử lý)
                if ket_qua == "X\u1ec9u":
                    ket_qua = "Xỉu"
                elif ket_qua == "T\u00e0i":
                    ket_qua = "Tài"
                
                if ket_qua not in ["Tài", "Xỉu"] or phien is None:
                    logging.warning(f"Invalid data received from API: {data}")
                    time.sleep(5)
                    continue
                
                with app.lock:
                    if not app.session_ids or phien > app.session_ids[-1]:
                        # Lưu lại lịch sử trước khi thêm kết quả mới để huấn luyện
                        prev_history_str_for_training = _get_history_strings(list(app.history))
                        
                        app.session_ids.append(phien)
                        app.history.append({'ket_qua': ket_qua, 'phien': phien})
                        logging.info(f"New result from API for session {phien}: {ket_qua}")

                        # --- Bước học Online (Online Learning) ---
                        # Chỉ học khi có kết quả mới và có dự đoán cho phiên này
                        if app.last_prediction and app.last_prediction['session'] == phien:
                            actual_result = ket_qua
                            
                            # Huấn luyện cho Logistic Regression
                            train_logistic_regression(app, app.last_prediction['features'], actual_result)
                            # Huấn luyện cho Markov Chain
                            if len(prev_history_str_for_training) > 0:
                                update_transition_matrix(app, prev_history_str_for_training[-1], actual_result)
                            # Huấn luyện cho Pattern
                            update_pattern_accuracy(app, app.last_prediction['pattern'], app.last_prediction['prediction'], actual_result)
                            
                            # Cập nhật hiệu suất của từng mô hình con (bao gồm RF và Meta-model)
                            for model_name, model_pred_val in app.last_prediction['individual_predictions'].items():
                                app.model_performance[model_name]['total'] += 1
                                if model_pred_val == actual_result:
                                    app.model_performance[model_name]['success'] += 1
                            
                            # Thu thập dữ liệu để huấn luyện RF và Meta-model sau này
                            app.rf_meta_training_data.append({
                                'features': app.last_prediction['features'],
                                'base_predictions_vals': [
                                    1 if app.last_prediction['individual_predictions']['pattern'] == 'Tài' else 0,
                                    1 if app.last_prediction['individual_predictions']['markov'] == 'Tài' else 0,
                                    1 if app.last_prediction['individual_predictions']['logistic'] == 'Tài' else 0,
                                    1 if app.last_prediction['individual_predictions'].get('random_forest', 'Xỉu') == 'Tài' else 0, # RF có thể chưa có
                                ],
                                'actual_result_val': 1 if actual_result == 'Tài' else 0
                            })

                            if len(app.rf_meta_training_data) >= app.TRAIN_BATCH_SIZE:
                                logging.info(f"Retraining RF and Meta-model with {len(app.rf_meta_training_data)} samples.")
                                X_rf_train = np.array([d['features'] for d in app.rf_meta_training_data])
                                y_rf_train = np.array([d['actual_result_val'] for d in app.rf_meta_training_data])
                                
                                if len(np.unique(y_rf_train)) > 1: # Cần ít nhất 2 lớp để huấn luyện
                                    try:
                                        app.random_forest_model.fit(X_rf_train, y_rf_train)
                                        logging.info("Random Forest retrained.")
                                    except Exception as e:
                                        logging.error(f"Error retraining Random Forest: {e}")

                                    X_meta_train_batch = np.array([d['base_predictions_vals'] for d in app.rf_meta_training_data])
                                    if len(np.unique(y_rf_train)) > 1: # Cần ít nhất 2 lớp
                                        try:
                                            app.meta_model.fit(X_meta_train_batch, y_rf_train)
                                            logging.info("Meta-model retrained.")
                                        except Exception as e:
                                            logging.error(f"Error retraining Meta-model: {e}")
                                    else:
                                        logging.warning("Not enough classes in training batch for Meta-model.")
                                else:
                                    logging.warning("Not enough classes in training batch for Random Forest.")

                                app.rf_meta_training_data.clear() # Xóa dữ liệu đã dùng
                            
                            # Cập nhật lại trọng số của ensemble model
                            update_model_weights(app)
                            
                            # Lưu trạng thái mô hình sau khi học
                            save_model_state(app)

                            logging.info(f"Learned from session {phien}. Prediction was {app.last_prediction['prediction']}, actual was {actual_result}. Pattern: {app.last_prediction['pattern']}")
                            
                        app.last_prediction = None # Reset last_prediction để chờ phiên tiếp theo
                    else:
                        logging.debug(f"Session {phien} already processed or older. Current latest: {app.session_ids[-1]}")
                        
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching data from API: {e}")
            except json.JSONDecodeError:
                logging.error(f"Failed to decode JSON from API response: {response.text}")
            except Exception as e:
                logging.error(f"Unhandled error in API data fetching thread: {e}")
            
            time.sleep(10) # Đợi 10 giây trước khi gọi API lần tiếp theo

    # --- API Endpoints ---
    @app.route("/api/taixiu", methods=["GET"])
    def get_taixiu_prediction():
        with app.lock:
            if len(app.history) < 15: # Yêu cầu nhiều lịch sử hơn cho dự đoán nâng cao
                return jsonify({"error": "Chưa có đủ dữ liệu lịch sử để dự đoán. Cần ít nhất 15 phiên."}), 500
            
            history_copy = list(app.history)
            current_session_number = history_copy[-1]['phien'] if history_copy else 0
        
        # --- Bước Dự đoán (Prediction) ---
        history_str_for_prediction = _get_history_strings(history_copy)
        prediction_str, pattern_str, confidence, individual_preds = predict_advanced(app, history_str_for_prediction)
        
        # Lưu lại thông tin dự đoán để học ở lần tiếp theo
        with app.lock:
            app.last_prediction = {
                'session': current_session_number + 1, # Dự đoán cho phiên tiếp theo
                'prediction': prediction_str,
                'pattern': pattern_str,
                'features': get_advanced_features(history_str_for_prediction).tolist(), # Lưu features dạng list để dễ dàng JSON serialize
                'individual_predictions': individual_preds,
            }
            current_result = history_copy[-1]['ket_qua'] if history_copy else "N/A"
        
        # Tinh chỉnh hiển thị
        prediction_display = prediction_str
        final_confidence_display = round(confidence, 1)
        
        if confidence < 70.0 and "Bẻ cầu" not in pattern_str and "Meta-model" not in pattern_str: 
             prediction_display = f"Đang phân tích"


        return jsonify({
            "current_session": current_session_number,
            "current_result": current_result,
            "next_session": current_session_number + 1,
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

    # Khởi động luồng lấy dữ liệu API
    api_fetch_thread = threading.Thread(target=fetch_data_from_api, daemon=True)
    api_fetch_thread.start()
    return app

# --- Thực thi chính ---
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"Flask app ready. Serving on http://0.0.0.0:{port}")
    from waitress import serve
    serve(app, host="0.0.0.0", port=port, threads=8)

