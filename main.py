import sqlite3
from collections import deque
import os
import requests # Thêm thư viện này để gọi API bên ngoài
from flask import Flask, request, jsonify

# --- Cấu hình ---
DATABASE_NAME = 'game_data.db'
RECENT_SESSIONS_COUNT = 100  # Số phiên gần nhất để phân tích (có thể thử 50, 100, 200)
PATTERN_LENGTHS_TO_ANALYZE = [3, 5, 8, 10, 20] # Các độ dài mẫu con để phân tích chi tiết
CONSECUTIVE_LOSSES_TO_REVERSE = 3 # Số lần trật liên tiếp để kích hoạt đảo ngược
MIN_CONFIDENCE_FOR_REVERSE_TRIGGER = 0.65 # Ngưỡng tự tin để không đảo ngược nếu dự đoán dựa trên xác suất mạnh (0.0 - 1.0)
EXTERNAL_API_URL = "https://apib52.up.railway.app/api/taixiumd5" # URL của API bên ngoài

app = Flask(__name__)

# --- Hàm tương tác với cơ sở dữ liệu ---
def get_db_connection():
    """Tạo và trả về kết nối đến cơ sở dữ liệu."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row # Cho phép truy cập cột bằng tên
    return conn

def init_db():
    """Khởi tạo cơ sở dữ liệu và tạo bảng nếu chưa tồn tại."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            outcome TEXT NOT NULL, -- 't' for Tài, 'x' for Xỉu
            predicted_outcome TEXT,
            is_correct INTEGER, -- 1 for correct, 0 for incorrect
            prediction_time DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database '{DATABASE_NAME}' initialized successfully.")

def add_session_result(session_id: int, outcome: str, predicted_outcome: str):
    """Thêm kết quả phiên vào cơ sở dữ liệu và cập nhật độ chính xác."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    is_correct = 1 if (outcome == 't' and predicted_outcome == 'Tài') or \
                        (outcome == 'x' and predicted_outcome == 'Xỉu') else 0
    
    # Kiểm tra xem session_id đã tồn tại chưa để tránh trùng lặp khi fetch từ API
    cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
    if cursor.fetchone():
        # print(f"Session ID {session_id} already exists. Skipping insertion.")
        conn.close()
        return False # Trả về False nếu đã tồn tại
        
    cursor.execute(
        "INSERT INTO sessions (id, outcome, predicted_outcome, is_correct) VALUES (?, ?, ?, ?)",
        (session_id, outcome, predicted_outcome, is_correct)
    )
    conn.commit()
    conn.close()
    return True # Trả về True nếu thêm thành công

def get_recent_outcomes_from_db(count: int) -> deque:
    """Lấy N kết quả phiên gần nhất từ cơ sở dữ liệu nội bộ."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT outcome FROM sessions ORDER BY id DESC LIMIT {count}")
    rows = cursor.fetchall()
    conn.close()
    
    return deque([row['outcome'] for row in reversed(rows)])

def get_last_n_predictions_and_outcomes_from_db(n: int):
    """Lấy N dự đoán và kết quả thực tế gần nhất từ DB để kiểm tra chuỗi trật."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT predicted_outcome, outcome FROM sessions WHERE predicted_outcome IS NOT NULL ORDER BY id DESC LIMIT {n}")
    rows = cursor.fetchall()
    conn.close()
    
    predictions = [row['predicted_outcome'] for row in reversed(rows)]
    outcomes = ["Tài" if row['outcome'] == 't' else "Xỉu" for row in reversed(rows)]
    
    return predictions, outcomes

def fetch_and_store_historical_data(limit=RECENT_SESSIONS_COUNT * 2): # Fetch nhiều hơn để đảm bảo có đủ 100 phiên
    """
    Fetch dữ liệu lịch sử từ API bên ngoài và lưu vào DB nội bộ.
    Chỉ lưu các phiên chưa có trong DB.
    """
    print(f"Fetching {limit} latest sessions from external API...")
    try:
        response = requests.get(EXTERNAL_API_URL, params={"limit": limit})
        response.raise_for_status() # Gây lỗi cho các mã trạng thái HTTP xấu
        data = response.json()
        
        inserted_count = 0
        # Dữ liệu từ API thường là mới nhất trước, nên lặp ngược lại để thêm vào DB theo thứ tự ID tăng dần
        for entry in reversed(data): 
            session_id = entry.get("Phien")
            ket_qua = entry.get("Ket_qua")
            # Chuyển đổi "Tài" thành 't', "Xỉu" thành 'x'
            outcome_char = 't' if ket_qua == 'Tài' else 'x' if ket_qua == 'Xỉu' else None
            
            if session_id and outcome_char:
                # predicted_outcome ở đây là None vì đây là dữ liệu lịch sử từ API, không phải dự đoán của hệ thống này
                if add_session_result(session_id, outcome_char, None): 
                    inserted_count += 1
            
        print(f"Successfully fetched and inserted {inserted_count} new sessions from external API.")
        if inserted_count == 0 and len(data) > 0:
            print("No new sessions inserted. Database is up-to-date or older sessions already exist.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from external API: {e}")
    except ValueError as e:
        print(f"Error parsing JSON from external API: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during data fetching: {e}")


# --- Hàm phân tích chính ---
def analyze_recent_sessions(recent_outcomes: deque):
    """
    Phân tích chi tiết N phiên gần nhất (RECENT_SESSIONS_COUNT) để tìm cầu và xu hướng.
    Trả về một dictionary chứa các kết quả phân tích.
    """
    analysis_results = {}
    
    # Kiểm tra xem có đủ dữ liệu để phân tích số phiên mong muốn không
    if len(recent_outcomes) < RECENT_SESSIONS_COUNT:
        outcomes_to_analyze = list(recent_outcomes)
        analysis_results['warning'] = f"Chưa đủ {RECENT_SESSIONS_COUNT} phiên để phân tích chi tiết. Chỉ có {len(recent_outcomes)} phiên."
    else:
        outcomes_to_analyze = list(recent_outcomes)[-RECENT_SESSIONS_COUNT:]

    total_analyzed_count = len(outcomes_to_analyze)
    if total_analyzed_count == 0:
        return {'error': 'Không có dữ liệu phiên nào để phân tích.'}

    # 1. Tỷ lệ Tài/Xỉu tổng thể trong N phiên được phân tích
    t_count_total = outcomes_to_analyze.count('t')
    x_count_total = outcomes_to_analyze.count('x')
    analysis_results['overall_analyzed_percent'] = {
        't_percent': round((t_count_total / total_analyzed_count) * 100, 2),
        'x_percent': round((x_count_total / total_analyzed_count) * 100, 2)
    }

    # 2. Phân tích các mẫu độ dài khác nhau
    analysis_results['multi_length_patterns'] = {}
    for length in PATTERN_LENGTHS_TO_ANALYZE:
        if total_analyzed_count >= length:
            pattern_slice = outcomes_to_analyze[-length:]
            t_count = pattern_slice.count('t')
            x_count = pattern_slice.count('x')
            analysis_results['multi_length_patterns'][f'last_{length}_sessions'] = {
                'pattern': "".join(pattern_slice),
                't_percent': round((t_count / length) * 100, 2),
                'x_percent': round((x_count / length) * 100, 2)
            }
            
    # 3. Phân tích chuỗi (streaks) hiện tại
    current_streak = {'outcome': None, 'length': 0}
    if outcomes_to_analyze:
        last_outcome_char = outcomes_to_analyze[-1]
        for i in range(len(outcomes_to_analyze) - 1, -1, -1):
            if outcomes_to_analyze[i] == last_outcome_char:
                current_streak['length'] += 1
            else:
                break
        current_streak['outcome'] = 'Tài' if last_outcome_char == 't' else 'Xỉu'
    analysis_results['current_streak'] = current_streak

    # 4. Phân tích tần suất mẫu con và xác suất điều kiện
    analysis_results['subpattern_conditional_probabilities'] = {}
    
    subpattern_lengths_for_prediction = sorted([pl for pl in PATTERN_LENGTHS_TO_ANALYZE if total_analyzed_count >= pl], reverse=True)
    
    for sub_len in subpattern_lengths_for_prediction:
        current_sub_pattern = "".join(outcomes_to_analyze[-sub_len:])
        
        next_t_count = 0
        next_x_count = 0
        total_sub_pattern_occurrences = 0

        for i in range(total_analyzed_count - sub_len):
            if "".join(outcomes_to_analyze[i : i + sub_len]) == current_sub_pattern:
                total_sub_pattern_occurrences += 1
                if outcomes_to_analyze[i + sub_len] == 't':
                    next_t_count += 1
                else:
                    next_x_count += 1
        
        prob_next_t = round((next_t_count / total_sub_pattern_occurrences) * 100, 2) if total_sub_pattern_occurrences > 0 else 0
        prob_next_x = round((next_x_count / total_sub_pattern_occurrences) * 100, 2) if total_sub_pattern_occurrences > 0 else 0
        
        analysis_results['subpattern_conditional_probabilities'][current_sub_pattern] = {
            'total_occurrences_in_analyzed_range': total_sub_pattern_occurrences,
            'prob_next_t': prob_next_t,
            'prob_next_x': prob_next_x
        }
    
    return analysis_results

# --- API Endpoints ---

@app.route('/api/predict', methods=['GET'])
def get_prediction():
    """
    Endpoint để lấy dự đoán tiếp theo.
    Trước khi dự đoán, cố gắng fetch dữ liệu lịch sử mới nhất từ API bên ngoài.
    """
    current_session_id = request.args.get('current_session_id', type=int)
    if not current_session_id:
        return jsonify({"error": "current_session_id là bắt buộc."}), 400

    next_session_id_predicted = current_session_id + 1

    # --- Fetch và cập nhật dữ liệu lịch sử từ API bên ngoài ---
    fetch_and_store_historical_data() 
    
    # Lấy N kết quả gần nhất từ DB nội bộ
    recent_outcomes_deque = get_recent_outcomes_from_db(RECENT_SESSIONS_COUNT)
    
    if not recent_outcomes_deque:
        return jsonify({
            "predicted_outcome": "Không thể dự đoán",
            "prediction_reason": "Chưa có đủ dữ liệu lịch sử để phân tích.",
            "next_session_id_predicted": next_session_id_predicted,
            "prediction_confidence_score": 0.0,
            "analysis_details": {}
        }), 200

    detailed_analysis = analyze_recent_sessions(recent_outcomes_deque)

    predicted_outcome = None
    prediction_confidence = 0.0
    prediction_reason_parts = []
    
    # --- Logic dự đoán chính ---

    # 1. Ưu tiên dựa trên xác suất điều kiện của mẫu con gần nhất (mẫu dài nhất có dữ liệu)
    best_cond_pattern = None
    for sub_len in sorted(PATTERN_LENGTHS_TO_ANALYZE, reverse=True): 
        if len(recent_outcomes_deque) >= sub_len: 
            current_sub_pattern_str = "".join(list(recent_outcomes_deque)[-sub_len:])
            cond_data = detailed_analysis['subpattern_conditional_probabilities'].get(current_sub_pattern_str)
            if cond_data and cond_data['total_occurrences_in_analyzed_range'] > 0:
                best_cond_pattern = current_sub_pattern_str
                break 

    if best_cond_pattern:
        cond_probs = detailed_analysis['subpattern_conditional_probabilities'][best_cond_pattern]
        if cond_probs['prob_next_t'] > cond_probs['prob_next_x']:
            predicted_outcome = "Tài"
            prediction_confidence = cond_probs['prob_next_t'] / 100
            prediction_reason_parts.append(f"Theo XS điều kiện mẫu '{best_cond_pattern}' ({cond_probs['prob_next_t']}%)")
        else:
            predicted_outcome = "Xỉu"
            prediction_confidence = cond_probs['prob_next_x'] / 100
            prediction_reason_parts.append(f"Theo XS điều kiện mẫu '{best_cond_pattern}' ({cond_probs['prob_next_x']}%)")
    
    # 2. Nếu xác suất điều kiện không đủ mạnh hoặc không có dữ liệu, dùng tỷ lệ chung
    if predicted_outcome is None or prediction_confidence < MIN_CONFIDENCE_FOR_REVERSE_TRIGGER:
        overall_analyzed_percent_t = detailed_analysis['overall_analyzed_percent']['t_percent']
        overall_analyzed_percent_x = detailed_analysis['overall_analyzed_percent']['x_percent']
        
        if overall_analyzed_percent_t >= overall_analyzed_percent_x:
            if predicted_outcome is None: 
                predicted_outcome = "Tài"
                prediction_confidence = overall_analyzed_percent_t / 100
            if f"Theo TL Tài trong {RECENT_SESSIONS_COUNT} phiên ({overall_analyzed_percent_t}%)" not in prediction_reason_parts:
                prediction_reason_parts.append(f"Theo TL Tài trong {RECENT_SESSIONS_COUNT} phiên ({overall_analyzed_percent_t}%)")
        else:
            if predicted_outcome is None:
                predicted_outcome = "Xỉu"
                prediction_confidence = overall_analyzed_percent_x / 100
            if f"Theo TL Xỉu trong {RECENT_SESSIONS_COUNT} phiên ({overall_analyzed_percent_x}%)" not in prediction_reason_parts:
                prediction_reason_parts.append(f"Theo TL Xỉu trong {RECENT_SESSIONS_COUNT} phiên ({overall_analyzed_percent_x}%)")
            
    # Đảm bảo có một dự đoán mặc định nếu mọi thứ đều không chắc chắn
    if predicted_outcome is None:
        predicted_outcome = "Tài" 
        prediction_confidence = 0.5
        prediction_reason_parts.append("Không đủ cơ sở dữ liệu/phân tích không rõ ràng, dự đoán mặc định Tài.")


    # 3. Áp dụng quy tắc "Đảo ngược" nếu đang trật liên tiếp và độ tin cậy không quá cao
    last_predictions, last_outcomes = get_last_n_predictions_and_outcomes_from_db(CONSECUTIVE_LOSSES_TO_REVERSE)
    
    consecutive_failures = 0
    if len(last_predictions) == CONSECUTIVE_LOSSES_TO_REVERSE: 
        temp_failures = 0
        for i in range(len(last_predictions) -1, -1, -1): 
            if last_predictions[i] != last_outcomes[i]:
                temp_failures += 1
            else:
                break 
        consecutive_failures = temp_failures
    
    if consecutive_failures >= CONSECUTIVE_LOSSES_TO_REVERSE and prediction_confidence < MIN_CONFIDENCE_FOR_REVERSE_TRIGGER:
        original_prediction = predicted_outcome
        predicted_outcome = "Xỉu" if original_prediction == "Tài" else "Tài"
        prediction_reason_parts.append(f"Trật {consecutive_failures} lần liên tiếp, Auto ĐẢO NGƯỢC từ {original_prediction}.")
        prediction_confidence = max(0.5, 1 - prediction_confidence) 

    # Lấy mẫu hiện tại cho response (thường là 8 phiên gần nhất như mẫu bạn cung cấp)
    current_pattern_str_for_display = "".join(list(recent_outcomes_deque)[-8:]) if len(recent_outcomes_deque) >= 8 else "".join(list(recent_outcomes_deque))
    current_pattern_details_for_display = {}
    if current_pattern_str_for_display:
        t_count_display = current_pattern_str_for_display.count('t')
        x_count_display = current_pattern_str_for_display.count('x')
        current_pattern_details_for_display = {
            't_count_in_pattern': t_count_display,
            'x_count_in_pattern': x_count_display,
            't_percent_in_pattern': round((t_count_display / len(current_pattern_str_for_display)) * 100, 2),
            'x_percent_in_pattern': round((x_count_display / len(current_pattern_str_for_display)) * 100, 2)
        }
    
    response_data = {
        "Phien_du_doan": next_session_id_predicted,
        "pattern_length": len(current_pattern_str_for_display), # Độ dài pattern hiện tại
        "pattern": current_pattern_str_for_display,
        "pattern_details": current_pattern_details_for_display, # Chi tiết T/X trong pattern hiện tại
        "Tong_phan_tich_tu_phien": len(recent_outcomes_deque), # Tổng số phiên thực tế được phân tích
        "Ty_le_phan_tich": detailed_analysis.get('overall_analyzed_percent', {}),
        "Chuoi_hien_tai": detailed_analysis.get('current_streak', {}),
        "Du_doan": predicted_outcome,
        "Do_tin_cay": round(prediction_confidence, 4), 
        "Ly_do": " | ".join(prediction_reason_parts),
        "Chi_tiet_phan_tich_nang_cao": { # Cung cấp thêm chi tiết phân tích
            "Xac_suat_dieu_kien_mau": detailed_analysis.get('subpattern_conditional_probabilities', {}),
            "Ty_le_mau_da_phan_tich": detailed_analysis.get('multi_length_patterns', {})
        }
    }
    
    return jsonify(response_data)

@app.route('/api/record_outcome', methods=['POST'])
def record_outcome():
    """
    Endpoint để ghi lại kết quả thực tế của một phiên (ví dụ, sau khi hệ thống của bạn đã dự đoán và có kết quả).
    Cần thiết để kiểm tra hiệu suất và áp dụng quy tắc đảo ngược.
    """
    data = request.json
    session_id = data.get('session_id')
    outcome = data.get('outcome') # 't' hoặc 'x'
    predicted_outcome = data.get('predicted_outcome') # Dự đoán mà hệ thống này đã đưa ra cho phiên đó

    if not all([session_id, outcome, predicted_outcome]):
        return jsonify({"error": "session_id, outcome, và predicted_outcome là bắt buộc."}), 400

    if outcome not in ['t', 'x']:
        return jsonify({"error": "Kết quả phải là 't' (Tài) hoặc 'x' (Xỉu)."}), 400
    
    if predicted_outcome not in ['Tài', 'Xỉu']:
        return jsonify({"error": "Dự đoán phải là 'Tài' hoặc 'Xỉu'."}), 400

    try:
        # Sử dụng hàm add_session_result để thêm vào DB
        success = add_session_result(session_id, outcome, predicted_outcome)
        if success:
            return jsonify({"message": f"Kết quả cho phiên {session_id} đã được ghi lại thành công."}), 200
        else:
            return jsonify({"message": f"Phiên {session_id} đã tồn tại trong DB, không ghi lại."}), 200 # Hoặc 409 Conflict
    except Exception as e:
        return jsonify({"error": f"Lỗi khi ghi kết quả: {str(e)}"}), 500

if __name__ == '__main__':
    init_db() # Khởi tạo database khi chạy file main.py

    # IMPORTANT: Fetch data on startup (can be scheduled later)
    # This ensures your local DB is populated when the app starts on Render
    fetch_and_store_historical_data(limit=RECENT_SESSIONS_COUNT * 5) # Fetch nhiều hơn để đảm bảo đủ dữ liệu ban đầu

    # Chạy ứng dụng trên cổng 8080 (hoặc PORT được Render chỉ định)
    # Render sẽ cung cấp biến môi trường PORT. Nếu không, mặc định 8080.
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
