import collections
import copy
import logging
import os
import random
import requests
from flask import Flask, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler # Import thư viện cho bộ lập lịch

# --- Cấu hình Logging ---
# Thiết lập logging để hiển thị thông báo ra console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Lớp cấu hình ứng dụng ---
class AppConfig:
    """
    Quản lý các cấu hình toàn ứng dụng, giúp dễ dàng điều chỉnh.
    """
    # Cập nhật URL API bên ngoài để lấy dữ liệu từ link mới
    EXTERNAL_API_URL: str = os.getenv("EXTERNAL_API_URL", "https://apib52.up.railway.app/api/taixiumd5")
    HISTORY_MAX_LEN: int = 100  # Số phiên tối đa lưu trong lịch sử
    DEFAULT_PATTERN_LENGTH: int = 8

    # Ngưỡng chiến lược dự đoán
    MIN_STREAK_FOR_PREDICTION: int = 3
    BREAK_STREAK_THRESHOLD: int = 5
    CONDITIONAL_PROB_LOOKBACK: int = 3
    PROB_THRESHOLD_STRONG: float = 0.60
    REVERSE_THRESHOLD: int = 3 # Số lần thua liên tiếp để kích hoạt đảo ngược

    # CẤU HÌNH TẦN SUẤT KIỂM TRA TỰ ĐỘNG
    # Tần suất (giây) API của bạn sẽ gọi API ngoài để cập nhật dữ liệu
    # Đã thay đổi thành 5 giây
    CHECK_INTERVAL_SECONDS: int = 5


# --- Lớp trạng thái ứng dụng ---
class AppState:
    """
    Quản lý trạng thái toàn cục của ứng dụng, bao gồm lịch sử và thông tin dự đoán.
    """
    def __init__(self):
        self.history_results: collections.deque = collections.deque(maxlen=AppConfig.HISTORY_MAX_LEN)
        self.last_prediction_info: dict = {
            "predicted_expect": None,
            "predicted_result": None,
            "consecutive_losses": 0,
            "last_actual_result": None,
            # MỚI: Thống kê dự đoán
            "total_predictions": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0
        }
        self.initial_api_data_template: dict = {
            "Phien_moi": None,
            "pattern_length": AppConfig.DEFAULT_PATTERN_LENGTH,
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
            "admin_info": "@heheviptool"
        }

    def reset_prediction_info(self):
        """Đặt lại số lần thua liên tiếp và theo dõi dự đoán liên quan, không phải tổng số thống kê."""
        self.last_prediction_info["predicted_expect"] = None
        self.last_prediction_info["predicted_result"] = None
        self.last_prediction_info["consecutive_losses"] = 0
        self.last_prediction_info["last_actual_result"] = None


    def update_last_prediction_info(self, predicted_expect: str, predicted_result: str):
        """Cập nhật chi tiết dự đoán cuối cùng để theo dõi."""
        self.last_prediction_info["predicted_expect"] = predicted_expect
        self.last_prediction_info["predicted_result"] = predicted_result

    def update_prediction_stats(self, processed_session_expect: str, actual_result_char: str):
        """
        Cập nhật thống kê dự đoán (tổng, đúng, sai) và số lần thua liên tiếp.
        Hàm này được gọi *sau khi* kết quả thực tế của một phiên mới được biết
        và so sánh nó với dự đoán *trước đó*.
        """
        predicted_expect = self.last_prediction_info["predicted_expect"]
        predicted_result_str = self.last_prediction_info["predicted_result"]

        # Chỉ đánh giá nếu có dự đoán được đưa ra cho phiên cụ thể này
        if predicted_expect is not None and predicted_expect == processed_session_expect and predicted_result_str is not None:
            
            predicted_res_char = "t" if predicted_result_str == "Tài" else "x"
            
            self.last_prediction_info["total_predictions"] += 1
            logging.debug(f"Đang đánh giá dự đoán cho phiên {processed_session_expect}. Dự đoán: {predicted_result_str} ({predicted_res_char}), Thực tế: {actual_result_char}")

            if predicted_res_char != actual_result_char:
                self.last_prediction_info["consecutive_losses"] += 1
                self.last_prediction_info["incorrect_predictions"] += 1
                logging.info(f"Dự đoán '{predicted_result_str}' cho phiên Expect {processed_session_expect} TRẬT. Số lần thua liên tiếp: {self.last_prediction_info['consecutive_losses']}")
            else:
                self.last_prediction_info["consecutive_losses"] = 0
                self.last_prediction_info["correct_predictions"] += 1
                logging.info(f"Dự đoán '{predicted_result_str}' cho phiên Expect {processed_session_expect} ĐÚNG. Đặt lại số lần thua.")
        else:
            # Xử lý các trường hợp không có dự đoán nào được đưa ra cho phiên hiện tại,
            # hoặc đây là lần chạy đầu tiên, hoặc ứng dụng đã khởi động lại. Đặt lại số lần thua nếu không có dự đoán phù hợp.
            self.last_prediction_info["consecutive_losses"] = 0
            logging.debug(f"Không có dự đoán phù hợp trước đó cho phiên {processed_session_expect} để đánh giá. Đặt lại số lần thua.")
        
        self.last_prediction_info["last_actual_result"] = actual_result_char


# --- Hàm trợ giúp ---
def calculate_tai_xiu(open_code_str: str) -> tuple[str, int]:
    """
    Tính tổng giá trị xúc xắc và xác định "Tài" (Lớn) hoặc "Xỉu" (Nhỏ).
    Trả về một tuple (loại kết quả: str, tổng: int).
    """
    try:
        dice_values = [int(x.strip()) for x in open_code_str.split(',')]
        total_sum = sum(dice_values)

        if 4 <= total_sum <= 10:
            return "Xỉu", total_sum
        elif 11 <= total_sum <= 17:
            return "Tài", total_sum
        elif total_sum == 3: # Ba con 1 - được coi là Xỉu
            return "Xỉu", total_sum
        elif total_sum == 18: # Ba con 6 - được coi là Tài
            return "Tài", total_sum
        else:
            logging.warning(f"Tổng không mong muốn cho OpenCode '{open_code_str}': {total_sum}")
            return "Không xác định", total_sum
    except (ValueError, TypeError) as e:
        logging.error(f"Lỗi khi tính Tài/Xỉu từ OpenCode '{open_code_str}': {e}")
        return "Lỗi", 0

def get_next_expect_code(current_expect_code: str) -> str | None:
    """
    Tính toán mã phiên tiếp theo bằng cách tăng 4 chữ số cuối.
    Giả định định dạng mã phiên là 'YYYYMMDDXXXX' hoặc chỉ là số.
    """
    # Xử lý trường hợp mã phiên chỉ là số (ví dụ: '1450828')
    if current_expect_code.isdigit():
        try:
            next_suffix_int = int(current_expect_code) + 1
            return str(next_suffix_int)
        except ValueError:
            logging.error(f"Không thể chuyển đổi hậu tố '{current_expect_code}' thành số nguyên để tăng.")
            return None
    
    # Giữ lại logic cũ cho định dạng 'YYYYMMDDXXXX' nếu cần
    if len(current_expect_code) < 4 or not current_expect_code[-4:].isdigit():
        logging.warning(f"Mã phiên '{current_expect_code}' không khớp định dạng mong muốn để tăng. Không thể tính toán mã phiên tiếp theo.")
        return None

    prefix = current_expect_code[:-4]
    suffix_str = current_expect_code[-4:]
    
    try:
        suffix_int = int(suffix_str)
        next_suffix_int = suffix_int + 1
        next_suffix_str = str(next_suffix_int).zfill(len(suffix_str))
        return prefix + next_suffix_str
    except ValueError:
        logging.error(f"Không thể chuyển đổi hậu tố '{suffix_str}' thành số nguyên để tăng.")
        return None

# --- Bộ xử lý dữ liệu ---
class SessionDataProcessor:
    """
    Xử lý dữ liệu phiên mới từ API bên ngoài và cập nhật AppState.
    """
    def __init__(self, app_state: AppState):
        self.app_state = app_state

    def process_new_session(self, new_session_data: dict):
        """
        Cập nhật lịch sử và trạng thái toàn cục dựa trên dữ liệu phiên mới.
        Đã cập nhật để phù hợp với định dạng dữ liệu mới.
        """
        # Lấy dữ liệu từ định dạng JSON mới
        current_id = new_session_data.get('id', 'unknown_id') # Sử dụng 'id' từ JSON
        current_expect_code = str(new_session_data.get('Phien')) # Sử dụng 'Phien' làm mã phiên
        
        # Tạo OpenCode từ các giá trị xúc xắc
        xuc_xac_1 = new_session_data.get('Xuc_xac_1')
        xuc_xac_2 = new_session_data.get('Xuc_xac_2')
        xuc_xac_3 = new_session_data.get('Xuc_xac_3')
        # Đảm bảo tất cả các giá trị xúc xắc đều tồn tại trước khi tạo chuỗi
        if all(v is not None for v in [xuc_xac_1, xuc_xac_2, xuc_xac_3]):
            current_open_code = f"{xuc_xac_1},{xuc_xac_2},{xuc_xac_3}"
        else:
            current_open_code = "0,0,0" # Giá trị mặc định nếu thiếu dữ liệu xúc xắc
            logging.warning(f"Thiếu dữ liệu xúc xắc cho phiên {current_expect_code}. OpenCode mặc định: {current_open_code}")

        # Lấy kết quả trực tiếp từ trường 'Ket_qua' của API mới
        actual_result_type_from_api = new_session_data.get('Ket_qua', 'Không xác định')
        actual_result_char = "t" if "Tài" in actual_result_type_from_api else "x"

        # Kiểm tra xem phiên này đã có trong lịch sử để tránh trùng lặp
        # Sử dụng current_expect_code (tức là 'Phien') để kiểm tra tính duy nhất
        if not any(entry['Expect'] == current_expect_code for entry in self.app_state.history_results):
            self.app_state.history_results.append({
                "ID": current_id,
                "Expect": current_expect_code,
                "OpenCode": current_open_code,
                "Result": actual_result_char
            })
            logging.info(f"Đã thêm phiên mới vào lịch sử: ID {current_id}, Expect {current_expect_code} - Kết quả: {actual_result_type_from_api}")
            
            # Sau khi thêm một phiên mới, cập nhật số lần thua liên tiếp VÀ số liệu thống kê dự đoán
            self.app_state.update_prediction_stats(current_expect_code, actual_result_char)
        else:
            logging.debug(f"Phiên Expect {current_expect_code} đã có trong lịch sử. Bỏ qua việc thêm vào.")


        # Cập nhật các trường cốt lõi trong template
        self.app_state.initial_api_data_template["Phien_moi"] = current_expect_code
        
        next_expect_code = get_next_expect_code(current_expect_code)
        self.app_state.initial_api_data_template["phien_du_doan"] = next_expect_code if next_expect_code else "Không xác định"

        # Cập nhật pattern và phần trăm pattern
        current_pattern_chars = "".join([entry['Result'] for entry in self.app_state.history_results])
        # Đảm bảo độ dài pattern không vượt quá lịch sử có sẵn
        effective_pattern_length = min(AppConfig.DEFAULT_PATTERN_LENGTH, len(current_pattern_chars))
        self.app_state.initial_api_data_template['pattern'] = current_pattern_chars[-effective_pattern_length:]
        self.app_state.initial_api_data_template['pattern_length'] = effective_pattern_length
        
        tai_count = self.app_state.initial_api_data_template['pattern'].count('t')
        xiu_count = self.app_state.initial_api_data_template['pattern'].count('x')
        
        self.app_state.initial_api_data_template['pattern_tai'] = tai_count
        self.app_state.initial_api_data_template['pattern_xiu'] = xiu_count

        total_pattern_chars = len(self.app_state.initial_api_data_template['pattern'])
        if total_pattern_chars > 0:
            self.app_state.initial_api_data_template['pattern_percent_tai'] = round((tai_count / total_pattern_chars) * 100, 2)
            self.app_state.initial_api_data_template['pattern_percent_xiu'] = round((xiu_count / total_pattern_chars) * 100, 2)
        else:
            self.app_state.initial_api_data_template['pattern_percent_tai'] = 0
            self.app_state.initial_api_data_template['pattern_percent_xiu'] = 0

        # Cập nhật 'matches' (giả sử nó đề cập đến kết quả mới nhất)
        if self.app_state.history_results:
            self.app_state.initial_api_data_template['matches'] = [self.app_state.history_results[-1]['Result']]
        else:
            self.app_state.initial_api_data_template['matches'] = []

        # Giả sử phan_tram_tai/xiu và tong_tai/xiu được lấy từ pattern_percent
        self.app_state.initial_api_data_template['phan_tram_tai'] = self.app_state.initial_api_data_template['pattern_percent_tai']
        self.app_state.initial_api_data_template['phan_tram_xiu'] = self.app_state.initial_api_data_template['pattern_percent_xiu']
        
        # Tùy ý điều chỉnh 'tong_tai' và 'tong_xiu' cho đầu ra JSON
        self.app_state.initial_api_data_template['tong_tai'] = round(self.app_state.initial_api_data_template['phan_tram_tai'] * 10, 2)
        self.app_state.initial_api_data_template['tong_xiu'] = round(self.app_state.initial_api_data_template['phan_tram_xiu'] * 10, 2)


# --- Lớp chiến lược dự đoán ---
class PredictionStrategy:
    """
    Đóng gói logic dự đoán thông minh.
    Ưu tiên các phương pháp dự đoán:
    1. Đảo ngược dựa trên thua (ghi đè nếu số lần thua liên tiếp đạt ngưỡng)
    2. Xác suất có điều kiện (nếu tín hiệu mạnh)
    3. Phân tích chuỗi (nếu xu hướng rõ ràng)
    4. Mặc định (dựa trên đa số pattern tổng thể hoặc ngẫu nhiên nếu cân bằng)
    """
    def __init__(self, app_state: AppState):
        self.app_state = app_state

    def _analyze_streaks(self) -> tuple[int, str | None]:
        """Phân tích độ dài và loại chuỗi hiện tại từ lịch sử."""
        if not self.app_state.history_results:
            return 0, None

        current_streak_length = 0
        current_streak_type = None

        for i in range(len(self.app_state.history_results) - 1, -1, -1):
            result = self.app_state.history_results[i]['Result']
            if current_streak_type is None:
                current_streak_type = result
                current_streak_length = 1
            elif result == current_streak_type:
                current_streak_length += 1
            else:
                break
        return current_streak_length, current_streak_type

    def _calculate_conditional_probability(self, lookback_length: int) -> dict[str, dict[str, float]]:
        """
        Tính toán xác suất có điều kiện của 't' hoặc 'x' dựa trên
        các kết quả 'lookback_length' trước đó.
        Trả về: {'tiền tố': {'t': prob_t, 'x': prob_x}}
        """
        if len(self.app_state.history_results) < lookback_length + 1:
            return {}

        probabilities: dict[str, dict[str, int | float]] = {}
        results_chars = "".join([entry['Result'] for entry in self.app_state.history_results])

        for i in range(len(results_chars) - lookback_length):
            prefix = results_chars[i : i + lookback_length]
            next_char = results_chars[i + lookback_length]

            if prefix not in probabilities:
                probabilities[prefix] = {'t': 0, 'x': 0, 'total': 0}
            
            probabilities[prefix][next_char] = int(probabilities[prefix][next_char]) + 1
            probabilities[prefix]['total'] = int(probabilities[prefix]['total']) + 1
        
        final_probs: dict[str, dict[str, float]] = {}
        for prefix, counts in probabilities.items():
            total_count = float(counts['total']) # Đảm bảo phép chia là số thực
            if total_count > 0:
                final_probs[prefix] = {
                    't': float(counts['t']) / total_count,
                    'x': float(counts['x']) / total_count
                }
            else:
                final_probs[prefix] = {'t': 0.0, 'x': 0.0}

        return final_probs

    def perform_prediction(self):
        """
        Thực hiện logic dự đoán thông minh cho phiên tiếp theo.
        Cập nhật 'du_doan' và 'ly_do' trong trạng thái ứng dụng.
        """
        predicted_result: str = "Không có"
        prediction_reason: str = "Chưa có dữ liệu dự đoán."

        # --- Chiến lược 1: Đảo ngược dựa trên thua ---
        # Đây là ưu tiên cao nhất, ghi đè các dự đoán khác nếu đang hoạt động.
        if self.app_state.last_prediction_info["consecutive_losses"] >= AppConfig.REVERSE_THRESHOLD:
            # Nếu chúng ta đã thua, đảo ngược kết quả thực tế cuối cùng nếu có, hoặc chỉ đơn giản là lật
            last_actual = self.app_state.last_prediction_info["last_actual_result"]
            if last_actual == 't':
                predicted_result = "Xỉu"
            elif last_actual == 'x':
                predicted_result = "Tài"
            else: # Dự phòng nếu không có kết quả thực tế cuối cùng
                predicted_result = random.choice(["Tài", "Xỉu"])
            
            prediction_reason = f"Đang trật {self.app_state.last_prediction_info['consecutive_losses']} lần → Auto đảo ngược."
            logging.info(f"Đã kích hoạt đảo ngược dựa trên thua: {prediction_reason}")
        
        # --- Chiến lược 2: Xác suất có điều kiện ---
        # Chỉ xem xét nếu không có đảo ngược dựa trên thua
        if predicted_result == "Không có" or "Đang trật" not in prediction_reason:
            if len(self.app_state.history_results) >= AppConfig.CONDITIONAL_PROB_LOOKBACK:
                recent_prefix_chars = "".join([entry['Result'] for entry in self.app_state.history_results])[-AppConfig.CONDITIONAL_PROB_LOOKBACK:]
                conditional_probs = self._calculate_conditional_probability(AppConfig.CONDITIONAL_PROB_LOOKBACK)

                if recent_prefix_chars in conditional_probs:
                    prob_t = conditional_probs[recent_prefix_chars]['t']
                    prob_x = conditional_probs[recent_prefix_chars]['x']

                    if prob_t > prob_x and prob_t >= AppConfig.PROB_THRESHOLD_STRONG:
                        # Ghi đè dự đoán nếu xác suất có điều kiện mạnh hơn
                        predicted_result = "Tài"
                        prediction_reason = f"Xác suất Tài cao ({round(prob_t*100, 2)}%) sau '{recent_prefix_chars}'."
                        logging.info(f"Dự đoán xác suất có điều kiện: {prediction_reason}")
                    elif prob_x > prob_t and prob_x >= AppConfig.PROB_THRESHOLD_STRONG:
                        # Ghi đè dự đoán nếu xác suất có điều kiện mạnh hơn
                        predicted_result = "Xỉu"
                        prediction_reason = f"Xác suất Xỉu cao ({round(prob_x*100, 2)}%) sau '{recent_prefix_chars}'."
                        logging.info(f"Dự đoán xác suất có điều kiện: {prediction_reason}")
        
        # --- Chiến lược 3: Phân tích chuỗi ---
        # Chỉ xem xét nếu không có đảo ngược dựa trên thua và không có xác suất có điều kiện mạnh
        if predicted_result == "Không có" or ("Đang trật" not in prediction_reason and "Xác suất" not in prediction_reason):
            current_streak_length, current_streak_type = self._analyze_streaks()

            if current_streak_type:
                if current_streak_length >= AppConfig.MIN_STREAK_FOR_PREDICTION:
                    if current_streak_length < AppConfig.BREAK_STREAK_THRESHOLD:
                        # Theo cầu
                        if current_streak_type == 't':
                            predicted_result = "Tài"
                            prediction_reason = f"Theo cầu Tài dài ({current_streak_length} lần)."
                        else:
                            predicted_result = "Xỉu"
                            prediction_reason = f"Theo cầu Xỉu dài ({current_streak_length} lần)."
                        logging.info(f"Dự đoán theo chuỗi: {prediction_reason}")
                    else:
                        # Cân nhắc bẻ cầu
                        if current_streak_type == 't':
                            predicted_result = "Xỉu"
                            prediction_reason = f"Bẻ cầu Tài dài ({current_streak_length} lần) có khả năng đảo chiều."
                        else:
                            predicted_result = "Tài"
                            prediction_reason = f"Bẻ cầu Xỉu dài ({current_streak_length} lần) có khả năng đảo chiều."
                        logging.info(f"Dự đoán bẻ cầu: {prediction_reason}")
                else:
                    prediction_reason = "Không có cầu rõ ràng."
            else:
                prediction_reason = "Chưa đủ dữ liệu để phân tích cầu."

        # --- Chiến lược 4: Dự phòng mặc định ---
        # Nếu không có tín hiệu mạnh từ các chiến lược trên
        if predicted_result == "Không có" or ("Đang trật" not in prediction_reason and "Xác suất" not in prediction_reason and "cầu" not in prediction_reason):
            pattern_percent_tai = self.app_state.initial_api_data_template['pattern_percent_tai']
            pattern_percent_xiu = self.app_state.initial_api_data_template['pattern_percent_xiu']

            if pattern_percent_tai > pattern_percent_xiu:
                predicted_result = "Tài"
                prediction_reason = "Mặc định: Theo tỷ lệ pattern Tài lớn hơn (không có tín hiệu mạnh khác)."
            elif pattern_percent_xiu > pattern_percent_tai:
                predicted_result = "Xỉu"
                prediction_reason = "Mặc định: Theo tỷ lệ pattern Xỉu lớn hơn (không có tín hiệu mạnh khác)."
            else:
                predicted_result = random.choice(["Tài", "Xỉu"])
                prediction_reason = "Mặc định: Các tín hiệu cân bằng, dự đoán ngẫu nhiên."
            logging.info(f"Dự đoán mặc định: {prediction_reason}")

        self.app_state.initial_api_data_template['du_doan'] = predicted_result
        self.app_state.initial_api_data_template['ly_do'] = prediction_reason

        # Lưu dự đoán này để đánh giá trong phiên tiếp theo
        self.app_state.update_last_prediction_info(
            predicted_expect=self.app_state.initial_api_data_template["phien_du_doan"],
            predicted_result=predicted_result
        )

# --- Thiết lập ứng dụng Flask ---
app = Flask(__name__)
app_state = AppState()
session_processor = SessionDataProcessor(app_state)
prediction_strategy = PredictionStrategy(app_state)

# Hàm sẽ được chạy bởi scheduler
def scheduled_api_check():
    logging.info("Tác vụ kiểm tra API tự động đã được kích hoạt.")
    try:
        response = requests.get(AppConfig.EXTERNAL_API_URL, timeout=10)
        response.raise_for_status() # Ném lỗi nếu mã trạng thái là lỗi (4xx hoặc 5xx)
        external_data = response.json()
        logging.debug(f"Dữ liệu thô từ API bên ngoài (tự động): {external_data}")

        new_session_data = external_data # Dữ liệu API mới nằm trực tiếp ở root
        
        # Kiểm tra xem dữ liệu có hợp lệ không trước khi xử lý (ví dụ: có trường 'Phien' và 'Ket_qua' không)
        if new_session_data and new_session_data.get("Phien") is not None and new_session_data.get("Ket_qua") is not None:
            session_processor.process_new_session(new_session_data)
            prediction_strategy.perform_prediction()
            logging.info(f"Cập nhật tự động thành công. Dự đoán hiện tại cho {app_state.initial_api_data_template['phien_du_doan']}: {app_state.initial_api_data_template['du_doan']}")
        else:
            logging.error(f"Kiểm tra tự động: Dữ liệu không hợp lệ hoặc thiếu 'Phien'/'Ket_qua' từ API bên ngoài. Phản hồi thô: {external_data}")
    except requests.exceptions.Timeout:
        logging.error(f"Kiểm tra tự động: Yêu cầu đến API bên ngoài đã hết thời gian chờ sau 10 giây.")
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Kiểm tra tự động: Không thể kết nối đến API bên ngoài: {e}.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Kiểm tra tự động: Lỗi trong quá trình yêu cầu đến API bên ngoài: {e}.")
    except Exception as e:
        logging.exception(f"Kiểm tra tự động: Lỗi máy chủ nội bộ: {e}") # Ghi log toàn bộ traceback cho lỗi không mong muốn


@app.route('/')
def home():
    """Điểm cuối trang chủ."""
    return "Chào mừng đến với API dự đoán Tài Xỉu trên Render! Truy cập /predict để xem dự đoán."

@app.route('/predict', methods=['GET'])
def get_prediction():
    """
    Điểm cuối chính để lấy dữ liệu mới nhất từ API bên ngoài,
    cập nhật trạng thái và trả về dự đoán cho phiên tiếp theo.
    Hàm này vẫn có thể được gọi thủ công để kích hoạt cập nhật ngay lập tức.
    """
    logging.info("Điểm cuối /predict thủ công đã được gọi.")
    try:
        response = requests.get(AppConfig.EXTERNAL_API_URL, timeout=10) # Đã thêm timeout
        response.raise_for_status()
        external_data = response.json()
        logging.debug(f"Dữ liệu thô từ API bên ngoài (thủ công): {external_data}")

        new_session_data = external_data # Dữ liệu API mới nằm trực tiếp ở root
        
        if new_session_data and new_session_data.get("Phien") is not None and new_session_data.get("Ket_qua") is not None:
            session_processor.process_new_session(new_session_data)
            prediction_strategy.perform_prediction()

            return jsonify(copy.deepcopy(app_state.initial_api_data_template)), 200
        else:
            error_message = "Dữ liệu không hợp lệ hoặc thiếu 'Phien'/'Ket_qua' từ API bên ngoài."
            logging.error(f"Lỗi: {error_message} - Phản hồi thô: {external_data}")
            return jsonify({"error": error_message, "raw_response": external_data}), 500

    except requests.exceptions.Timeout:
        error_message = f"Yêu cầu đến API bên ngoài đã hết thời gian chờ sau 10 giây."
        logging.error(error_message)
        return jsonify({"error": error_message}), 504 # Gateway Timeout
    except requests.exceptions.ConnectionError as e:
        error_message = f"Không thể kết nối đến API bên ngoài: {e}. Vui lòng kiểm tra URL và mạng."
        logging.error(error_message)
        return jsonify({"error": error_message}), 503 # Service Unavailable
    except requests.exceptions.RequestException as e:
        error_message = f"Lỗi trong quá trình yêu cầu đến API bên ngoài: {e}."
        logging.error(error_message)
        return jsonify({"error": error_message}), 500
    except Exception as e:
        error_message = f"Lỗi máy chủ nội bộ trong quá trình dự đoán: {e}"
        logging.exception(error_message) # Sử dụng exception để in toàn bộ traceback
        return jsonify({"error": error_message}), 500

@app.route('/status', methods=['GET'])
def get_current_status():
    """
    Điểm cuối để lấy trạng thái dự đoán hiện tại mà không cần gọi API bên ngoài.
    """
    return jsonify(copy.deepcopy(app_state.initial_api_data_template)), 200

@app.route('/history', methods=['GET'])
def get_history():
    """
    Điểm cuối để xem lịch sử phiên đã xử lý (trong bộ nhớ).
    """
    return jsonify(list(app_state.history_results)), 200

@app.route('/last_prediction_info', methods=['GET'])
def get_last_prediction_info_route():
    """
    Điểm cuối để xem thông tin về dự đoán cuối cùng, số lần thua liên tiếp,
    và thống kê dự đoán mới.
    """
    return jsonify(app_state.last_prediction_info), 200

@app.route('/prediction_stats', methods=['GET'])
def get_prediction_stats():
    """
    ĐIỂM CUỐI MỚI: Trả về tổng số dự đoán đã đưa ra,
    và bao nhiêu dự đoán đúng so với sai.
    """
    stats = {
        "total_predictions": app_state.last_prediction_info["total_predictions"],
        "correct_predictions": app_state.last_prediction_info["correct_predictions"],
        "incorrect_predictions": app_state.last_prediction_info["incorrect_predictions"]
    }
    return jsonify(stats), 200

# --- Chạy ứng dụng Flask ---
if __name__ == '__main__':
    # Khởi tạo và bắt đầu scheduler
    scheduler = BackgroundScheduler()
    # Thêm job để chạy hàm scheduled_api_check mỗi AppConfig.CHECK_INTERVAL_SECONDS
    scheduler.add_job(func=scheduled_api_check, trigger="interval", seconds=AppConfig.CHECK_INTERVAL_SECONDS)
    scheduler.start()
    logging.info("Scheduler đã khởi động. Các tác vụ kiểm tra API tự động hiện đang hoạt động.")

    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Đang khởi động ứng dụng Flask trên cổng {port}")
    # Chạy ứng dụng Flask. use_reloader=False để tránh job chạy 2 lần trong debug mode
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
