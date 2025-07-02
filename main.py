import copy
import logging
import os
import random
import requests
import collections # Explicitly import collections for deque
from flask import Flask, jsonify, request

# --- Configure Logging ---
# Set up logging to output messages to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Class ---
class AppConfig:
    """
    Manages application-wide configurations, making them easily adjustable.
    """
    # UPDATED: New External API URL
    EXTERNAL_API_URL: str = os.getenv("EXTERNAL_API_URL", "https://apib52.up.railway.app/api/taixiumd5")
    HISTORY_MAX_LEN: int = 100  # Max number of sessions to keep in history
    DEFAULT_PATTERN_LENGTH: int = 8

    # Prediction Strategy Thresholds
    MIN_STREAK_FOR_PREDICTION: int = 3
    BREAK_STREAK_THRESHOLD: int = 5
    CONDITIONAL_PROB_LOOKBACK: int = 3
    PROB_THRESHOLD_STRONG: float = 0.60
    REVERSE_THRESHOLD: int = 3 # Number of consecutive losses to trigger reversal


# --- Application State Class ---
class AppState:
    """
    Manages the global state of the application, including history and prediction info.
    """
    def __init__(self):
        self.history_results: collections.deque = collections.deque(maxlen=AppConfig.HISTORY_MAX_LEN)
        self.last_prediction_info: dict = {
            "predicted_expect": None, # Will store the 'Phien' number now
            "predicted_result": None,
            "consecutive_losses": 0,
            "last_actual_result": None, # 't' or 'x'
            "total_predictions": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0
        }
        self.initial_api_data_template: dict = {
            # UPDATED: Use 'Phien_moi' for current session number
            "Phien_moi": None,
            "pattern_length": AppConfig.DEFAULT_PATTERN_LENGTH,
            "pattern": "xxxxxxxx",
            "matches": ["x"], # Keep as is, will store last result char
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
            # UPDATED: 'phien_du_doan' for the next session number
            "phien_du_doan": None,
            "admin_info": "@heheviptool"
        }

    def reset_prediction_info(self):
        """Resets consecutive losses and related prediction tracking, but not total stats."""
        self.last_prediction_info["predicted_expect"] = None
        self.last_prediction_info["predicted_result"] = None
        self.last_prediction_info["consecutive_losses"] = 0
        self.last_prediction_info["last_actual_result"] = None


    def update_last_prediction_info(self, predicted_expect: int, predicted_result: str):
        """Updates the last prediction details for tracking."""
        self.last_prediction_info["predicted_expect"] = predicted_expect
        self.last_prediction_info["predicted_result"] = predicted_result

    def update_prediction_stats(self, processed_session_expect: int, actual_result_char: str):
        """
        Updates the prediction statistics (total, correct, incorrect) and consecutive losses.
        This function is called *after* a new session's actual result is known
        and compares it against the *previous* prediction.
        """
        predicted_expect = self.last_prediction_info["predicted_expect"]
        predicted_result_str = self.last_prediction_info["predicted_result"]

        # Only evaluate if there was a prediction made for this specific session
        if predicted_expect is not None and predicted_expect == processed_session_expect and predicted_result_str is not None:
            
            predicted_res_char = "t" if predicted_result_str == "Tài" else "x"
            
            self.last_prediction_info["total_predictions"] += 1
            logging.debug(f"Evaluating prediction for session {processed_session_expect}. Predicted: {predicted_result_str} ({predicted_res_char}), Actual: {actual_result_char}")

            if predicted_res_char != actual_result_char:
                self.last_prediction_info["consecutive_losses"] += 1
                self.last_prediction_info["incorrect_predictions"] += 1
                logging.info(f"Prediction '{predicted_result_str}' for session {processed_session_expect} MISSED. Consecutive losses: {self.last_prediction_info['consecutive_losses']}")
            else:
                self.last_prediction_info["consecutive_losses"] = 0
                self.last_prediction_info["correct_predictions"] += 1
                logging.info(f"Prediction '{predicted_result_str}' for session {processed_session_expect} CORRECT. Resetting losses.")
        else:
            # This handles cases where no prediction was made for the current incoming session,
            # or it's the first run, or app restarted. Reset losses if no matching prediction.
            self.last_prediction_info["consecutive_losses"] = 0
            logging.debug(f"No matching previous prediction for session {processed_session_expect} to evaluate. Resetting losses.")
        
        self.last_prediction_info["last_actual_result"] = actual_result_char


# --- Helper Functions ---
# UPDATED: calculate_tai_xiu now takes individual dice values
def calculate_tai_xiu(dice_sum: int) -> tuple[str, int]:
    """
    Calculates the sum of dice values and determines "Tài" (Over) or "Xỉu" (Under).
    Returns a tuple of (result_type: str, total_sum: int).
    """
    try:
        # The new API provides 'Tong' directly, so we use that.
        total_sum = dice_sum

        if 4 <= total_sum <= 10:
            return "Xỉu", total_sum
        elif 11 <= total_sum <= 17:
            return "Tài", total_sum
        # The API is for Tai Xiu MD5, usually 1,1,1 and 6,6,6 are special cases.
        # Assuming typical rules: Triple 1s (sum 3) is Xỉu, Triple 6s (sum 18) is Tài.
        elif total_sum == 3: # Triple 1s - considered Xỉu
            return "Xỉu", total_sum
        elif total_sum == 18: # Triple 6s - considered Tài
            return "Tài", total_sum
        else:
            logging.warning(f"Unexpected sum: {total_sum}")
            return "Không xác định", total_sum
    except (ValueError, TypeError) as e:
        logging.error(f"Error calculating Tai/Xiu from sum {dice_sum}: {e}")
        return "Lỗi", 0

# UPDATED: get_next_expect_code now increments an integer (Phien number)
def get_next_expect_code(current_expect_code: int) -> int | None:
    """
    Calculates the Expect code (Phien number) for the next session by incrementing.
    """
    if not isinstance(current_expect_code, int):
        logging.warning(f"Expect code '{current_expect_code}' is not an integer. Cannot calculate next expect code.")
        return None
    
    return current_expect_code + 1

# --- Data Processor ---
class SessionDataProcessor:
    """
    Handles processing of new session data from the external API and updates AppState.
    """
    def __init__(self, app_state: AppState):
        self.app_state = app_state

    def process_new_session(self, new_session_data: dict):
        """
        Updates history and global state based on new session data.
        New API format: {"Ket_qua":"Tài","Phien":1450954,"Tong":11,"Xuc_xac_1":5,"Xuc_xac_2":2,"Xuc_xac_3":4,"id": <some_id>}
        """
        # UPDATED: Extract data based on new API keys
        current_phien = new_session_data.get('Phien')
        current_ket_qua = new_session_data.get('Ket_qua')
        current_tong = new_session_data.get('Tong')
        current_id = new_session_data.get('id') # Using 'id' if available for history tracking

        if current_phien is None or current_ket_qua is None or current_tong is None:
            logging.error(f"Missing required data in new session: {new_session_data}")
            return # Exit if critical data is missing

        actual_result_type, _ = calculate_tai_xiu(current_tong)
        # Ensure 't' or 'x' is used for internal logic
        actual_result_char = "t" if "Tài" in actual_result_type else "x"

        # Check if this session is already in history to avoid duplicates
        # Using 'Phien' as the unique identifier if 'id' is not reliable/present
        if not any(entry['Phien'] == current_phien for entry in self.app_state.history_results):
            self.app_state.history_results.append({
                "ID": current_id, # Can still store 'id' from new API
                "Phien": current_phien,
                "Tong": current_tong,
                "Result": actual_result_char # Store 't' or 'x'
            })
            logging.info(f"Added new session to history: Phien {current_phien}, Result: {actual_result_type}")
            
            # After adding a new session, update consecutive losses AND prediction stats
            # Pass current_phien as the processed_session_expect
            self.app_state.update_prediction_stats(current_phien, actual_result_char)
        else:
            logging.debug(f"Session Phien {current_phien} already in history. Skipping addition.")


        # UPDATED: Update core fields in the template based on new API fields
        self.app_state.initial_api_data_template["Phien_moi"] = current_phien
        
        next_expect_code = get_next_expect_code(current_phien)
        self.app_state.initial_api_data_template["phien_du_doan"] = next_expect_code if next_expect_code else "Không xác định"

        # Update pattern and pattern percentages
        # UPDATED: Access 'Result' from history_results
        current_pattern_chars = "".join([entry['Result'] for entry in self.app_state.history_results])
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

        # Update 'matches' (assuming it refers to the latest result)
        if self.app_state.history_results:
            self.app_state.initial_api_data_template['matches'] = [self.app_state.history_results[-1]['Result']]
        else:
            self.app_state.initial_api_data_template['matches'] = []

        # Assume phan_tram_tai/xiu and tong_tai/xiu are derived from pattern_percent
        self.app_state.initial_api_data_template['phan_tram_tai'] = self.app_state.initial_api_data_template['pattern_percent_tai']
        self.app_state.initial_api_data_template['phan_tram_xiu'] = self.app_state.initial_api_data_template['pattern_percent_xiu']
        
        # Arbitrarily scale 'tong_tai' and 'tong_xiu' for the JSON output
        self.app_state.initial_api_data_template['tong_tai'] = round(self.app_state.initial_api_data_template['phan_tram_tai'] * 10, 2)
        self.app_state.initial_api_data_template['tong_xiu'] = round(self.app_state.initial_api_data_template['phan_tram_xiu'] * 10, 2)


# --- Prediction Strategy Class ---
class PredictionStrategy:
    """
    Encapsulates the intelligent prediction logic.
    Prioritizes prediction methods:
    1. Loss-based reversal (override if consecutive losses hit threshold)
    2. Conditional Probability (if strong signal)
    3. Streak Analysis (if clear trend)
    4. Default (based on overall pattern majority or random if balanced)
    """
    def __init__(self, app_state: AppState):
        self.app_state = app_state

    def _analyze_streaks(self) -> tuple[int, str | None]:
        """Analyzes the current streak length and type from history."""
        if not self.app_state.history_results:
            return 0, None

        current_streak_length = 0
        current_streak_type = None

        for i in range(len(self.app_state.history_results) - 1, -1, -1):
            result = self.app_state.history_results[i]['Result'] # Access 'Result' field
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
        Calculates conditional probabilities of 't' or 'x' based on 'lookback_length'
        previous results.
        Returns: {'prefix': {'t': prob_t, 'x': prob_x}}
        """
        if len(self.app_state.history_results) < lookback_length + 1:
            return {}

        probabilities: dict[str, dict[str, int | float]] = {}
        # Access 'Result' field
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
            total_count = float(counts['total']) # Ensure division is float
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
        Executes the intelligent prediction logic for the next session.
        Updates 'du_doan' and 'ly_do' in the application state.
        """
        predicted_result: str = "Không có"
        prediction_reason: str = "Chưa có dữ liệu dự đoán."

        # --- Strategy 1: Loss-based reversal ---
        # This is the highest priority, overriding other predictions if active.
        if self.app_state.last_prediction_info["consecutive_losses"] >= AppConfig.REVERSE_THRESHOLD:
            # If we've been losing, reverse the last actual outcome if available, or just flip
            last_actual = self.app_state.last_prediction_info["last_actual_result"]
            if last_actual == 't':
                predicted_result = "Xỉu"
            elif last_actual == 'x':
                predicted_result = "Tài"
            else: # Fallback if no last actual result
                predicted_result = random.choice(["Tài", "Xỉu"])
            
            prediction_reason = f"Đang trật {self.app_state.last_prediction_info['consecutive_losses']} lần → Auto đảo ngược."
            logging.info(f"Loss-based reversal activated: {prediction_reason}")
        
        # --- Strategy 2: Conditional Probability ---
        # Only consider if no loss-based reversal
        if predicted_result == "Không có" or "Đang trật" not in prediction_reason:
            if len(self.app_state.history_results) >= AppConfig.CONDITIONAL_PROB_LOOKBACK:
                # Access 'Result' field
                recent_prefix_chars = "".join([entry['Result'] for entry in self.app_state.history_results])[-AppConfig.CONDITIONAL_PROB_LOOKBACK:]
                conditional_probs = self._calculate_conditional_probability(AppConfig.CONDITIONAL_PROB_LOOKBACK)

                if recent_prefix_chars in conditional_probs:
                    prob_t = conditional_probs[recent_prefix_chars]['t']
                    prob_x = conditional_probs[recent_prefix_chars]['x']

                    if prob_t > prob_x and prob_t >= AppConfig.PROB_THRESHOLD_STRONG:
                        # Ghi đè dự đoán nếu xác suất có điều kiện mạnh hơn
                        predicted_result = "Tài"
                        prediction_reason = f"Xác suất Tài cao ({round(prob_t*100, 2)}%) sau '{recent_prefix_chars}'."
                        logging.info(f"Conditional probability prediction: {prediction_reason}")
                    elif prob_x > prob_t and prob_x >= AppConfig.PROB_THRESHOLD_STRONG:
                        # Ghi đè dự đoán nếu xác suất có điều kiện mạnh hơn
                        predicted_result = "Xỉu"
                        prediction_reason = f"Xác suất Xỉu cao ({round(prob_x*100, 2)}%) sau '{recent_prefix_chars}'."
                        logging.info(f"Conditional probability prediction: {prediction_reason}")
        
        # --- Strategy 3: Streak Analysis ---
        # Only consider if no loss-based reversal and no strong conditional probability
        if predicted_result == "Không có" or ("Đang trật" not in prediction_reason and "Xác suất" not in prediction_reason):
            current_streak_length, current_streak_type = self._analyze_streaks()

            if current_streak_type:
                if current_streak_length >= AppConfig.MIN_STREAK_FOR_PREDICTION:
                    if current_streak_length < AppConfig.BREAK_STREAK_THRESHOLD:
                        # Follow the streak
                        if current_streak_type == 't':
                            predicted_result = "Tài"
                            prediction_reason = f"Theo cầu Tài dài ({current_streak_length} lần)."
                        else:
                            predicted_result = "Xỉu"
                            prediction_reason = f"Theo cầu Xỉu dài ({current_streak_length} lần)."
                        logging.info(f"Streak prediction: {prediction_reason}")
                    else:
                        # Consider breaking the streak
                        if current_streak_type == 't':
                            predicted_result = "Xỉu"
                            prediction_reason = f"Bẻ cầu Tài dài ({current_streak_length} lần) có khả năng đảo chiều."
                        else:
                            predicted_result = "Tài"
                            prediction_reason = f"Bẻ cầu Xỉu dài ({current_streak_length} lần) có khả năng đảo chiều."
                        logging.info(f"Streak break prediction: {prediction_reason}")
                else:
                    prediction_reason = "Không có cầu rõ ràng."
            else:
                prediction_reason = "Chưa đủ dữ liệu để phân tích cầu."

        # --- Strategy 4: Default Fallback ---
        # If no strong signals from the above strategies
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
            logging.info(f"Default prediction: {prediction_reason}")

        self.app_state.initial_api_data_template['du_doan'] = predicted_result
        self.app_state.initial_api_data_template['ly_do'] = prediction_reason

        # Save this prediction for evaluation in the next session
        self.app_state.update_last_prediction_info(
            predicted_expect=self.app_state.initial_api_data_template["phien_du_doan"],
            predicted_result=predicted_result
        )

# --- Flask Application Setup ---
app = Flask(__name__)
app_state = AppState()
session_processor = SessionDataProcessor(app_state)
prediction_strategy = PredictionStrategy(app_state)

@app.route('/')
def home():
    """Home endpoint."""
    return "Chào mừng đến với API dự đoán Tài Xỉu trên Render! Truy cập /predict để xem dự đoán."

@app.route('/predict', methods=['GET'])
def get_prediction():
    """
    Main endpoint to fetch the latest data from the external API,
    update the state, and return the prediction for the next session.
    """
    try:
        logging.info(f"Calling external API: {AppConfig.EXTERNAL_API_URL}")
        response = requests.get(AppConfig.EXTERNAL_API_URL, timeout=10) # Added timeout
        response.raise_for_status()
        external_data = response.json()
        logging.debug(f"Raw data from external API: {external_data}")

        # The new API directly returns the session data, not wrapped in 'data' key or 'state'
        # Example: {"Ket_qua":"Tài","Phien":1450954,"Tong":11,"Xuc_xac_1":5,"Xuc_xac_2":2,"Xuc_xac_3":4,"id": <some_id>}
        new_session_data = external_data # The whole response is the session data

        # Validate basic expected fields
        if not all(k in new_session_data for k in ["Ket_qua", "Phien", "Tong"]):
            error_message = "Invalid or incomplete data from external API. Missing 'Ket_qua', 'Phien', or 'Tong'."
            logging.error(f"Error: {error_message} - Raw response: {external_data}")
            return jsonify({"error": error_message, "raw_response": external_data}), 500

        session_processor.process_new_session(new_session_data)
        prediction_strategy.perform_prediction()

        return jsonify(copy.deepcopy(app_state.initial_api_data_template)), 200

    except requests.exceptions.Timeout:
        error_message = f"Request to external API timed out after 10 seconds."
        logging.error(error_message)
        return jsonify({"error": error_message}), 504 # Gateway Timeout
    except requests.exceptions.ConnectionError as e:
        error_message = f"Failed to connect to external API: {e}. Please check URL and network."
        logging.error(error_message)
        return jsonify({"error": error_message}), 503 # Service Unavailable
    except requests.exceptions.RequestException as e:
        error_message = f"Error during request to external API: {e}."
        logging.error(error_message)
        return jsonify({"error": error_message}), 500
    except Exception as e:
        error_message = f"Internal server error during prediction: {e}"
        logging.exception(error_message) # Use exception for full traceback
        return jsonify({"error": error_message}), 500

@app.route('/status', methods=['GET'])
def get_current_status():
    """
    Endpoint to get the current prediction status without calling the external API.
    """
    return jsonify(copy.deepcopy(app_state.initial_api_data_template)), 200

@app.route('/history', methods=['GET'])
def get_history():
    """
    Endpoint to view the processed session history (in memory).
    """
    return jsonify(list(app_state.history_results)), 200

@app.route('/last_prediction_info', methods=['GET'])
def get_last_prediction_info_route():
    """
    Endpoint to view information about the last prediction, consecutive losses,
    and the new prediction statistics.
    """
    return jsonify(app_state.last_prediction_info), 200

@app.route('/prediction_stats', methods=['GET'])
def get_prediction_stats():
    """
    NEW ENDPOINT: Returns the total number of predictions made,
    and how many were correct vs. incorrect.
    """
    stats = {
        "total_predictions": app_state.last_prediction_info["total_predictions"],
        "correct_predictions": app_state.last_prediction_info["correct_predictions"],
        "incorrect_predictions": app_state.last_prediction_info["incorrect_predictions"]
    }
    return jsonify(stats), 200

# --- Run the Flask Application ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Starting Flask app on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)

