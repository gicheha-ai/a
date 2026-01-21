import os
import json
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.graph_objects as go
import plotly.utils
import requests
import warnings
import logging
from logging.handlers import RotatingFileHandler
import pickle
import hashlib

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configuration
API_KEYS = ['TZHOOJ7CNDMQ7HYD' , 'QSCWPKVUYLOD506J']
current_key_index = 0
BASE_URL = "https://www.alphavantage.co/query"

# Trading parameters
PREDICTION_WINDOW_MINUTES = 2
TRADING_SYMBOL = "EURGBP"
INITIAL_BALANCE = 10000.0  # Starting virtual balance

# Global state
trading_state = {
    'current_price': 0.0,
    'prediction': 'NEUTRAL',
    'action': 'HOLD',
    'confidence': 0.0,
    'next_prediction_time': None,
    'current_trade': None,
    'balance': INITIAL_BALANCE,
    'total_trades': 0,
    'profitable_trades': 0,
    'total_profit': 0.0,
    'win_rate': 0.0,
    'indicators': {},
    'is_demo_data': False,
    'last_update': None,
    'model_accuracy': 0.0
}

# Setup logging
log_dir = 'trading_logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'trading_{datetime.now().strftime("%Y%m%d")}.log')

handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger('trading_bot')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def switch_api_key():
    """Switch to next API key if rate limited"""
    global current_key_index
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    logger.info(f"Switched to API key index: {current_key_index}")

def fetch_forex_data(symbol='EURGBP', interval='1min', outputsize='compact'):
    """Fetch forex data from Alpha Vantage"""
    for attempt in range(len(API_KEYS)):
        try:
            params = {
                'function': 'FX_INTRADAY',
                'from_symbol': symbol[:3],
                'to_symbol': symbol[3:],
                'interval': interval,
                'outputsize': outputsize,
                'apikey': API_KEYS[current_key_index],
                'datatype': 'json'
            }
            
            logger.info(f"Fetching data with API key {current_key_index}...")
            response = requests.get(BASE_URL, params=params, timeout=15)
            data = response.json()
            
            if "Time Series FX (" + interval + ")" not in data:
                if "Note" in data:
                    logger.warning(f"Rate limited: {data['Note']}")
                    switch_api_key()
                    time.sleep(1)
                    continue
                logger.error(f"API Error response: {data}")
                return create_mock_data(symbol), True
            
            time_series = data["Time Series FX (" + interval + ")"]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['open', 'high', 'low', 'close']
            
            # Convert to numeric
            df = df.apply(pd.to_numeric, errors='coerce')
            
            logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
            return df, False
            
        except Exception as e:
            logger.error(f"API attempt {attempt + 1} failed: {str(e)}")
            switch_api_key()
            time.sleep(1)
    
    logger.warning("All API attempts failed, using mock data")
    return create_mock_data(symbol), True

def create_mock_data(symbol='EURGBP'):
    """Create realistic mock data for EUR/GBP"""
    dates = pd.date_range(end=datetime.now(), periods=200, freq='1min')
    
    # EUR/GBP typically ranges between 0.85 and 0.90
    base_price = 0.8575
    prices = []
    volatility = 0.0003
    
    for i in range(len(dates)):
        # Add some trend and randomness
        trend = 0.00001 if i % 50 < 25 else -0.00001
        random_move = np.random.normal(0, volatility)
        base_price += trend + random_move
        
        # Keep within realistic bounds
        base_price = max(0.8550, min(0.8600, base_price))
        prices.append(base_price)
    
    df = pd.DataFrame(index=dates[-100:])
    df['open'] = [p * (1 + np.random.normal(0, 0.0001)) for p in prices[-100:]]
    df['high'] = [p * (1 + abs(np.random.normal(0, 0.0002))) for p in prices[-100:]]
    df['low'] = [p * (1 - abs(np.random.normal(0, 0.0002))) for p in prices[-100:]]
    df['close'] = prices[-100:]
    
    return df

def calculate_advanced_indicators(df):
    """Calculate comprehensive technical indicators"""
    try:
        if len(df) < 50:
            return df
        
        # Price-based indicators
        df['SMA_10'] = ta.sma(df['close'], length=10)
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['EMA_8'] = ta.ema(df['close'], length=8)
        df['EMA_13'] = ta.ema(df['close'], length=13)
        df['EMA_21'] = ta.ema(df['close'], length=21)
        
        # RSI with multiple timeframes
        df['RSI_7'] = ta.rsi(df['close'], length=7)
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        
        # MACD variations
        macd_fast = ta.macd(df['close'], fast=8, slow=17, signal=9)
        if macd_fast is not None:
            df['MACD_fast'] = macd_fast['MACD_8_17_9']
            df['MACD_signal_fast'] = macd_fast['MACDs_8_17_9']
        
        macd_slow = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd_slow is not None:
            df['MACD_slow'] = macd_slow['MACD_12_26_9']
            df['MACD_signal_slow'] = macd_slow['MACDs_12_26_9']
        
        # Bollinger Bands with multiple deviations
        bb_1 = ta.bbands(df['close'], length=20, std=1)
        bb_2 = ta.bbands(df['close'], length=20, std=2)
        if bb_1 is not None and bb_2 is not None:
            df['BB_lower_1'] = bb_1['BBL_20_1.0']
            df['BB_upper_1'] = bb_1['BBU_20_1.0']
            df['BB_lower_2'] = bb_2['BBL_20_2.0']
            df['BB_upper_2'] = bb_2['BBU_20_2.0']
            df['BB_percent'] = (df['close'] - bb_2['BBL_20_2.0']) / (bb_2['BBU_20_2.0'] - bb_2['BBL_20_2.0'])
        
        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        if stoch is not None:
            df['Stoch_K'] = stoch['STOCHk_14_3_3']
            df['Stoch_D'] = stoch['STOCHd_14_3_3']
        
        # ATR for volatility
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # ADX for trend strength
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['ADX'] = adx['ADX_14']
            df['DMP'] = adx['DMP_14']
            df['DMN'] = adx['DMN_14']
        
        # Williams %R
        df['Williams_R'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        # CCI
        df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=20)
        
        # Momentum indicators
        df['Momentum_5'] = ta.mom(df['close'], length=5)
        df['Momentum_10'] = ta.mom(df['close'], length=10)
        
        # Rate of Change
        df['ROC_5'] = ta.roc(df['close'], length=5)
        df['ROC_10'] = ta.roc(df['close'], length=10)
        
        # Volume indicators (using tick volume proxy)
        price_change = df['close'].diff().abs()
        df['Volume_Proxy'] = price_change.rolling(20).mean()
        df['OBV'] = ta.obv(df['close'], df['Volume_Proxy'])
        
        # Ichimoku Cloud (simplified)
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
        if ichimoku is not None:
            df['Ichimoku_Conversion'] = ichimoku['ITS_9']
            df['Ichimoku_Base'] = ichimoku['IKS_26']
            df['Ichimoku_SpanA'] = ichimoku['ISA_9']
            df['Ichimoku_SpanB'] = ichimoku['ISB_26']
        
        # Parabolic SAR
        df['PSAR'] = ta.psar(df['high'], df['low'], df['close'])
        
        # Price patterns
        df['Higher_High'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['Higher_Low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['Lower_High'] = (df['high'] < df['high'].shift(1)).astype(int)
        df['Lower_Low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Price position relative to indicators
        df['Price_vs_SMA20'] = (df['close'] / df['SMA_20'] - 1) * 100
        df['Price_vs_SMA50'] = (df['close'] / df['SMA_50'] - 1) * 100
        df['SMA_Crossover'] = (df['SMA_10'] > df['SMA_20']).astype(int)
        
        # Volatility measures
        df['Returns'] = df['close'].pct_change()
        df['Volatility_10'] = df['Returns'].rolling(10).std() * np.sqrt(252 * 24 * 60)  # Annualized
        df['Volatility_20'] = df['Returns'].rolling(20).std() * np.sqrt(252 * 24 * 60)
        
        # Support and Resistance
        df['Resistance_20'] = df['high'].rolling(20).max()
        df['Support_20'] = df['low'].rolling(20).min()
        df['Distance_to_Resistance'] = (df['Resistance_20'] - df['close']) / df['close'] * 100
        df['Distance_to_Support'] = (df['close'] - df['Support_20']) / df['close'] * 100
        
        # Clean data
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"Calculated {len(df.columns)} indicators")
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return df

def prepare_features_for_prediction(df, prediction_minutes=2):
    """Prepare features for 2-minute price prediction"""
    try:
        # Use last 100 data points
        lookback = min(100, len(df))
        df_recent = df.iloc[-lookback:].copy()
        
        # Feature list - comprehensive set
        base_features = [
            'RSI_7', 'RSI_14', 'MACD_fast', 'MACD_signal_fast',
            'MACD_slow', 'MACD_signal_slow', 'Stoch_K', 'Stoch_D',
            'ATR_14', 'ADX', 'Williams_R', 'CCI', 'Momentum_5',
            'Momentum_10', 'ROC_5', 'ROC_10', 'BB_percent',
            'Price_vs_SMA20', 'Price_vs_SMA50', 'SMA_Crossover',
            'Volatility_10', 'Distance_to_Resistance', 'Distance_to_Support'
        ]
        
        # Only use available features
        available_features = [f for f in base_features if f in df_recent.columns]
        
        X = pd.DataFrame()
        
        # Current values
        for feature in available_features:
            X[f'{feature}_current'] = df_recent[feature].iloc[-1]
        
        # Rate of change for key indicators
        for feature in ['RSI_14', 'MACD_fast', 'Stoch_K', 'ADX']:
            if feature in df_recent.columns:
                X[f'{feature}_roc_3'] = df_recent[feature].iloc[-1] / df_recent[feature].iloc[-4] - 1
        
        # Price action features
        X['price_trend_5'] = (df_recent['close'].iloc[-1] / df_recent['close'].iloc[-6] - 1) * 100
        X['price_trend_10'] = (df_recent['close'].iloc[-1] / df_recent['close'].iloc[-11] - 1) * 100
        
        # Volatility features
        X['recent_volatility'] = df_recent['Returns'].iloc[-10:].std() * 100
        
        # Momentum combination
        if all(f in df_recent.columns for f in ['RSI_14', 'Stoch_K', 'MACD_fast']):
            X['momentum_score'] = (
                (df_recent['RSI_14'].iloc[-1] - 50) / 50 +
                (df_recent['Stoch_K'].iloc[-1] - 50) / 50 +
                np.sign(df_recent['MACD_fast'].iloc[-1])
            ) / 3
        
        # Pattern recognition
        X['higher_high_pattern'] = df_recent['Higher_High'].iloc[-3:].sum() if 'Higher_High' in df_recent.columns else 0
        X['lower_low_pattern'] = df_recent['Lower_Low'].iloc[-3:].sum() if 'Lower_Low' in df_recent.columns else 0
        
        # Time-based features
        now = datetime.now()
        X['hour_of_day'] = now.hour
        X['minute_of_hour'] = now.minute
        X['is_london_session'] = 1 if 8 <= now.hour <= 16 else 0
        X['is_us_session'] = 1 if 13 <= now.hour <= 21 else 0
        
        # Fill any NaN values
        X = X.fillna(0)
        
        return X
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        return pd.DataFrame()

def create_2min_labels(df, start_idx, prediction_minutes=2):
    """Create labels for 2-minute prediction (1 for price increase, 0 for decrease)"""
    try:
        if start_idx + prediction_minutes >= len(df):
            return None
        
        start_price = df['close'].iloc[start_idx]
        end_price = df['close'].iloc[start_idx + prediction_minutes]
        
        # Label: 1 if price increased by at least 0.01%, 0 otherwise
        price_change_pct = (end_price / start_price - 1) * 100
        return 1 if price_change_pct >= 0.01 else 0
        
    except Exception as e:
        logger.error(f"Error creating labels: {str(e)}")
        return None

def train_prediction_model(df, prediction_minutes=2):
    """Train ensemble model for 2-minute predictions"""
    try:
        if len(df) < 100:
            logger.warning(f"Not enough data for training: {len(df)} points")
            return None, None, 0.0
        
        # Prepare training data
        X_list = []
        y_list = []
        
        # Create multiple training examples using sliding window
        for i in range(20, len(df) - prediction_minutes - 10):
            features = prepare_features_for_prediction(df.iloc[:i+1], prediction_minutes)
            if not features.empty:
                label = create_2min_labels(df, i, prediction_minutes)
                if label is not None:
                    X_list.append(features.iloc[0].values)  # Single row
                    y_list.append(label)
        
        if len(X_list) < 50:
            logger.warning(f"Not enough training examples: {len(X_list)}")
            return None, None, 0.0
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train ensemble of models
        models = []
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_scaled, y)
        models.append(('rf', rf))
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb.fit(X_scaled, y)
        models.append(('gb', gb))
        
        # Calculate accuracy on recent data
        recent_cutoff = int(len(X_scaled) * 0.8)
        X_recent = X_scaled[recent_cutoff:]
        y_recent = y[recent_cutoff:]
        
        accuracies = []
        for name, model in models:
            predictions = model.predict(X_recent)
            accuracy = np.mean(predictions == y_recent)
            accuracies.append(accuracy)
            logger.info(f"{name} recent accuracy: {accuracy:.2%}")
        
        avg_accuracy = np.mean(accuracies)
        
        # Create ensemble prediction function
        def ensemble_predict(features):
            features_scaled = scaler.transform(features.reshape(1, -1))
            predictions = []
            weights = [acc for acc in accuracies]
            weights = [w/sum(weights) for w in weights]
            
            for (name, model), weight in zip(models, weights):
                pred = model.predict_proba(features_scaled)[0][1]  # Probability of class 1
                predictions.append(pred * weight)
            
            return sum(predictions)
        
        logger.info(f"Ensemble model trained with {len(X)} examples, accuracy: {avg_accuracy:.2%}")
        return ensemble_predict, scaler, avg_accuracy
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None, None, 0.0

def make_2min_prediction(df, current_price):
    """Make 2-minute price prediction"""
    try:
        # Prepare features for current moment
        features_df = prepare_features_for_prediction(df)
        
        if features_df.empty:
            logger.warning("No features available for prediction")
            return 0.5, 0.5, 'NEUTRAL'  # Neutral prediction with 50% confidence
        
        features = features_df.iloc[0].values
        
        # Train model on historical data
        predict_fn, scaler, accuracy = train_prediction_model(df)
        
        if predict_fn is None:
            logger.warning("Could not train prediction model")
            return 0.5, accuracy, 'NEUTRAL'
        
        # Make prediction
        probability_up = predict_fn(features)
        
        # Determine direction and confidence
        if probability_up > 0.55:
            direction = 'BULLISH'
            confidence = min(probability_up * 100, 95)
        elif probability_up < 0.45:
            direction = 'BEARISH'
            confidence = min((1 - probability_up) * 100, 95)
        else:
            direction = 'NEUTRAL'
            confidence = 50
        
        logger.info(f"Prediction: {direction} with {confidence:.1f}% confidence (Accuracy: {accuracy:.2%})")
        return probability_up, confidence, direction
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return 0.5, 0.5, 'NEUTRAL'

def determine_trading_action(prediction_prob, confidence, indicators, current_price):
    """Determine optimal trading action based on prediction"""
    try:
        # Extract key indicators
        rsi = indicators.get('RSI_14', 50)
        macd = indicators.get('MACD_fast', 0)
        stoch = indicators.get('Stoch_K', 50)
        adx = indicators.get('ADX', 25)
        
        # Strong buy conditions
        if prediction_prob > 0.65 and confidence > 70:
            if rsi < 65 and macd > 0 and stoch < 80:
                return 'BUY', confidence
        
        # Strong sell conditions
        elif prediction_prob < 0.35 and confidence > 70:
            if rsi > 35 and macd < 0 and stoch > 20:
                return 'SELL', confidence
        
        # Moderate buy conditions
        elif prediction_prob > 0.60 and confidence > 60:
            if rsi < 70:
                return 'BUY', confidence * 0.9
        
        # Moderate sell conditions
        elif prediction_prob < 0.40 and confidence > 60:
            if rsi > 30:
                return 'SELL', confidence * 0.9
        
        # Extreme indicator conditions
        if rsi > 80 and stoch > 85:
            return 'SELL', 75
        elif rsi < 20 and stoch < 15:
            return 'BUY', 75
        
        # Trend following with ADX
        if adx > 30:
            if prediction_prob > 0.55:
                return 'BUY', 65
            elif prediction_prob < 0.45:
                return 'SELL', 65
        
        return 'HOLD', confidence
        
    except Exception as e:
        logger.error(f"Error determining action: {str(e)}")
        return 'HOLD', 50

def execute_trade(action, current_price, confidence, trade_size=1000.0):
    """Execute a virtual trade"""
    global trading_state
    
    try:
        if action == 'HOLD' or trading_state['current_trade'] is not None:
            return None
        
        trade = {
            'action': action,
            'entry_price': current_price,
            'entry_time': datetime.now(),
            'size': trade_size,
            'confidence': confidence,
            'target_profit_pct': 0.0005,  # 0.05% target
            'stop_loss_pct': 0.0003,     # 0.03% stop loss
            'status': 'OPEN'
        }
        
        trading_state['current_trade'] = trade
        trading_state['total_trades'] += 1
        
        logger.info(f"Executed {action} trade at {current_price:.6f} "
                   f"with {confidence:.1f}% confidence")
        
        return trade
        
    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        return None

def check_trade_result(current_price):
    """Check if current trade should be closed"""
    global trading_state
    
    try:
        if trading_state['current_trade'] is None:
            return None
        
        trade = trading_state['current_trade']
        entry_price = trade['entry_price']
        price_change_pct = (current_price / entry_price - 1) * 100
        
        if trade['action'] == 'BUY':
            profit_pct = price_change_pct
        else:  # SELL
            profit_pct = -price_change_pct
        
        # Check profit target
        if profit_pct >= trade['target_profit_pct'] * 100:
            close_trade(current_price, 'PROFIT', profit_pct)
            return 'PROFIT'
        
        # Check stop loss
        elif profit_pct <= -trade['stop_loss_pct'] * 100:
            close_trade(current_price, 'LOSS', profit_pct)
            return 'LOSS'
        
        # Check time-based exit (after 2 minutes)
        trade_duration = (datetime.now() - trade['entry_time']).total_seconds()
        if trade_duration >= 120:  # 2 minutes
            close_trade(current_price, 'TIMEOUT', profit_pct)
            return 'TIMEOUT'
        
        return None
        
    except Exception as e:
        logger.error(f"Error checking trade: {str(e)}")
        return None

def close_trade(exit_price, result, profit_pct):
    """Close the current trade and update statistics"""
    global trading_state
    
    try:
        if trading_state['current_trade'] is None:
            return
        
        trade = trading_state['current_trade']
        trade_size = trade['size']
        
        # Calculate profit in base currency
        if result == 'PROFIT':
            profit_amount = trade_size * (abs(profit_pct) / 100)
            trading_state['profitable_trades'] += 1
            trading_state['total_profit'] += profit_amount
            trading_state['balance'] += profit_amount
            logger.info(f"💰 Trade PROFIT: +{profit_pct:.4f}% (+${profit_amount:.2f})")
        else:
            loss_amount = trade_size * (abs(profit_pct) / 100)
            trading_state['balance'] -= loss_amount
            logger.info(f"📉 Trade {result}: {profit_pct:.4f}% (-${loss_amount:.2f})")
        
        # Update win rate
        if trading_state['total_trades'] > 0:
            trading_state['win_rate'] = (
                trading_state['profitable_trades'] / trading_state['total_trades'] * 100
            )
        
        # Log trade details
        trade['exit_price'] = exit_price
        trade['exit_time'] = datetime.now()
        trade['result'] = result
        trade['profit_pct'] = profit_pct
        trade['status'] = 'CLOSED'
        
        # Save trade log
        log_trade(trade)
        
        # Clear current trade
        trading_state['current_trade'] = None
        
    except Exception as e:
        logger.error(f"Error closing trade: {str(e)}")

def log_trade(trade):
    """Log trade details to file"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': trade['action'],
            'entry_price': trade['entry_price'],
            'exit_price': trade.get('exit_price', 0),
            'result': trade.get('result', ''),
            'profit_pct': trade.get('profit_pct', 0),
            'confidence': trade.get('confidence', 0),
            'duration_seconds': (
                trade['exit_time'] - trade['entry_time']
            ).total_seconds() if 'exit_time' in trade else 0
        }
        
        trade_log_file = os.path.join(log_dir, 'trades.json')
        trades = []
        
        if os.path.exists(trade_log_file):
            with open(trade_log_file, 'r') as f:
                try:
                    trades = json.load(f)
                except:
                    trades = []
        
        trades.append(log_entry)
        
        # Keep only last 1000 trades
        if len(trades) > 1000:
            trades = trades[-1000:]
        
        with open(trade_log_file, 'w') as f:
            json.dump(trades, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error logging trade: {str(e)}")

def create_chart_data(df):
    """Create interactive price chart"""
    try:
        # Use last 30 data points for chart
        chart_len = min(30, len(df))
        chart_df = df.iloc[-chart_len:] if chart_len > 0 else df
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df['open'],
            high=chart_df['high'],
            low=chart_df['low'],
            close=chart_df['close'],
            name='EUR/GBP',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ))
        
        # Add key indicators if available
        if 'SMA_20' in chart_df.columns:
            fig.add_trace(go.Scatter(
                x=chart_df.index,
                y=chart_df['SMA_20'],
                line=dict(color='orange', width=1.5, dash='dash'),
                name='SMA 20'
            ))
        
        # Add prediction markers if we have a trade
        if trading_state['current_trade']:
            trade = trading_state['current_trade']
            fig.add_trace(go.Scatter(
                x=[chart_df.index[-1]],
                y=[trade['entry_price']],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color='green' if trade['action'] == 'BUY' else 'red',
                    symbol='triangle-up' if trade['action'] == 'BUY' else 'triangle-down'
                ),
                text=[f"{trade['action']} @ {trade['entry_price']:.5f}"],
                textposition="top center",
                name=f"Current {trade['action']}"
            ))
        
        # Update layout
        fig.update_layout(
            title=f'EUR/GBP Price Chart - Next Prediction in {trading_state.get("next_prediction_time", 0):.0f}s',
            yaxis_title='Price',
            xaxis_title='Time',
            template='plotly_dark',
            height=500,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        return None

def trading_cycle():
    """Main trading cycle - runs every 2 minutes"""
    global trading_state
    
    while True:
        try:
            cycle_start = datetime.now()
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting trading cycle at {cycle_start}")
            logger.info(f"{'='*60}")
            
            # Fetch data
            df, is_demo = fetch_forex_data(TRADING_SYMBOL, interval='1min', outputsize='compact')
            
            if df.empty:
                logger.warning("No data received, waiting for next cycle")
                time.sleep(60)
                continue
            
            # Calculate indicators
            df_with_indicators = calculate_advanced_indicators(df.copy())
            
            # Get current price
            current_price = df['close'].iloc[-1] if len(df) > 0 else 0.8575
            
            # Make 2-minute prediction
            prediction_prob, confidence, direction = make_2min_prediction(df_with_indicators, current_price)
            
            # Get current indicators
            current_indicators = {}
            if len(df_with_indicators) > 0:
                for col in ['RSI_14', 'MACD_fast', 'Stoch_K', 'ADX', 'ATR_14', 'Williams_R']:
                    if col in df_with_indicators.columns:
                        current_indicators[col] = float(df_with_indicators[col].iloc[-1])
            
            # Determine trading action
            action, action_confidence = determine_trading_action(
                prediction_prob, confidence, current_indicators, current_price
            )
            
            # Check existing trade
            trade_result = check_trade_result(current_price)
            
            # Execute new trade if no existing trade
            if trading_state['current_trade'] is None and action != 'HOLD':
                execute_trade(action, current_price, action_confidence)
            
            # Create chart
            chart_data = create_chart_data(df)
            
            # Update trading state
            next_prediction_time = 120 - (datetime.now() - cycle_start).seconds
            next_prediction_time = max(0, next_prediction_time)
            
            trading_state.update({
                'current_price': round(float(current_price), 6),
                'prediction': direction,
                'action': action,
                'confidence': round(float(action_confidence), 1),
                'next_prediction_time': next_prediction_time,
                'indicators': current_indicators,
                'is_demo_data': is_demo,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'chart_data': chart_data
            })
            
            # Log cycle summary
            logger.info(f"Cycle Summary:")
            logger.info(f"  Price: {current_price:.6f}")
            logger.info(f"  Prediction: {direction} ({prediction_prob:.2%})")
            logger.info(f"  Action: {action} ({action_confidence:.1f}% confidence)")
            logger.info(f"  Balance: ${trading_state['balance']:.2f}")
            logger.info(f"  Win Rate: {trading_state['win_rate']:.1f}%")
            logger.info(f"  Next prediction in: {next_prediction_time}s")
            
            if trading_state['current_trade']:
                trade = trading_state['current_trade']
                current_pnl = (current_price / trade['entry_price'] - 1) * 100
                if trade['action'] == 'SELL':
                    current_pnl = -current_pnl
                logger.info(f"  Active Trade: {trade['action']} @ {trade['entry_price']:.6f}")
                logger.info(f"  Current P&L: {current_pnl:.4f}%")
            
            # Wait for next cycle
            cycle_duration = (datetime.now() - cycle_start).seconds
            wait_time = max(1, 120 - cycle_duration)
            logger.info(f"Waiting {wait_time}s for next cycle...")
            time.sleep(wait_time)
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
            import traceback
            traceback.print_exc()
            time.sleep(60)

def start_trading_bot():
    """Start the trading bot in a separate thread"""
    try:
        thread = threading.Thread(target=trading_cycle, daemon=True)
        thread.start()
        logger.info("Trading bot started successfully")
        print("✅ Trading bot started successfully")
    except Exception as e:
        logger.error(f"Error starting trading bot: {e}")
        print(f"❌ Error starting trading bot: {e}")

# Flask Routes
@app.route('/')
def index():
    """Render main trading dashboard"""
    return render_template('index.html')

@app.route('/api/trading_state')
def get_trading_state():
    """Get current trading state"""
    return jsonify(trading_state)

@app.route('/api/trade_history')
def get_trade_history():
    """Get recent trade history"""
    try:
        trade_log_file = os.path.join(log_dir, 'trades.json')
        if os.path.exists(trade_log_file):
            with open(trade_log_file, 'r') as f:
                trades = json.load(f)
            return jsonify(trades[-50:])  # Last 50 trades
        return jsonify([])
    except Exception as e:
        logger.error(f"Error getting trade history: {str(e)}")
        return jsonify([])

@app.route('/api/execute_manual/<action>')
def execute_manual(action):
    """Execute manual trade (for testing)"""
    if action.upper() in ['BUY', 'SELL']:
        if trading_state['current_trade'] is None:
            execute_trade(action.upper(), trading_state['current_price'], 100)
            return jsonify({'success': True, 'action': action.upper()})
        return jsonify({'success': False, 'error': 'Trade already in progress'})
    return jsonify({'success': False, 'error': 'Invalid action'})

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'balance': trading_state['balance'],
        'total_trades': trading_state['total_trades'],
        'win_rate': trading_state['win_rate']
    })

@app.route('/api/reset')
def reset_trading():
    """Reset trading state (for testing)"""
    global trading_state
    trading_state.update({
        'balance': INITIAL_BALANCE,
        'total_trades': 0,
        'profitable_trades': 0,
        'total_profit': 0.0,
        'win_rate': 0.0,
        'current_trade': None
    })
    return jsonify({'success': True, 'message': 'Trading reset'})

if __name__ == '__main__':
    # Initialize logging
    print("=" * 60)
    print("EUR/GBP 2-Minute Profit Predictor")
    print("=" * 60)
    print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"Trading Symbol: {TRADING_SYMBOL}")
    print(f"Prediction Window: {PREDICTION_WINDOW_MINUTES} minutes")
    print(f"API Keys Available: {len(API_KEYS)}")
    
    # Start trading bot
    start_trading_bot()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    print(f"Web server starting on port {port}")
    print("Dashboard available at: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)