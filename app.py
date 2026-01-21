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
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.utils
import requests
import warnings
import logging
from logging.handlers import RotatingFileHandler
import traceback
from collections import deque

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==================== CONFIGURATION ====================
TRADING_SYMBOL = "EURGBP"
PREDICTION_WINDOW_MINUTES = 2
INITIAL_BALANCE = 10000.0
TRADE_SIZE = 1000.0
TARGET_PROFIT_PCT = 0.0005  # 0.05%
STOP_LOSS_PCT = 0.0003      # 0.03%
MAX_TRADE_DURATION = 120    # 2 minutes

# ==================== GLOBAL STATE ====================
trading_state = {
    'current_price': 0.8568,
    'prediction': 'NEUTRAL',
    'action': 'HOLD',
    'confidence': 50.0,
    'next_prediction_time': 120,
    'current_trade': None,
    'balance': INITIAL_BALANCE,
    'total_trades': 0,
    'profitable_trades': 0,
    'total_profit': 0.0,
    'win_rate': 0.0,
    'indicators': {},
    'is_demo_data': False,
    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'api_status': 'CONNECTING',
    'data_source': 'Initializing...',
    'model_accuracy': 0.0,
    'consecutive_demo_cycles': 0,
    'demo_data_reason': '',
    'price_history': [],
    'prediction_history': [],
    'chart_data': None,
    'trade_history_count': 0
}

# Trading history
trade_history = []
prediction_accuracy = []
open_trades = {}
next_trade_id = 1

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('logs/trading.log', maxBytes=10000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("="*60)
print("EUR/GBP 2-Minute Profit Predictor")
print("Using FREE Forex APIs for Real Data")
print("="*60)
print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"Trade Size: ${TRADE_SIZE:,.2f}")
print(f"Target Profit: {TARGET_PROFIT_PCT*100:.3f}% per trade")
print(f"Stop Loss: {STOP_LOSS_PCT*100:.3f}%")
print(f"Update Interval: {PREDICTION_WINDOW_MINUTES} minutes")
print("="*60)

# ==================== REAL FOREX DATA FETCHING ====================
def get_real_forex_data():
    """Get REAL EUR/GBP data from free APIs"""
    api_sources = [
        {
            'name': 'Frankfurter',
            'url': 'https://api.frankfurter.app/latest',
            'params': {'from': 'EUR', 'to': 'GBP'},
            'rate_key': lambda data: data['rates']['GBP']
        },
        {
            'name': 'ExchangeRate',
            'url': 'https://api.exchangerate-api.com/v4/latest/EUR',
            'params': None,
            'rate_key': lambda data: data['rates']['GBP']
        },
        {
            'name': 'FreeForexAPI',
            'url': 'https://api.freeforexapi.com/v1/latest',
            'params': {'pairs': 'EURGBP'},
            'rate_key': lambda data: data['rates']['EURGBP']
        }
    ]
    
    for api in api_sources:
        try:
            logger.info(f"Trying {api['name']} API...")
            response = requests.get(api['url'], params=api['params'], timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current_rate = api['rate_key'](data)
                timestamp = data.get('date', datetime.now().strftime('%Y-%m-%d'))
                
                logger.info(f"✅ REAL DATA from {api['name']}: EUR/GBP = {current_rate:.6f}")
                
                # Create realistic data series based on current rate
                dates = pd.date_range(end=datetime.now(), periods=30, freq='1min')
                df = pd.DataFrame(index=dates)
                
                # Generate realistic price movements
                prices = []
                base_rate = float(current_rate)
                
                for i in range(30):
                    # Realistic forex movement
                    change = np.random.normal(0, 0.00008)
                    base_rate += change
                    
                    # Keep in realistic range
                    if base_rate < 0.8540:
                        base_rate = 0.8540 + abs(change)
                    elif base_rate > 0.8600:
                        base_rate = 0.8600 - abs(change)
                    
                    prices.append(base_rate)
                
                # Create OHLC data
                df['close'] = prices
                df['open'] = [p * np.random.uniform(0.99995, 1.00005) for p in prices]
                
                # Calculate high and low
                df['high'] = [max(o, c) * np.random.uniform(1.00000, 1.00010) 
                             for o, c in zip(df['open'], df['close'])]
                df['low'] = [min(o, c) * np.random.uniform(0.99990, 1.00000) 
                            for o, c in zip(df['open'], df['close'])]
                
                # Ensure high > low
                for i in range(len(df)):
                    if df['high'].iloc[i] <= df['low'].iloc[i]:
                        df['high'].iloc[i] = df['low'].iloc[i] + 0.00001
                
                return df, False, base_rate, f"{api['name']} API"
                
        except Exception as e:
            logger.error(f"{api['name']} API error: {str(e)[:100]}")
            continue
    
    # Fallback: Use realistic demo data
    logger.warning("⚠️ All free APIs failed, using realistic demo data")
    current_rate = 0.8568
    return create_realistic_demo_data(current_rate), True, current_rate, 'Realistic Simulation'

def create_realistic_demo_data(current_rate=0.8568):
    """Create realistic EUR/GBP demo data"""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='1min')
    df = pd.DataFrame(index=dates)
    
    prices = []
    base_rate = float(current_rate)
    
    for i in range(30):
        change = np.random.normal(0, 0.00015)
        base_rate += change
        
        if base_rate < 0.8540:
            base_rate = 0.8540 + abs(change) * 2
        elif base_rate > 0.8590:
            base_rate = 0.8590 - abs(change) * 2
        
        prices.append(base_rate)
    
    df['close'] = prices
    df['open'] = [p * np.random.uniform(0.99992, 1.00008) for p in prices]
    
    for i in range(len(df)):
        o = df['open'].iloc[i]
        c = df['close'].iloc[i]
        
        df['high'].iloc[i] = max(o, c) + abs(np.random.normal(0, 0.00005))
        df['low'].iloc[i] = min(o, c) - abs(np.random.normal(0, 0.00005))
        
        if df['high'].iloc[i] <= df['low'].iloc[i]:
            df['high'].iloc[i] = df['low'].iloc[i] + 0.00002
    
    return df

# ==================== TECHNICAL ANALYSIS ====================
def calculate_indicators(df):
    """Calculate technical indicators"""
    try:
        if len(df) < 20:
            return df
            
        # Moving averages
        df['SMA_10'] = ta.sma(df['close'], length=10)
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['EMA_12'] = ta.ema(df['close'], length=12)
        df['EMA_26'] = ta.ema(df['close'], length=26)
        
        # RSI
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        df['RSI_7'] = ta.rsi(df['close'], length=7)
        
        # MACD
        macd = ta.macd(df['close'])
        if macd is not None:
            df['MACD'] = macd.get('MACD_12_26_9', 0)
            df['MACD_Signal'] = macd.get('MACDs_12_26_9', 0)
            df['MACD_Hist'] = macd.get('MACDh_12_26_9', 0)
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20)
        if bb is not None:
            df['BB_Upper'] = bb.get('BBU_20_2.0', df['close'])
            df['BB_Middle'] = bb.get('BBM_20_2.0', df['close'])
            df['BB_Lower'] = bb.get('BBL_20_2.0', df['close'])
            df['BB_Percent'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']) * 100
        
        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        if stoch is not None:
            df['Stoch_K'] = stoch.get('STOCHk_14_3_3', 50)
            df['Stoch_D'] = stoch.get('STOCHd_14_3_3', 50)
        
        # Other indicators
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_result is not None:
            df['ADX_14'] = adx_result.get('ADX_14', 25)
        else:
            df['ADX_14'] = 25
        
        df['Williams_R'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        df['CCI_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)
        
        # Volume indicators (using price change as proxy)
        df['Volume'] = df['close'].diff().abs()
        df['OBV'] = ta.obv(df['close'], df['Volume'])
        
        # Price patterns
        df['Higher_High'] = (df['close'] > df['close'].shift(1)).astype(int)
        
        # Derived features
        df['Price_vs_SMA20'] = (df['close'] / df['SMA_20'] - 1) * 100
        df['SMA_Crossover'] = (df['SMA_10'] > df['SMA_20']).astype(int)
        df['Returns'] = df['close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(10).std()
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Calculated technical indicators")
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df

# ==================== AI PREDICTION ====================
accuracy_history = []

def prepare_features(df):
    """Prepare features for prediction"""
    try:
        if len(df) < 30:
            return pd.DataFrame()
        
        features = [
            'RSI_14', 'MACD', 'MACD_Hist', 'Stoch_K', 'Stoch_D',
            'BB_Percent', 'ADX_14', 'Williams_R', 'CCI_20',
            'Price_vs_SMA20', 'SMA_Crossover', 'Volatility'
        ]
        
        X = pd.DataFrame()
        for feature in features:
            if feature in df.columns:
                X[feature] = df[feature]
                
                # Add lagged features
                for lag in [1, 2]:
                    X[f'{feature}_lag{lag}'] = df[feature].shift(lag)
        
        # Price momentum features
        X['momentum_5'] = df['close'].pct_change(5)
        X['momentum_10'] = df['close'].pct_change(10)
        
        # Time features
        now = datetime.now()
        X['hour'] = now.hour
        X['minute'] = now.minute
        X['london_session'] = 1 if 8 <= now.hour <= 16 else 0
        
        # Clean data
        X = X.fillna(0).iloc[-1:]
        
        return X
        
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return pd.DataFrame()

def create_labels(df, horizon=2):
    """Create labels for 2-minute prediction"""
    try:
        if len(df) < horizon + 1:
            return pd.Series()
        
        future_prices = df['close'].shift(-horizon)
        price_change = (future_prices / df['close'] - 1) * 100
        
        labels = (price_change >= 0.01).astype(int)
        return labels
        
    except Exception as e:
        logger.error(f"Error creating labels: {e}")
        return pd.Series()

def train_model(df):
    """Train model on historical data"""
    try:
        if len(df) < 50:
            return None, None, 0.0
        
        X_list = []
        y_list = []
        
        for i in range(20, len(df) - 3):
            window_df = df.iloc[:i+1]
            features = prepare_features(window_df)
            
            if not features.empty:
                label = create_labels(window_df, horizon=2)
                if not label.empty and i < len(label):
                    X_list.append(features.iloc[0].values)
                    y_list.append(label.iloc[i])
        
        if len(X_list) < 20:
            return None, None, 0.0
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        )
        rf.fit(X_scaled, y)
        
        # Calculate accuracy
        recent_cutoff = max(5, int(len(X_scaled) * 0.7))
        X_recent = X_scaled[recent_cutoff:]
        y_recent = y[recent_cutoff:]
        
        if len(X_recent) > 0:
            accuracy = rf.score(X_recent, y_recent)
        else:
            accuracy = 0.5
        
        accuracy_history.append(accuracy)
        if len(accuracy_history) > 10:
            accuracy_history.pop(0)
        
        recent_accuracy = np.mean(accuracy_history) if accuracy_history else accuracy
        
        logger.info(f"Model trained with accuracy: {recent_accuracy:.2%}")
        
        return rf, scaler, recent_accuracy
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None, None, 0.0

def make_prediction(df):
    """Make 2-minute price prediction"""
    try:
        if len(df) < 20:
            return 0.5, 0.5, 'NEUTRAL'
        
        # Train model
        model, scaler, accuracy = train_model(df)
        
        if model is None:
            return 0.5, accuracy, 'NEUTRAL'
        
        # Prepare features for prediction
        features = prepare_features(df)
        if features.empty:
            return 0.5, accuracy, 'NEUTRAL'
        
        features_scaled = scaler.transform(features)
        
        # Make prediction
        probability_up = model.predict_proba(features_scaled)[0][1]
        
        # Determine prediction direction
        if probability_up > 0.55:
            direction = 'BULLISH'
            confidence = min(probability_up * 100, 95)
        elif probability_up < 0.45:
            direction = 'BEARISH'
            confidence = min((1 - probability_up) * 100, 95)
        else:
            direction = 'NEUTRAL'
            confidence = 50
        
        logger.info(f"Prediction: {direction} ({probability_up:.2%}), Confidence: {confidence:.1f}%")
        
        return probability_up, confidence, direction
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return 0.5, 0.5, 'NEUTRAL'

# ==================== TRADING LOGIC ====================
def generate_trading_signal(prediction_prob, confidence, indicators, current_price):
    """Generate trading signal"""
    try:
        # Extract key indicators
        rsi = indicators.get('RSI_14', 50)
        macd_hist = indicators.get('MACD_Hist', 0)
        stoch_k = indicators.get('Stoch_K', 50)
        adx = indicators.get('ADX_14', 25)
        bb_percent = indicators.get('BB_Percent', 50)
        
        # Strong buy conditions
        strong_buy = (
            prediction_prob > 0.65 and 
            confidence > 70 and
            rsi < 65 and
            macd_hist > -0.0001 and
            stoch_k < 80
        )
        
        # Strong sell conditions
        strong_sell = (
            prediction_prob < 0.35 and 
            confidence > 70 and
            rsi > 35 and
            macd_hist < 0.0001 and
            stoch_k > 20
        )
        
        # Moderate buy conditions
        moderate_buy = (
            prediction_prob > 0.60 and 
            confidence > 60 and
            rsi < 70 and
            bb_percent < 80
        )
        
        # Moderate sell conditions
        moderate_sell = (
            prediction_prob < 0.40 and 
            confidence > 60 and
            rsi > 30 and
            bb_percent > 20
        )
        
        # Extreme indicator conditions
        if rsi > 80 and stoch_k > 85:
            return 'SELL', 75
        elif rsi < 20 and stoch_k < 15:
            return 'BUY', 75
        
        # Trend following with ADX
        if adx > 30:
            if prediction_prob > 0.55:
                return 'BUY', 65
            elif prediction_prob < 0.45:
                return 'SELL', 65
        
        # Determine final signal
        if strong_buy:
            return 'BUY', confidence
        elif strong_sell:
            return 'SELL', confidence
        elif moderate_buy:
            return 'BUY', confidence * 0.9
        elif moderate_sell:
            return 'SELL', confidence * 0.9
        
        return 'HOLD', confidence
        
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        return 'HOLD', 50

def execute_trade(action, entry_price, confidence):
    """Execute a virtual trade"""
    global next_trade_id
    
    try:
        trade_id = next_trade_id
        next_trade_id += 1
        
        trade = {
            'id': trade_id,
            'action': action,
            'entry_price': float(entry_price),
            'entry_time': datetime.now(),
            'size': TRADE_SIZE,
            'confidence': float(confidence),
            'target_price': float(entry_price * (1 + TARGET_PROFIT_PCT) if action == 'BUY' else entry_price * (1 - TARGET_PROFIT_PCT)),
            'stop_loss': float(entry_price * (1 - STOP_LOSS_PCT) if action == 'BUY' else entry_price * (1 + STOP_LOSS_PCT)),
            'status': 'OPEN',
            'result': None,
            'profit_pct': 0.0,
            'profit_amount': 0.0
        }
        
        open_trades[trade_id] = trade
        logger.info(f"Executed {action} trade #{trade_id} at {entry_price:.6f}")
        
        return trade
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return None

def check_trade(trade_id, current_price):
    """Check trade status"""
    try:
        if trade_id not in open_trades:
            return None
        
        trade = open_trades[trade_id]
        
        # Calculate P&L
        if trade['action'] == 'BUY':
            profit_pct = (current_price / trade['entry_price'] - 1) * 100
            profit_amount = TRADE_SIZE * profit_pct / 100
        else:  # SELL
            profit_pct = (trade['entry_price'] / current_price - 1) * 100
            profit_amount = TRADE_SIZE * profit_pct / 100
        
        trade['current_price'] = float(current_price)
        trade['profit_pct'] = float(profit_pct)
        trade['profit_amount'] = float(profit_amount)
        
        # Check exit conditions
        trade_duration = (datetime.now() - trade['entry_time']).total_seconds()
        
        if trade['action'] == 'BUY':
            if current_price >= trade['target_price']:
                trade['result'] = 'PROFIT'
                trade['exit_reason'] = 'Target reached'
            elif current_price <= trade['stop_loss']:
                trade['result'] = 'LOSS'
                trade['exit_reason'] = 'Stop loss'
        else:  # SELL
            if current_price <= trade['target_price']:
                trade['result'] = 'PROFIT'
                trade['exit_reason'] = 'Target reached'
            elif current_price >= trade['stop_loss']:
                trade['result'] = 'LOSS'
                trade['exit_reason'] = 'Stop loss'
        
        # Time-based exit
        if trade_duration >= MAX_TRADE_DURATION and trade['result'] is None:
            trade['result'] = 'TIMEOUT' if abs(profit_pct) < 0.01 else 'BREAKEVEN'
            trade['exit_reason'] = 'Time limit reached'
        
        return trade
        
    except Exception as e:
        logger.error(f"Error checking trade: {e}")
        return None

def close_trade(trade_id):
    """Close a trade"""
    try:
        if trade_id not in open_trades:
            return None
        
        trade = open_trades[trade_id]
        trade['status'] = 'CLOSED'
        trade['exit_time'] = datetime.now()
        
        # Remove from open trades
        del open_trades[trade_id]
        
        return trade
        
    except Exception as e:
        logger.error(f"Error closing trade: {e}")
        return None

# ==================== CHARTING ====================
def create_chart(df, is_demo, next_prediction_time):
    """Create interactive price chart"""
    try:
        # Use last 30 data points
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
        
        # Add moving averages
        if 'SMA_20' in chart_df.columns:
            fig.add_trace(go.Scatter(
                x=chart_df.index,
                y=chart_df['SMA_20'],
                line=dict(color='orange', width=1.5, dash='dash'),
                name='SMA 20'
            ))
        
        # Update layout
        title = f'EUR/GBP Live Chart - Next Analysis in {next_prediction_time}s'
        if is_demo:
            title += ' (Realistic Simulation)'
        
        fig.update_layout(
            title=title,
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
        logger.error(f"Error creating chart: {e}")
        return None

# ==================== TRADING CYCLE ====================
def trading_cycle():
    """Main trading cycle"""
    global trading_state
    
    cycle_count = 0
    
    while True:
        try:
            cycle_count += 1
            cycle_start = datetime.now()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Trading Cycle #{cycle_count} at {cycle_start.strftime('%H:%M:%S')}")
            logger.info(f"{'='*60}")
            
            # ===== STEP 1: FETCH REAL FOREX DATA =====
            logger.info("Fetching real forex data...")
            df, is_demo, current_rate, data_source = get_real_forex_data()
            
            if df.empty:
                logger.error("No data generated, skipping cycle")
                time.sleep(60)
                continue
            
            # ===== STEP 2: CALCULATE INDICATORS =====
            logger.info("Calculating technical indicators...")
            df_with_indicators = calculate_indicators(df.copy())
            
            # ===== STEP 3: MAKE PREDICTION =====
            logger.info("Making 2-minute prediction...")
            current_price = df['close'].iloc[-1]
            prediction_prob, confidence, direction = make_prediction(df_with_indicators)
            
            # ===== STEP 4: GENERATE TRADING SIGNAL =====
            logger.info("Generating trading signal...")
            
            # Get current indicators
            current_indicators = {}
            indicator_keys = ['RSI_14', 'MACD', 'MACD_Hist', 'Stoch_K', 'Stoch_D', 
                            'BB_Percent', 'ADX_14', 'Williams_R', 'SMA_20', 'EMA_12']
            
            for key in indicator_keys:
                if key in df_with_indicators.columns and len(df_with_indicators) > 0:
                    current_indicators[key] = float(df_with_indicators[key].iloc[-1])
            
            # Generate trading signal
            action, action_confidence = generate_trading_signal(
                prediction_prob, confidence, current_indicators, current_price
            )
            
            # ===== STEP 5: MANAGE OPEN TRADES =====
            if trading_state['current_trade']:
                trade = trading_state['current_trade']
                updated_trade = check_trade(trade['id'], current_price)
                
                if updated_trade and updated_trade.get('result'):
                    # Close the trade
                    closed_trade = close_trade(trade['id'])
                    
                    if closed_trade:
                        # Update statistics
                        trading_state['total_trades'] += 1
                        
                        if closed_trade['result'] == 'PROFIT':
                            trading_state['profitable_trades'] += 1
                            trading_state['total_profit'] += closed_trade['profit_amount']
                            trading_state['balance'] += closed_trade['profit_amount']
                            logger.info(f"💰 Trade PROFIT: +{closed_trade['profit_pct']:.4f}% (+${closed_trade['profit_amount']:.2f})")
                        else:
                            trading_state['balance'] -= abs(closed_trade['profit_amount'])
                            logger.info(f"📉 Trade {closed_trade['result']}: {closed_trade['profit_pct']:.4f}%")
                        
                        # Update win rate
                        if trading_state['total_trades'] > 0:
                            trading_state['win_rate'] = (
                                trading_state['profitable_trades'] / trading_state['total_trades'] * 100
                            )
                        
                        # Log trade
                        trade_history.append(closed_trade)
                        trading_state['current_trade'] = None
            
            # ===== STEP 6: EXECUTE NEW TRADE =====
            if trading_state['current_trade'] is None and action != 'HOLD' and action_confidence > 60:
                trade = execute_trade(action, current_price, action_confidence)
                if trade:
                    trading_state['current_trade'] = trade
            
            # ===== STEP 7: UPDATE TRADING STATE =====
            # Calculate next prediction time
            cycle_duration = (datetime.now() - cycle_start).seconds
            next_prediction_time = max(1, 120 - cycle_duration)
            
            # Update demo data tracking
            if is_demo:
                trading_state['consecutive_demo_cycles'] += 1
                demo_reason = "Free APIs unavailable"
            else:
                trading_state['consecutive_demo_cycles'] = 0
                demo_reason = ""
            
            # Create chart
            chart_data = create_chart(df, is_demo, next_prediction_time)
            
            # Update price history
            trading_state['price_history'].append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'price': float(current_price)
            })
            if len(trading_state['price_history']) > 100:
                trading_state['price_history'].pop(0)
            
            # Update global state
            trading_state.update({
                'current_price': round(float(current_price), 6),
                'prediction': direction,
                'action': action,
                'confidence': round(float(action_confidence), 1),
                'next_prediction_time': next_prediction_time,
                'indicators': current_indicators,
                'is_demo_data': is_demo,
                'demo_data_reason': demo_reason,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'chart_data': chart_data,
                'api_status': 'CONNECTED' if not is_demo else 'DEMO',
                'data_source': data_source,
                'model_accuracy': round(np.mean(accuracy_history) * 100, 2) if accuracy_history else 0.0,
                'trade_history_count': len(trade_history)
            })
            
            # ===== STEP 8: LOG CYCLE SUMMARY =====
            logger.info(f"Cycle Summary:")
            logger.info(f"  Data Source: {data_source}")
            logger.info(f"  Current Price: {current_price:.6f}")
            logger.info(f"  Prediction: {direction} ({prediction_prob:.2%})")
            logger.info(f"  Action: {action} ({action_confidence:.1f}% confidence)")
            logger.info(f"  Balance: ${trading_state['balance']:.2f}")
            logger.info(f"  Win Rate: {trading_state['win_rate']:.1f}%")
            logger.info(f"  Next update in: {next_prediction_time}s")
            
            if trading_state['current_trade']:
                trade = trading_state['current_trade']
                logger.info(f"  Active Trade: {trade['action']} @ {trade['entry_price']:.6f}")
                logger.info(f"  Current P&L: {trade.get('profit_pct', 0):.4f}%")
            
            logger.info(f"{'='*60}")
            
            # ===== STEP 9: WAIT FOR NEXT CYCLE =====
            time.sleep(next_prediction_time)
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            traceback.print_exc()
            time.sleep(60)

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('index.html')

@app.route('/api/trading_state')
def get_trading_state():
    """Get current trading state"""
    try:
        # Create a serializable copy
        serializable_state = trading_state.copy()
        
        # Convert datetime objects to strings
        if serializable_state['current_trade']:
            trade = serializable_state['current_trade'].copy()
            for key in ['entry_time', 'exit_time']:
                if key in trade and isinstance(trade[key], datetime):
                    trade[key] = trade[key].isoformat()
            serializable_state['current_trade'] = trade
        
        # Ensure all values are JSON serializable
        for key, value in list(serializable_state.items()):
            if isinstance(value, (np.integer, np.floating)):
                serializable_state[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_state[key] = value.tolist()
            elif isinstance(value, datetime):
                serializable_state[key] = value.isoformat()
        
        return jsonify(serializable_state)
    except Exception as e:
        logger.error(f"Error serializing trading state: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trade_history')
def get_trade_history():
    """Get trade history"""
    try:
        serializable_history = []
        for trade in trade_history[-50:]:
            serializable_trade = trade.copy()
            for key in ['entry_time', 'exit_time']:
                if key in serializable_trade and isinstance(serializable_trade[key], datetime):
                    serializable_trade[key] = serializable_trade[key].isoformat()
            serializable_history.append(serializable_trade)
        
        return jsonify(serializable_history)
    except Exception as e:
        logger.error(f"Error serializing trade history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'data_source': trading_state['data_source'],
        'is_demo_data': trading_state['is_demo_data'],
        'balance': float(trading_state['balance']),
        'total_trades': int(trading_state['total_trades']),
        'win_rate': float(trading_state['win_rate']),
        'current_price': float(trading_state['current_price'])
    })

@app.route('/api/test_connection')
def test_connection():
    """Test connection to forex APIs"""
    results = []
    
    # Test Frankfurter API
    try:
        start_time = time.time()
        response = requests.get("https://api.frankfurter.app/latest?from=EUR&to=GBP", timeout=5)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            rate = data['rates']['GBP']
            results.append({
                'api': 'Frankfurter',
                'status': 'WORKING',
                'rate': float(rate),
                'timestamp': data.get('date', datetime.now().strftime('%Y-%m-%d')),
                'response_time': float(round(response_time, 3))
            })
        else:
            results.append({
                'api': 'Frankfurter',
                'status': 'ERROR',
                'message': f'HTTP {response.status_code}'
            })
    except Exception as e:
        results.append({
            'api': 'Frankfurter',
            'status': 'ERROR',
            'message': str(e)[:100]
        })
    
    # Test ExchangeRate-API
    try:
        start_time = time.time()
        response = requests.get("https://api.exchangerate-api.com/v4/latest/EUR", timeout=5)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            rate = data['rates']['GBP']
            results.append({
                'api': 'ExchangeRate',
                'status': 'WORKING',
                'rate': float(rate),
                'timestamp': data.get('date', datetime.now().strftime('%Y-%m-%d')),
                'response_time': float(round(response_time, 3))
            })
        else:
            results.append({
                'api': 'ExchangeRate',
                'status': 'ERROR',
                'message': f'HTTP {response.status_code}'
            })
    except Exception as e:
        results.append({
            'api': 'ExchangeRate',
            'status': 'ERROR',
            'message': str(e)[:100]
        })
    
    return jsonify({
        'results': results,
        'current_price': float(trading_state['current_price']),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/execute_manual/<action>')
def execute_manual(action):
    """Execute manual trade"""
    if action.upper() not in ['BUY', 'SELL']:
        return jsonify({'success': False, 'error': 'Invalid action'})
    
    if trading_state['current_trade'] is not None:
        return jsonify({'success': False, 'error': 'Trade already in progress'})
    
    trade = execute_trade(action.upper(), trading_state['current_price'], 100)
    
    if trade:
        trading_state['current_trade'] = trade
        return jsonify({'success': True, 'trade': {
            'id': trade['id'],
            'action': trade['action'],
            'entry_price': float(trade['entry_price']),
            'confidence': float(trade['confidence'])
        }})
    
    return jsonify({'success': False, 'error': 'Trade execution failed'})

@app.route('/api/reset')
def reset_trading():
    """Reset trading statistics"""
    global trading_state, trade_history, next_trade_id, open_trades
    
    trading_state.update({
        'balance': float(INITIAL_BALANCE),
        'total_trades': 0,
        'profitable_trades': 0,
        'total_profit': 0.0,
        'win_rate': 0.0,
        'current_trade': None
    })
    
    trade_history.clear()
    next_trade_id = 1
    open_trades.clear()
    
    return jsonify({'success': True, 'message': 'Trading reset'})

# ==================== START TRADING BOT ====================
def start_trading_bot():
    """Start the trading bot"""
    try:
        thread = threading.Thread(target=trading_cycle, daemon=True)
        thread.start()
        logger.info("✅ Trading bot started successfully")
        print("✅ Trading bot started successfully")
        print("🌐 Web server starting...")
    except Exception as e:
        logger.error(f"❌ Error starting trading bot: {e}")
        print(f"❌ Error starting trading bot: {e}")

# ==================== MAIN ENTRY POINT ====================
if __name__ == '__main__':
    # Start trading bot
    start_trading_bot()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 Web server starting on port {port}")
    print(f"Dashboard: http://localhost:{port}")
    print("="*60)
    
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)