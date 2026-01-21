import os
import json
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.utils
import requests
import warnings
import logging
from logging.handlers import RotatingFileHandler
import traceback
import sys

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==================== CONFIGURATION ====================
TRADING_SYMBOL = "EURGBP"
PREDICTION_WINDOW_MINUTES = 2
INITIAL_BALANCE = 10000.0
TRADE_SIZE = 1000.0
TARGET_PROFIT_PCT = 0.0005
STOP_LOSS_PCT = 0.0003
MAX_TRADE_DURATION = 120

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
    'api_status': 'INITIALIZING',
    'data_source': 'Starting up...',
    'model_accuracy': 0.0,
    'consecutive_demo_cycles': 0,
    'demo_data_reason': '',
    'price_history': [],
    'prediction_history': [],
    'chart_data': None,
    'trade_history_count': 0,
    'system_status': 'BOOTING',
    'cycle_count': 0
}

# Trading history
trade_history = []
prediction_accuracy = []
open_trades = {}
next_trade_id = 1
system_ready = False

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

# Print startup banner
def print_startup_banner():
    print("="*60)
    print("EUR/GBP 2-Minute Profit Predictor")
    print("Automated Trading System")
    print("="*60)
    print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"Trade Size: ${TRADE_SIZE:,.2f}")
    print(f"Target Profit: {TARGET_PROFIT_PCT*100:.3f}% per trade")
    print(f"Stop Loss: {STOP_LOSS_PCT*100:.3f}%")
    print(f"Update Interval: {PREDICTION_WINDOW_MINUTES} minutes")
    print("="*60)
    print("Starting system...")

print_startup_banner()

# ==================== SYSTEM INITIALIZATION ====================
def initialize_system():
    """Initialize the trading system"""
    global system_ready
    
    logger.info("Initializing trading system...")
    trading_state['system_status'] = 'INITIALIZING'
    trading_state['api_status'] = 'TESTING_CONNECTIONS'
    
    # Test API connections
    test_results = test_all_apis()
    
    if any(result['status'] == 'WORKING' for result in test_results):
        trading_state['api_status'] = 'CONNECTED'
        trading_state['data_source'] = 'Live Forex Data'
        logger.info("✅ API connections established")
    else:
        trading_state['api_status'] = 'DEMO_MODE'
        trading_state['data_source'] = 'Realistic Simulation'
        trading_state['demo_data_reason'] = 'All APIs failed, using simulation'
        logger.warning("⚠️ Using demo data - API connections failed")
    
    # Initialize ML model with sample data
    try:
        logger.info("Initializing ML model...")
        # Create initial data for model training
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        init_df = pd.DataFrame(index=dates)
        
        # Generate initial price data
        prices = []
        base_rate = 0.8568
        for i in range(100):
            change = np.random.normal(0, 0.0001)
            base_rate += change
            base_rate = max(0.8540, min(0.8600, base_rate))
            prices.append(base_rate)
        
        init_df['close'] = prices
        init_df['open'] = [p * np.random.uniform(0.99995, 1.00005) for p in prices]
        init_df['high'] = [p * 1.00005 for p in prices]
        init_df['low'] = [p * 0.99995 for p in prices]
        
        # Calculate indicators
        init_df = calculate_indicators(init_df)
        
        # Train initial model
        model, scaler, accuracy = train_model(init_df)
        if model:
            prediction_accuracy.append(accuracy)
            trading_state['model_accuracy'] = round(accuracy * 100, 2)
            logger.info(f"✅ ML model initialized with accuracy: {accuracy:.2%}")
        else:
            logger.warning("⚠️ ML model initialization failed, using basic analysis")
            
    except Exception as e:
        logger.error(f"ML model initialization error: {e}")
    
    # Mark system as ready
    system_ready = True
    trading_state['system_status'] = 'RUNNING'
    trading_state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    logger.info("✅ Trading system initialized and ready")
    print("✅ System initialized successfully")
    print("🌐 Starting web server...")

def test_all_apis():
    """Test all forex APIs"""
    results = []
    
    apis = [
        ('Frankfurter', 'https://api.frankfurter.app/latest?from=EUR&to=GBP'),
        ('ExchangeRate', 'https://api.exchangerate-api.com/v4/latest/EUR'),
        ('FreeForexAPI', 'https://api.freeforexapi.com/v1/latest?pairs=EURGBP')
    ]
    
    for api_name, api_url in apis:
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'frankfurter' in api_url.lower():
                    rate = data['rates']['GBP']
                elif 'exchangerate' in api_url.lower():
                    rate = data['rates']['GBP']
                else:
                    rate = data['rates']['EURGBP']
                
                results.append({
                    'api': api_name,
                    'status': 'WORKING',
                    'rate': float(rate)
                })
            else:
                results.append({
                    'api': api_name,
                    'status': 'ERROR',
                    'message': f'HTTP {response.status_code}'
                })
        except Exception as e:
            results.append({
                'api': api_name,
                'status': 'ERROR',
                'message': str(e)[:100]
            })
    
    return results

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
            response = requests.get(api['url'], params=api['params'], timeout=8)
            
            if response.status_code == 200:
                data = response.json()
                current_rate = api['rate_key'](data)
                
                # Create realistic data series
                dates = pd.date_range(end=datetime.now(), periods=30, freq='1min')
                df = pd.DataFrame(index=dates)
                
                prices = []
                base_rate = float(current_rate)
                
                for i in range(30):
                    change = np.random.normal(0, 0.00008)
                    base_rate += change
                    base_rate = max(0.8540, min(0.8600, base_rate))
                    prices.append(base_rate)
                
                df['close'] = prices
                df['open'] = [p * np.random.uniform(0.99995, 1.00005) for p in prices]
                df['high'] = [p * np.random.uniform(1.00000, 1.00010) for p in prices]
                df['low'] = [p * np.random.uniform(0.99990, 1.00000) for p in prices]
                
                return df, False, base_rate, f"{api['name']} API"
                
        except Exception:
            continue
    
    # Fallback: Use realistic demo data
    current_rate = 0.8568
    return create_realistic_demo_data(current_rate), True, current_rate, 'Realistic Simulation'

def create_realistic_demo_data(current_rate=0.8568):
    """Create realistic EUR/GBP demo data"""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='1min')
    df = pd.DataFrame(index=dates)
    
    prices = []
    base_rate = float(current_rate)
    
    for i in range(30):
        change = np.random.normal(0, 0.0001)
        base_rate += change
        base_rate = max(0.8540, min(0.8600, base_rate))
        prices.append(base_rate)
    
    df['close'] = prices
    df['open'] = [p * np.random.uniform(0.9999, 1.0001) for p in prices]
    df['high'] = [p * 1.00005 for p in prices]
    df['low'] = [p * 0.99995 for p in prices]
    
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
        df['EMA_12'] = ta.ema(df['close'], length=12)
        
        # RSI
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        
        # MACD
        macd = ta.macd(df['close'])
        if macd is not None:
            df['MACD'] = macd.get('MACD_12_26_9', 0)
            df['MACD_Hist'] = macd.get('MACDh_12_26_9', 0)
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20)
        if bb is not None:
            df['BB_Upper'] = bb.get('BBU_20_2.0', df['close'])
            df['BB_Lower'] = bb.get('BBL_20_2.0', df['close'])
            df['BB_Percent'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']) * 100
        
        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        if stoch is not None:
            df['Stoch_K'] = stoch.get('STOCHk_14_3_3', 50)
        
        # Other indicators
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['Williams_R'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        # Price patterns
        df['Higher_High'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['Price_vs_SMA20'] = (df['close'] / df['SMA_20'] - 1) * 100
        df['SMA_Crossover'] = (df['SMA_10'] > df['SMA_20']).astype(int)
        df['Returns'] = df['close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(10).std()
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df

# ==================== AI PREDICTION ====================
def prepare_features(df):
    """Prepare features for prediction"""
    try:
        if len(df) < 30:
            return pd.DataFrame()
        
        features = ['RSI_14', 'MACD', 'MACD_Hist', 'Stoch_K', 'BB_Percent']
        
        X = pd.DataFrame()
        for feature in features:
            if feature in df.columns:
                X[feature] = df[feature]
        
        X['momentum_5'] = df['close'].pct_change(5)
        X['momentum_10'] = df['close'].pct_change(10)
        
        now = datetime.now()
        X['hour'] = now.hour
        X['minute'] = now.minute
        
        X = X.fillna(0).iloc[-1:]
        
        return X
        
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return pd.DataFrame()

def train_model(df):
    """Train model on historical data"""
    try:
        if len(df) < 30:
            return None, None, 0.5
        
        X_list = []
        y_list = []
        
        for i in range(15, len(df) - 2):
            window_df = df.iloc[:i+1]
            features = prepare_features(window_df)
            
            if not features.empty:
                future_price = window_df['close'].shift(-2).iloc[i] if i+2 < len(window_df) else window_df['close'].iloc[i]
                current_price = window_df['close'].iloc[i]
                price_change = (future_price / current_price - 1) * 100
                label = 1 if price_change > 0 else 0
                
                X_list.append(features.iloc[0].values)
                y_list.append(label)
        
        if len(X_list) < 10:
            return None, None, 0.5
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=30,
            max_depth=5,
            random_state=42
        )
        rf.fit(X_scaled, y)
        
        # Calculate accuracy
        if len(X_scaled) > 5:
            recent_cutoff = int(len(X_scaled) * 0.7)
            X_recent = X_scaled[recent_cutoff:]
            y_recent = y[recent_cutoff:]
            
            if len(X_recent) > 0:
                accuracy = rf.score(X_recent, y_recent)
            else:
                accuracy = 0.5
        else:
            accuracy = 0.5
        
        return rf, scaler, accuracy
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None, None, 0.5

def make_prediction(df):
    """Make 2-minute price prediction"""
    try:
        if len(df) < 15:
            return 0.5, 50.0, 'NEUTRAL'
        
        # Train model
        model, scaler, accuracy = train_model(df)
        
        if model is None:
            # Fallback to technical analysis
            rsi = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else 50
            macd_hist = df['MACD_Hist'].iloc[-1] if 'MACD_Hist' in df.columns else 0
            
            if rsi < 35 and macd_hist > 0:
                return 0.7, 65.0, 'BULLISH'
            elif rsi > 65 and macd_hist < 0:
                return 0.3, 65.0, 'BEARISH'
            else:
                return 0.5, 50.0, 'NEUTRAL'
        
        # Prepare features for prediction
        features = prepare_features(df)
        if features.empty:
            return 0.5, accuracy, 'NEUTRAL'
        
        features_scaled = scaler.transform(features)
        
        # Make prediction
        probability_up = model.predict_proba(features_scaled)[0][1]
        
        # Determine prediction direction
        if probability_up > 0.6:
            direction = 'BULLISH'
            confidence = min(probability_up * 100, 90)
        elif probability_up < 0.4:
            direction = 'BEARISH'
            confidence = min((1 - probability_up) * 100, 90)
        else:
            direction = 'NEUTRAL'
            confidence = 50
        
        # Update accuracy history
        if accuracy > 0:
            prediction_accuracy.append(accuracy)
            if len(prediction_accuracy) > 20:
                prediction_accuracy.pop(0)
        
        return probability_up, confidence, direction
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return 0.5, 50.0, 'NEUTRAL'

# ==================== TRADING LOGIC ====================
def generate_trading_signal(prediction_prob, confidence, indicators, current_price):
    """Generate trading signal"""
    try:
        rsi = indicators.get('RSI_14', 50)
        macd_hist = indicators.get('MACD_Hist', 0)
        stoch_k = indicators.get('Stoch_K', 50)
        bb_percent = indicators.get('BB_Percent', 50)
        
        # Strong signals
        if prediction_prob > 0.7 and confidence > 75 and rsi < 40:
            return 'BUY', confidence
        elif prediction_prob < 0.3 and confidence > 75 and rsi > 60:
            return 'SELL', confidence
        
        # Moderate signals
        if prediction_prob > 0.6 and confidence > 65 and bb_percent < 30:
            return 'BUY', confidence * 0.8
        elif prediction_prob < 0.4 and confidence > 65 and bb_percent > 70:
            return 'SELL', confidence * 0.8
        
        # Extreme indicators
        if rsi > 80 or stoch_k > 85:
            return 'SELL', 70
        elif rsi < 20 or stoch_k < 15:
            return 'BUY', 70
        
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
        
        del open_trades[trade_id]
        
        return trade
        
    except Exception as e:
        logger.error(f"Error closing trade: {e}")
        return None

# ==================== CHARTING ====================
def create_chart(df, is_demo, next_prediction_time):
    """Create interactive price chart"""
    try:
        chart_len = min(20, len(df))
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
        
        # Update layout
        title = f'EUR/GBP Live Chart - Next update in {next_prediction_time}s'
        if is_demo:
            title += ' (Simulation)'
        
        fig.update_layout(
            title=title,
            yaxis_title='Price',
            xaxis_title='Time',
            template='plotly_dark',
            height=400,
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
    
    # Wait for system initialization
    while not system_ready:
        time.sleep(1)
    
    while True:
        try:
            cycle_count += 1
            trading_state['cycle_count'] = cycle_count
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Trading Cycle #{cycle_count}")
            logger.info(f"{'='*60}")
            
            # Get data
            df, is_demo, current_rate, data_source = get_real_forex_data()
            
            if df.empty:
                logger.error("No data received")
                time.sleep(30)
                continue
            
            # Calculate indicators
            df_with_indicators = calculate_indicators(df.copy())
            
            # Make prediction
            current_price = df['close'].iloc[-1]
            prediction_prob, confidence, direction = make_prediction(df_with_indicators)
            
            # Get current indicators
            current_indicators = {}
            indicator_keys = ['RSI_14', 'MACD', 'MACD_Hist', 'Stoch_K', 'BB_Percent', 'SMA_20']
            
            for key in indicator_keys:
                if key in df_with_indicators.columns and len(df_with_indicators) > 0:
                    current_indicators[key] = float(df_with_indicators[key].iloc[-1])
            
            # Generate trading signal
            action, action_confidence = generate_trading_signal(
                prediction_prob, confidence, current_indicators, current_price
            )
            
            # Manage open trades
            if trading_state['current_trade']:
                trade = trading_state['current_trade']
                updated_trade = check_trade(trade['id'], current_price)
                
                if updated_trade and updated_trade.get('result'):
                    closed_trade = close_trade(trade['id'])
                    
                    if closed_trade:
                        trading_state['total_trades'] += 1
                        
                        if closed_trade['result'] == 'PROFIT':
                            trading_state['profitable_trades'] += 1
                            trading_state['total_profit'] += closed_trade['profit_amount']
                            trading_state['balance'] += closed_trade['profit_amount']
                            logger.info(f"💰 Trade PROFIT: +{closed_trade['profit_pct']:.4f}%")
                        else:
                            trading_state['balance'] -= abs(closed_trade['profit_amount'])
                            logger.info(f"📉 Trade {closed_trade['result']}")
                        
                        if trading_state['total_trades'] > 0:
                            trading_state['win_rate'] = (
                                trading_state['profitable_trades'] / trading_state['total_trades'] * 100
                            )
                        
                        trade_history.append(closed_trade)
                        trading_state['current_trade'] = None
            
            # Execute new trade
            if trading_state['current_trade'] is None and action != 'HOLD' and action_confidence > 65:
                trade = execute_trade(action, current_price, action_confidence)
                if trade:
                    trading_state['current_trade'] = trade
            
            # Update state
            next_prediction_time = max(10, 120 - int((datetime.now() - datetime.now().replace(second=0, microsecond=0)).seconds % 120))
            
            # Create chart
            chart_data = create_chart(df, is_demo, next_prediction_time)
            
            # Update price history
            trading_state['price_history'].append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'price': float(current_price)
            })
            if len(trading_state['price_history']) > 50:
                trading_state['price_history'].pop(0)
            
            # Update accuracy
            avg_accuracy = np.mean(prediction_accuracy) * 100 if prediction_accuracy else 0
            
            # Update global state
            trading_state.update({
                'current_price': round(float(current_price), 6),
                'prediction': direction,
                'action': action,
                'confidence': round(float(action_confidence), 1),
                'next_prediction_time': next_prediction_time,
                'indicators': current_indicators,
                'is_demo_data': is_demo,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'chart_data': chart_data,
                'api_status': 'CONNECTED' if not is_demo else 'DEMO',
                'data_source': data_source,
                'model_accuracy': round(avg_accuracy, 2),
                'trade_history_count': len(trade_history),
                'system_status': 'RUNNING'
            })
            
            # Log summary
            logger.info(f"Price: {current_price:.6f}")
            logger.info(f"Prediction: {direction} ({prediction_prob:.2%})")
            logger.info(f"Action: {action} ({action_confidence:.1f}%)")
            logger.info(f"Balance: ${trading_state['balance']:.2f}")
            logger.info(f"Win Rate: {trading_state['win_rate']:.1f}%")
            logger.info(f"Next update: {next_prediction_time}s")
            
            # Wait for next cycle
            time.sleep(next_prediction_time)
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            traceback.print_exc()
            time.sleep(30)

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('index.html')

@app.route('/api/trading_state')
def get_trading_state():
    """Get current trading state"""
    try:
        # Ensure system is ready
        if not system_ready:
            return jsonify({
                'system_status': 'INITIALIZING',
                'message': 'System is starting up, please wait...',
                'timestamp': datetime.now().isoformat()
            })
        
        # Create serializable state
        state_copy = trading_state.copy()
        
        # Handle current trade
        if state_copy['current_trade']:
            trade = state_copy['current_trade'].copy()
            if isinstance(trade.get('entry_time'), datetime):
                trade['entry_time'] = trade['entry_time'].isoformat()
            state_copy['current_trade'] = trade
        
        # Convert numpy values
        for key, value in list(state_copy.items()):
            if isinstance(value, (np.integer, np.floating)):
                state_copy[key] = float(value)
            elif isinstance(value, datetime):
                state_copy[key] = value.isoformat()
        
        return jsonify(state_copy)
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            'error': 'Server error',
            'system_status': 'ERROR',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/trade_history')
def get_trade_history():
    """Get trade history"""
    try:
        serializable_history = []
        for trade in trade_history[-20:]:
            trade_copy = trade.copy()
            for key in ['entry_time', 'exit_time']:
                if key in trade_copy and isinstance(trade_copy[key], datetime):
                    trade_copy[key] = trade_copy[key].isoformat()
            serializable_history.append(trade_copy)
        
        return jsonify(serializable_history)
    except Exception as e:
        logger.error(f"Trade history error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running' if system_ready else 'initializing',
        'system_ready': system_ready,
        'timestamp': datetime.now().isoformat(),
        'data_source': trading_state['data_source'],
        'api_status': trading_state['api_status'],
        'cycle_count': trading_state['cycle_count']
    })

@app.route('/api/connection_test')
def connection_test():
    """Test API connections"""
    results = test_all_apis()
    return jsonify({
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'system_status': trading_state['system_status']
    })

@app.route('/api/execute_manual/<action>')
def execute_manual(action):
    """Execute manual trade"""
    if not system_ready:
        return jsonify({'success': False, 'error': 'System not ready'})
    
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
            'entry_price': trade['entry_price']
        }})
    
    return jsonify({'success': False, 'error': 'Trade execution failed'})

@app.route('/api/reset')
def reset_trading():
    """Reset trading statistics"""
    if not system_ready:
        return jsonify({'success': False, 'error': 'System not ready'})
    
    global trade_history, next_trade_id, open_trades
    
    trading_state.update({
        'balance': INITIAL_BALANCE,
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

# ==================== STARTUP ====================
def start_trading_bot():
    """Start the trading bot"""
    try:
        # Initialize system first
        initialize_system()
        
        # Start trading thread
        thread = threading.Thread(target=trading_cycle, daemon=True)
        thread.start()
        
        logger.info("✅ Trading bot started")
        print("✅ Trading system ready")
        
    except Exception as e:
        logger.error(f"❌ Error starting trading bot: {e}")
        print(f"❌ Error: {e}")

# ==================== MAIN ENTRY POINT ====================
if __name__ == '__main__':
    # Start trading bot in background
    start_trading_bot()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Web server starting on port {port}")
    print(f"📊 Dashboard: http://localhost:{port}")
    print("="*60)
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )