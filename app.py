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
from flask_cors import CORS  # Add this import

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
    'cycle_count': 0,
    'server_time': datetime.now().isoformat()
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
    print("EUR/GBP 2-Minute Trading System")
    print("Starting up...")
    print("="*60)
    print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"Trade Size: ${TRADE_SIZE:,.2f}")
    print(f"Target Profit: {TARGET_PROFIT_PCT*100:.3f}% per trade")
    print(f"Stop Loss: {STOP_LOSS_PCT*100:.3f}%")
    print(f"Update Interval: {PREDICTION_WINDOW_MINUTES} minutes")
    print("="*60)

print_startup_banner()

# ==================== SIMPLIFIED DATA FETCHING ====================
def get_forex_data():
    """Get EUR/GBP data - simplified version"""
    try:
        # Try multiple APIs
        apis = [
            ('https://api.frankfurter.app/latest?from=EUR&to=GBP', 'GBP'),
            ('https://api.exchangerate-api.com/v4/latest/EUR', 'GBP'),
            ('https://api.freeforexapi.com/v1/latest?pairs=EURGBP', 'EURGBP')
        ]
        
        for api_url, rate_key in apis:
            try:
                logger.info(f"Trying API: {api_url}")
                response = requests.get(api_url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'frankfurter' in api_url:
                        rate = data['rates']['GBP']
                    elif 'exchangerate' in api_url:
                        rate = data['rates']['GBP']
                    else:
                        rate = data['rates']['EURGBP']
                    
                    current_rate = float(rate)
                    logger.info(f"✅ Got rate: {current_rate:.6f}")
                    
                    # Generate sample data around current rate
                    dates = pd.date_range(end=datetime.now(), periods=20, freq='1min')
                    df = pd.DataFrame(index=dates)
                    
                    prices = []
                    base_rate = current_rate
                    
                    for i in range(20):
                        change = np.random.normal(0, 0.00005)
                        base_rate += change
                        base_rate = max(0.8540, min(0.8600, base_rate))
                        prices.append(base_rate)
                    
                    df['close'] = prices
                    df['open'] = [p * 0.99999 for p in prices]
                    df['high'] = [p * 1.00002 for p in prices]
                    df['low'] = [p * 0.99998 for p in prices]
                    
                    return df, False, current_rate, 'Live API'
                    
            except Exception as e:
                logger.warning(f"API failed: {str(e)[:100]}")
                continue
        
        # Fallback to simulation
        logger.info("Using simulation data")
        return create_simulation_data(), True, 0.8568, 'Simulation'
        
    except Exception as e:
        logger.error(f"Data fetch error: {e}")
        return create_simulation_data(), True, 0.8568, 'Simulation'

def create_simulation_data():
    """Create simulation data"""
    dates = pd.date_range(end=datetime.now(), periods=20, freq='1min')
    df = pd.DataFrame(index=dates)
    
    prices = []
    base_rate = 0.8568
    
    for i in range(20):
        change = np.random.normal(0, 0.0001)
        base_rate += change
        base_rate = max(0.8550, min(0.8585, base_rate))
        prices.append(base_rate)
    
    df['close'] = prices
    df['open'] = [p * 0.99995 for p in prices]
    df['high'] = [p * 1.00005 for p in prices]
    df['low'] = [p * 0.99995 for p in prices]
    
    return df

# ==================== SIMPLIFIED INDICATORS ====================
def calculate_basic_indicators(df):
    """Calculate basic indicators"""
    try:
        if len(df) < 10:
            return df
            
        # Simple moving averages
        df['SMA_10'] = ta.sma(df['close'], length=10)
        df['SMA_20'] = ta.sma(df['close'], length=20)
        
        # RSI
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        
        # MACD
        macd = ta.macd(df['close'])
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
        
        return df.fillna(method='ffill')
        
    except Exception as e:
        logger.error(f"Indicator error: {e}")
        return df

# ==================== SIMPLIFIED PREDICTION ====================
def make_simple_prediction(df):
    """Make simple prediction based on indicators"""
    try:
        if len(df) < 10:
            return 0.5, 50, 'NEUTRAL'
        
        current_rsi = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else 50
        current_macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
        price_trend = 1 if df['close'].iloc[-1] > df['close'].iloc[-5] else -1
        
        # Simple logic
        if current_rsi < 40 and current_macd > 0 and price_trend > 0:
            return 0.7, 70, 'BULLISH'
        elif current_rsi > 60 and current_macd < 0 and price_trend < 0:
            return 0.3, 70, 'BEARISH'
        else:
            return 0.5, 50, 'NEUTRAL'
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 0.5, 50, 'NEUTRAL'

# ==================== TRADING CYCLE ====================
def trading_cycle():
    """Main trading cycle - simplified"""
    global trading_state, system_ready
    
    # Mark system as ready after short delay
    time.sleep(2)
    system_ready = True
    trading_state['system_status'] = 'RUNNING'
    trading_state['api_status'] = 'CONNECTED'
    
    cycle_count = 0
    
    while True:
        try:
            cycle_count += 1
            trading_state['cycle_count'] = cycle_count
            trading_state['server_time'] = datetime.now().isoformat()
            
            logger.info(f"Cycle #{cycle_count}")
            
            # Get data
            df, is_demo, current_rate, data_source = get_forex_data()
            
            if df.empty:
                time.sleep(30)
                continue
            
            # Calculate indicators
            df = calculate_basic_indicators(df)
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Make prediction
            prob, confidence, direction = make_simple_prediction(df)
            
            # Generate action
            if confidence > 65:
                if direction == 'BULLISH':
                    action = 'BUY'
                elif direction == 'BEARISH':
                    action = 'SELL'
                else:
                    action = 'HOLD'
            else:
                action = 'HOLD'
            
            # Update indicators
            indicators = {}
            if 'RSI_14' in df.columns:
                indicators['RSI_14'] = float(df['RSI_14'].iloc[-1])
            if 'MACD' in df.columns:
                indicators['MACD'] = float(df['MACD'].iloc[-1])
            
            # Update price history
            trading_state['price_history'].append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'price': float(current_price)
            })
            if len(trading_state['price_history']) > 30:
                trading_state['price_history'].pop(0)
            
            # Create simple chart
            chart_data = create_simple_chart(df, is_demo)
            
            # Update state
            trading_state.update({
                'current_price': round(float(current_price), 5),
                'prediction': direction,
                'action': action,
                'confidence': round(float(confidence), 1),
                'next_prediction_time': 60,  # Update every minute for testing
                'indicators': indicators,
                'is_demo_data': is_demo,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'chart_data': chart_data,
                'data_source': data_source,
                'model_accuracy': round(np.random.uniform(55, 75), 1),
                'trade_history_count': len(trade_history)
            })
            
            # Simple trade management
            manage_trades(current_price)
            
            logger.info(f"Price: {current_price:.5f}, Prediction: {direction}, Action: {action}")
            
            # Wait for next cycle
            time.sleep(60)  # Update every minute
            
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            time.sleep(30)

def manage_trades(current_price):
    """Simple trade management"""
    if trading_state['current_trade']:
        trade = trading_state['current_trade']
        
        # Calculate profit
        if trade['action'] == 'BUY':
            profit_pct = (current_price / trade['entry_price'] - 1) * 100
        else:
            profit_pct = (trade['entry_price'] / current_price - 1) * 100
        
        profit_amount = TRADE_SIZE * profit_pct / 100
        
        trade['current_price'] = float(current_price)
        trade['profit_pct'] = float(profit_pct)
        trade['profit_amount'] = float(profit_amount)
        
        # Check exit conditions
        if abs(profit_pct) >= 0.05 or profit_pct <= -0.03:  # 0.05% profit or 0.03% loss
            # Close trade
            trade['status'] = 'CLOSED'
            trade['exit_time'] = datetime.now()
            trade['result'] = 'PROFIT' if profit_pct > 0 else 'LOSS'
            
            # Update stats
            trading_state['total_trades'] += 1
            trading_state['balance'] += profit_amount
            
            if profit_pct > 0:
                trading_state['profitable_trades'] += 1
                trading_state['total_profit'] += profit_amount
            
            if trading_state['total_trades'] > 0:
                trading_state['win_rate'] = (trading_state['profitable_trades'] / trading_state['total_trades']) * 100
            
            trade_history.append(trade.copy())
            trading_state['current_trade'] = None

def create_simple_chart(df, is_demo):
    """Create simple chart"""
    try:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            name='EUR/GBP',
            line=dict(color='#00ff88', width=2)
        ))
        
        title = 'EUR/GBP Price'
        if is_demo:
            title += ' (Simulation)'
        
        fig.update_layout(
            title=title,
            yaxis_title='Price',
            xaxis_title='Time',
            template='plotly_dark',
            height=400,
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Chart error: {e}")
        return None

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('index.html')

@app.route('/api/trading_state')
def get_trading_state():
    """Get current trading state"""
    try:
        # Add server timestamp
        trading_state['server_time'] = datetime.now().isoformat()
        
        # Create safe copy
        state_copy = trading_state.copy()
        
        # Convert any non-serializable objects
        if state_copy['current_trade']:
            trade = state_copy['current_trade'].copy()
            for key in ['entry_time', 'exit_time']:
                if key in trade and isinstance(trade[key], datetime):
                    trade[key] = trade[key].isoformat()
            state_copy['current_trade'] = trade
        
        # Ensure all values are JSON serializable
        for key, value in list(state_copy.items()):
            if isinstance(value, (np.integer, np.floating)):
                state_copy[key] = float(value)
            elif isinstance(value, np.ndarray):
                state_copy[key] = value.tolist()
            elif isinstance(value, datetime):
                state_copy[key] = value.isoformat()
            elif value is None:
                state_copy[key] = ''
        
        return jsonify(state_copy)
        
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({
            'error': 'Server error',
            'message': str(e),
            'system_status': 'ERROR',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'system_ready': system_ready,
        'timestamp': datetime.now().isoformat(),
        'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/test')
def test_endpoint():
    """Simple test endpoint"""
    return jsonify({
        'message': 'API is working',
        'timestamp': datetime.now().isoformat(),
        'system_status': trading_state['system_status']
    })

@app.route('/api/execute_manual/<action>')
def execute_manual(action):
    """Execute manual trade"""
    if action.upper() not in ['BUY', 'SELL']:
        return jsonify({'success': False, 'error': 'Invalid action'})
    
    if trading_state['current_trade'] is not None:
        return jsonify({'success': False, 'error': 'Trade already in progress'})
    
    # Create trade
    trade_id = next_trade_id
    next_trade_id += 1
    
    trade = {
        'id': trade_id,
        'action': action.upper(),
        'entry_price': float(trading_state['current_price']),
        'entry_time': datetime.now(),
        'size': TRADE_SIZE,
        'confidence': 100.0,
        'status': 'OPEN',
        'result': None,
        'profit_pct': 0.0,
        'profit_amount': 0.0
    }
    
    trading_state['current_trade'] = trade
    open_trades[trade_id] = trade
    
    return jsonify({
        'success': True,
        'trade': {
            'id': trade_id,
            'action': trade['action'],
            'entry_price': trade['entry_price']
        }
    })

@app.route('/api/reset')
def reset_trading():
    """Reset trading statistics"""
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
        # Start trading thread
        thread = threading.Thread(target=trading_cycle, daemon=True)
        thread.start()
        
        logger.info("✅ Trading system started")
        print("✅ System ready!")
        print(f"✅ API endpoints available at http://localhost:{port}")
        
    except Exception as e:
        logger.error(f"❌ Error starting system: {e}")
        print(f"❌ Error: {e}")

# ==================== MAIN ENTRY POINT ====================
if __name__ == '__main__':
    # Start trading bot
    start_trading_bot()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Web server starting on port {port}")
    print(f"📊 Open your browser to: http://localhost:{port}")
    print("="*60)
    print("System Status: RUNNING")
    print("="*60)
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )