let autoRefreshInterval = null;
let lastUpdateTime = null;
let tradeHistory = [];

function formatTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatPrice(price) {
    return parseFloat(price).toFixed(6);
}

function updateTradingData(data) {
    // Update data source indicator
    const warningEl = document.getElementById('data-warning');
    if (warningEl) {
        if (data.is_demo_data) {
            warningEl.innerHTML = `
                <div class="demo-warning">
                    ⚠️ USING DEMONSTRATION DATA - Real-time predictions paused
                </div>
            `;
        } else {
            warningEl.innerHTML = `
                <div class="real-data-banner">
                    ✅ LIVE TRADING ACTIVE - Real-time EUR/GBP predictions
                </div>
            `;
        }
    }
    
    // Update stats
    document.getElementById('current-price').textContent = formatPrice(data.current_price);
    document.getElementById('account-balance').textContent = `$${data.balance.toFixed(2)}`;
    document.getElementById('total-trades').textContent = data.total_trades;
    document.getElementById('win-rate').textContent = `${data.win_rate.toFixed(1)}%`;
    document.getElementById('total-profit').textContent = `$${data.total_profit.toFixed(2)}`;
    
    // Update countdown
    const countdown = data.next_prediction_time || 120;
    document.getElementById('countdown').textContent = `${Math.floor(countdown)}s`;
    
    // Update prediction
    const predictionEl = document.getElementById('prediction');
    predictionEl.textContent = data.prediction;
    predictionEl.className = `prediction-value ${data.prediction.toLowerCase()}`;
    
    // Update confidence
    const confidence = data.confidence || 50;
    document.getElementById('confidence-percentage').textContent = `${confidence}%`;
    document.getElementById('confidence-fill').style.width = `${Math.min(confidence, 100)}%`;
    
    // Update action signal
    const actionEl = document.getElementById('action-signal');
    const actionTextEl = document.getElementById('action-text');
    const actionSubtextEl = document.getElementById('action-subtext');
    
    actionTextEl.textContent = data.action;
    actionEl.className = `action-signal ${data.action.toLowerCase()}-signal`;
    
    let subtext = '';
    if (data.action === 'BUY') {
        subtext = 'Predicting price increase in next 2 minutes';
    } else if (data.action === 'SELL') {
        subtext = 'Predicting price decrease in next 2 minutes';
    } else {
        subtext = 'Waiting for better trading opportunity';
    }
    
    if (data.is_demo_data) {
        subtext += ' (Demo Mode)';
    }
    
    actionSubtextEl.textContent = subtext;
    
    // Update active trade
    const activeTradeEl = document.getElementById('active-trade');
    if (data.current_trade) {
        const trade = data.current_trade;
        const currentPrice = data.current_price;
        const entryPrice = trade.entry_price;
        
        let pnlPct = ((currentPrice / entryPrice) - 1) * 100;
        if (trade.action === 'SELL') {
            pnlPct = -pnlPct;
        }
        
        const tradeDuration = Math.floor((new Date() - new Date(trade.entry_time)) / 1000);
        
        activeTradeEl.innerHTML = `
            <h3>💰 Active Trade</h3>
            <div class="trade-info">
                <div class="trade-item">
                    <div class="trade-label">Action</div>
                    <div class="trade-value ${trade.action.toLowerCase()}">${trade.action}</div>
                </div>
                <div class="trade-item">
                    <div class="trade-label">Entry Price</div>
                    <div class="trade-value">${formatPrice(entryPrice)}</div>
                </div>
                <div class="trade-item">
                    <div class="trade-label">Current P&L</div>
                    <div class="trade-value ${pnlPct >= 0 ? 'profit' : 'loss'}">
                        ${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(4)}%
                    </div>
                </div>
                <div class="trade-item">
                    <div class="trade-label">Time Elapsed</div>
                    <div class="trade-value">${tradeDuration}s</div>
                </div>
            </div>
        `;
        activeTradeEl.style.display = 'block';
    } else {
        activeTradeEl.style.display = 'none';
    }
    
    // Update indicators
    const indicators = data.indicators || {};
    const indicatorIds = ['RSI_14', 'MACD_fast', 'Stoch_K', 'ADX', 'ATR_14', 'Williams_R'];
    
    indicatorIds.forEach(id => {
        const el = document.getElementById(`indicator-${id.toLowerCase()}`);
        if (el) {
            const value = indicators[id] || 0;
            el.textContent = typeof value === 'number' ? value.toFixed(4) : '--';
        }
    });
    
    // Update chart
    if (data.chart_data) {
        try {
            const chartData = JSON.parse(data.chart_data);
            Plotly.react('price-chart', chartData.data, chartData.layout);
        } catch (e) {
            console.error('Error updating chart:', e);
        }
    }
    
    // Update timestamp
    if (data.last_update) {
        document.getElementById('last-updated').textContent = 
            `Last updated: ${formatTime(data.last_update)}`;
        lastUpdateTime = new Date(data.last_update);
    }
    
    // Show/hide loading
    document.getElementById('loading').style.display = 'none';
    document.getElementById('content').style.display = 'block';
}

function loadTradeHistory() {
    fetch('/api/trade_history')
        .then(response => response.json())
        .then(data => {
            tradeHistory = data;
            updateHistoryTable();
        })
        .catch(error => {
            console.error('Error loading trade history:', error);
        });
}

function updateHistoryTable() {
    const tbody = document.getElementById('history-tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    // Show last 10 trades
    const recentTrades = tradeHistory.slice(-10).reverse();
    
    recentTrades.forEach(trade => {
        const row = document.createElement('tr');
        
        const profitClass = trade.result === 'PROFIT' ? 'profit-cell' : 'loss-cell';
        const profitSign = trade.result === 'PROFIT' ? '+' : '';
        
        row.innerHTML = `
            <td>${formatTime(trade.timestamp)}</td>
            <td><strong>${trade.action}</strong></td>
            <td>${formatPrice(trade.entry_price)}</td>
            <td>${trade.exit_price ? formatPrice(trade.exit_price) : '--'}</td>
            <td class="${profitClass}">${profitSign}${parseFloat(trade.profit_pct || 0).toFixed(4)}%</td>
            <td>${trade.result || '--'}</td>
        `;
        
        tbody.appendChild(row);
    });
}

function executeManualTrade(action) {
    if (confirm(`Execute manual ${action} trade?`)) {
        fetch(`/api/execute_manual/${action}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(`Manual ${action} trade executed!`);
                    fetchTradingData();
                } else {
                    alert(`Error: ${data.error}`);
                }
            })
            .catch(error => {
                console.error('Error executing trade:', error);
                alert('Error executing trade');
            });
    }
}

function resetTrading() {
    if (confirm('Reset all trading statistics? This cannot be undone.')) {
        fetch('/api/reset')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Trading statistics reset!');
                    fetchTradingData();
                    loadTradeHistory();
                }
            })
            .catch(error => {
                console.error('Error resetting trading:', error);
            });
    }
}

function fetchTradingData() {
    fetch('/api/trading_state')
        .then(response => {
            if (!response.ok) throw new Error('Network error');
            return response.json();
        })
        .then(data => {
            updateTradingData(data);
        })
        .catch(error => {
            console.error('Error fetching trading data:', error);
            document.getElementById('loading').style.display = 'none';
            setTimeout(fetchTradingData, 5000);
        });
}

function startAutoRefresh() {
    // Fetch immediately
    fetchTradingData();
    loadTradeHistory();
    
    // Set up interval for every 5 seconds
    autoRefreshInterval = setInterval(fetchTradingData, 5000);
    
    // Update trade history every 30 seconds
    setInterval(loadTradeHistory, 30000);
    
    // Update countdown every second
    setInterval(updateCountdown, 1000);
}

function updateCountdown() {
    const countdownEl = document.getElementById('countdown');
    if (countdownEl && countdownEl.textContent) {
        const current = parseInt(countdownEl.textContent);
        if (current > 0) {
            countdownEl.textContent = `${current - 1}s`;
        }
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Start auto-refresh
    startAutoRefresh();
    
    // Set up manual trade buttons
    document.getElementById('manual-buy')?.addEventListener('click', () => executeManualTrade('buy'));
    document.getElementById('manual-sell')?.addEventListener('click', () => executeManualTrade('sell'));
    document.getElementById('reset-btn')?.addEventListener('click', resetTrading);
    
    // Check health every minute
    setInterval(() => {
        fetch('/api/health')
            .then(response => response.json())
            .then(data => {
                console.log('System health:', data);
            });
    }, 60000);
});