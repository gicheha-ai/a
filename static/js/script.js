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
    const isDemo = data.is_demo_data || false;
    const dataSource = data.data_source || 'Unknown';
    
    // Update data source indicator with better messaging
    const warningEl = document.getElementById('data-warning');
    if (warningEl) {
        if (isDemo) {
            warningEl.innerHTML = `
                <div class="demo-warning">
                    <i class="fas fa-exclamation-triangle blink"></i>
                    ⚠️ USING REALISTIC SIMULATION DATA
                    ${data.consecutive_demo_cycles ? `<span class="demo-counter">${data.consecutive_demo_cycles} cycles</span>` : ''}
                </div>
                <div class="demo-info">
                    <i class="fas fa-info-circle"></i>
                    ${data.demo_data_reason || 'Free API connection issues'}<br>
                    <small>Showing realistic EUR/GBP data based on current market patterns. Real-time data will resume when connection is restored.</small>
                </div>
            `;
        } else {
            warningEl.innerHTML = `
                <div class="real-data-banner">
                    <i class="fas fa-satellite-dish"></i>
                    ✅ LIVE MARKET DATA - ${dataSource.toUpperCase()}
                </div>
                <div class="real-data-info">
                    <i class="fas fa-broadcast-tower"></i>
                    Real EUR/GBP exchange rates updated every 2 minutes<br>
                    <small>Using free forex APIs for accurate market data</small>
                </div>
            `;
        }
    }
    
    // Update status indicators
    updateStatusIndicators(isDemo, data.api_status || 'UNKNOWN');
    
    // Update current price with demo indicator
    const priceEl = document.getElementById('current-price');
    priceEl.textContent = formatPrice(data.current_price);
    priceEl.className = `stat-value ${isDemo ? 'demo-price' : 'profit'}`;
    
    // Update account balance styling
    const balanceEl = document.getElementById('account-balance');
    if (isDemo) {
        balanceEl.innerHTML = `$${data.balance.toFixed(2)} <span style="color:#ff9800;font-size:0.7em;">(Virtual)</span>`;
    } else {
        balanceEl.textContent = `$${data.balance.toFixed(2)}`;
    }
    balanceEl.className = `stat-value ${isDemo ? '' : 'profit'}`;
    
    // Update stats
    document.getElementById('total-trades').textContent = data.total_trades;
    document.getElementById('win-rate').textContent = `${data.win_rate.toFixed(1)}%`;
    document.getElementById('total-profit').textContent = `$${data.total_profit.toFixed(2)}`;
    
    // Update countdown
    const countdown = data.next_prediction_time || 120;
    document.getElementById('countdown').textContent = `${Math.floor(countdown)}s`;
    
    // Update prediction with demo styling
    const predictionEl = document.getElementById('prediction');
    predictionEl.textContent = data.prediction;
    predictionEl.className = `prediction-value ${data.prediction.toLowerCase()} ${isDemo ? 'blink' : ''}`;
    
    // Update confidence
    const confidence = data.confidence || 50;
    document.getElementById('confidence-percentage').textContent = `${confidence}%`;
    const confidenceFill = document.getElementById('confidence-fill');
    confidenceFill.style.width = `${Math.min(confidence, 100)}%`;
    
    // Change confidence bar color for demo
    if (isDemo) {
        confidenceFill.style.background = 'linear-gradient(90deg, #ff9800, #ff5722)';
    } else {
        confidenceFill.style.background = 'linear-gradient(90deg, #ff4444, #ffcc00, #00ff88)';
    }
    
    // Update action signal
    const actionEl = document.getElementById('action-signal');
    const actionTextEl = document.getElementById('action-text');
    const actionSubtextEl = document.getElementById('action-subtext');
    
    actionTextEl.textContent = data.action;
    actionEl.className = `action-signal ${data.action.toLowerCase()}-signal ${isDemo ? 'demo-action' : ''}`;
    
    let subtext = '';
    if (data.action === 'BUY') {
        subtext = 'Predicting price increase in next 2 minutes';
    } else if (data.action === 'SELL') {
        subtext = 'Predicting price decrease in next 2 minutes';
    } else {
        subtext = 'Waiting for better trading opportunity';
    }
    
    if (isDemo) {
        subtext += ` (Simulation Mode)`;
        actionTextEl.innerHTML = `${data.action} <span style="font-size:0.7em;color:#ff9800;">(SIM)</span>`;
    }
    
    actionSubtextEl.textContent = subtext;
    
    // Update active trade with demo indicator
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
            <h3><i class="fas fa-trade ${isDemo ? 'blink' : ''}"></i> Active Trade ${isDemo ? '<span class="demo-counter">SIMULATION</span>' : ''}</h3>
            <div class="trade-info">
                <div class="trade-item">
                    <div class="trade-label">Action</div>
                    <div class="trade-value ${trade.action.toLowerCase()} ${isDemo ? 'blink' : ''}">${trade.action}</div>
                </div>
                <div class="trade-item">
                    <div class="trade-label">Entry Price</div>
                    <div class="trade-value ${isDemo ? 'demo-price' : ''}">${formatPrice(entryPrice)}</div>
                </div>
                <div class="trade-item">
                    <div class="trade-label">Current P&L</div>
                    <div class="trade-value ${pnlPct >= 0 ? 'profit' : 'loss'} ${isDemo ? 'blink' : ''}">
                        ${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(4)}%
                    </div>
                </div>
                <div class="trade-item">
                    <div class="trade-label">Time Elapsed</div>
                    <div class="trade-value">${tradeDuration}s</div>
                </div>
            </div>
            ${isDemo ? '<p style="text-align:center;color:#ff9800;margin-top:10px;"><i class="fas fa-exclamation-circle"></i> Simulation Trade - No real money involved</p>' : ''}
        `;
        activeTradeEl.style.display = 'block';
    } else {
        activeTradeEl.style.display = 'none';
    }
    
    // Update indicators with demo styling
    const indicators = data.indicators || {};
    const indicatorIds = ['RSI_14', 'MACD', 'Stoch_K', 'Stoch_D', 'SMA_10', 'SMA_20', 'MACD_Hist', 'BB_Upper', 'BB_Lower'];
    
    indicatorIds.forEach(id => {
        const el = document.getElementById(`indicator-${id.toLowerCase()}`);
        if (el) {
            const value = indicators[id] || 0;
            el.textContent = typeof value === 'number' ? value.toFixed(4) : '--';
            
            // Add demo class to indicator items
            const indicatorItem = el.closest('.indicator-item');
            if (indicatorItem) {
                if (isDemo) {
                    indicatorItem.classList.add('demo-indicator');
                } else {
                    indicatorItem.classList.remove('demo-indicator');
                }
            }
        }
    });
    
    // Update chart with demo watermark
    if (data.chart_data) {
        try {
            const chartData = JSON.parse(data.chart_data);
            
            // Add demo watermark to chart
            if (isDemo) {
                chartData.layout.title = chartData.layout.title + ' - SIMULATION DATA';
                chartData.layout.paper_bgcolor = 'rgba(255, 152, 0, 0.05)';
                
                if (!chartData.layout.annotations) {
                    chartData.layout.annotations = [];
                }
                
                chartData.layout.annotations.push({
                    text: 'SIMULATION DATA<br>Realistic EUR/GBP Pattern',
                    xref: 'paper',
                    yref: 'paper',
                    x: 0.5,
                    y: 0.5,
                    showarrow: false,
                    font: {
                        size: 20,
                        color: 'rgba(255, 152, 0, 0.15)',
                        family: 'Arial, sans-serif'
                    },
                    bgcolor: 'rgba(0,0,0,0)',
                    bordercolor: 'rgba(255, 152, 0, 0.2)',
                    borderwidth: 2,
                    borderpad: 10,
                    opacity: 0.7,
                    textangle: -30
                });
            }
            
            Plotly.react('price-chart', chartData.data, chartData.layout);
            
            // Add CSS class to chart container for demo
            const chartContainer = document.getElementById('price-chart').parentElement;
            if (isDemo) {
                chartContainer.classList.add('demo-chart');
            } else {
                chartContainer.classList.remove('demo-chart');
            }
            
        } catch (e) {
            console.error('Error updating chart:', e);
        }
    }
    
    // Update timestamp with demo info
    if (data.last_update) {
        let timestampText = `Last updated: ${formatTime(data.last_update)}`;
        if (isDemo) {
            timestampText += ` • <span class="status-disconnected status-indicator">SIMULATION MODE</span>`;
        } else {
            timestampText += ` • <span class="status-connected status-indicator">LIVE MARKET DATA</span>`;
        }
        document.getElementById('last-updated').innerHTML = timestampText;
        lastUpdateTime = new Date(data.last_update);
    }
    
    // Update footer with demo warning
    updateFooterWarning(isDemo, dataSource);
    
    // Show/hide loading
    document.getElementById('loading').style.display = 'none';
    document.getElementById('content').style.display = 'block';
}

function updateStatusIndicators(isDemo, apiStatus) {
    // Remove any existing status indicators
    const existingStatus = document.querySelectorAll('.status-indicator');
    existingStatus.forEach(el => {
        if (!el.closest('#last-updated')) {
            el.remove();
        }
    });
    
    // Add status to stats grid
    const statsGrid = document.querySelector('.stats-grid');
    if (statsGrid) {
        const existingStatusDiv = statsGrid.parentElement.querySelector('.status-indicator:not(#last-updated *)');
        if (existingStatusDiv) {
            existingStatusDiv.remove();
        }
        
        const statusDiv = document.createElement('div');
        statusDiv.className = `status-indicator ${isDemo ? 'status-disconnected' : 'status-connected'}`;
        statusDiv.style.margin = '10px auto';
        statusDiv.style.display = 'inline-block';
        statusDiv.innerHTML = `<i class="fas ${isDemo ? 'fa-sim-card' : 'fa-satellite'}"></i> ${isDemo ? 'SIMULATION' : 'LIVE'} | API: ${apiStatus}`;
        
        if (statsGrid.parentElement) {
            statsGrid.parentElement.insertBefore(statusDiv, statsGrid);
        }
    }
}

function updateFooterWarning(isDemo, dataSource) {
    const footer = document.querySelector('footer');
    if (!footer) return;
    
    // Remove existing warning
    const existingWarning = footer.querySelector('.footer-warning');
    if (existingWarning) {
        existingWarning.remove();
    }
    
    if (isDemo) {
        const warningDiv = document.createElement('div');
        warningDiv.className = 'footer-warning';
        warningDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            SYSTEM RUNNING IN SIMULATION MODE - Showing realistic EUR/GBP data patterns
        `;
        footer.insertBefore(warningDiv, footer.firstChild);
    } else {
        // Add success message for real data
        const successDiv = document.createElement('div');
        successDiv.className = 'real-data-info';
        successDiv.style.margin = '10px 0';
        successDiv.innerHTML = `
            <i class="fas fa-check-circle"></i>
            Connected to ${dataSource} - Real market data active
        `;
        footer.insertBefore(successDiv, footer.firstChild);
    }
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

let lastDemoState = null;
function fetchTradingData() {
    fetch('/api/trading_state')
        .then(response => {
            if (!response.ok) throw new Error('Network error');
            return response.json();
        })
        .then(data => {
            // Check if data source changed
            if (lastDemoState !== null && lastDemoState !== data.is_demo_data) {
                if (data.is_demo_data) {
                    showToast('⚠️ Switched to Simulation Mode', 'warning');
                } else {
                    showToast('✅ Connected to Live Market Data', 'success');
                }
            }
            lastDemoState = data.is_demo_data;
            
            updateTradingData(data);
        })
        .catch(error => {
            console.error('Error fetching trading data:', error);
            document.getElementById('loading').style.display = 'none';
            
            // Show connection error
            const warningEl = document.getElementById('data-warning');
            if (warningEl) {
                warningEl.innerHTML = `
                    <div class="demo-warning">
                        <i class="fas fa-exclamation-triangle blink"></i>
                        ⚠️ CONNECTION ERROR
                    </div>
                    <div class="demo-info">
                        <i class="fas fa-wifi"></i>
                        Cannot connect to trading server. Retrying...
                    </div>
                `;
            }
            
            setTimeout(fetchTradingData, 5000);
        });
}

function showToast(message, type) {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-content">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
            ${message}
        </div>
    `;
    
    // Add to page
    document.body.appendChild(toast);
    
    // Show with animation
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }, 3000);
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

// Add CSS for toast notifications
const toastCSS = `
.toast {
    position: fixed;
    top: 20px;
    right: 20px;
    background: #333;
    color: white;
    padding: 15px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    z-index: 9999;
    transform: translateX(150%);
    transition: transform 0.3s ease;
    max-width: 350px;
}

.toast.show {
    transform: translateX(0);
}

.toast-success {
    background: linear-gradient(45deg, #4CAF50, #2E7D32);
    border-left: 4px solid #00ff88;
}

.toast-warning {
    background: linear-gradient(45deg, #ff9800, #ff5722);
    border-left: 4px solid #ffcc00;
}

.toast-content {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: bold;
}

.toast-content i {
    font-size: 1.2em;
}
`;

// Add toast styles to page
const styleSheet = document.createElement("style");
styleSheet.type = "text/css";
styleSheet.innerText = toastCSS;
document.head.appendChild(styleSheet);

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Start auto-refresh
    startAutoRefresh();
    
    // Set up manual trade buttons
    document.getElementById('manual-buy')?.addEventListener('click', () => executeManualTrade('buy'));
    document.getElementById('manual-sell')?.addEventListener('click', () => executeManualTrade('sell'));
    document.getElementById('reset-btn')?.addEventListener('click', resetTrading);
    
    // Test API connection on load
    fetch('/api/test_connection')
        .then(response => response.json())
        .then(data => {
            console.log('API Connection Test:', data);
            
            // Show API status
            const workingAPIs = data.results?.filter(r => r.status === 'WORKING').length || 0;
            if (workingAPIs > 0) {
                showToast(`✅ ${workingAPIs} Forex API(s) connected`, 'success');
            } else {
                showToast('⚠️ Using simulation data - API connections failed', 'warning');
            }
        })
        .catch(error => {
            console.error('API test failed:', error);
        });
    
    // Check health every minute
    setInterval(() => {
        fetch('/api/health')
            .then(response => response.json())
            .then(data => {
                console.log('System health:', data);
            });
    }, 60000);
});