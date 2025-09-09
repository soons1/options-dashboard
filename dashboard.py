import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from datetime import datetime, timedelta

st.set_page_config(page_title="Options Pricing Dashboard", layout="wide")

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = -norm.cdf(-d1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega = S*norm.pdf(d1)*np.sqrt(T)/100
    theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2 if option_type=='call' else -d2))/365
    rho = K*T*np.exp(-r*T)*norm.cdf(d2 if option_type=='call' else -d2)/100
    return price, delta, gamma, vega, theta, rho

def calculate_pnl(S_range, K, T, r, sigma, option_type, premium, position='long'):
    prices = [black_scholes(S, K, T, r, sigma, option_type)[0] for S in S_range]
    pnl = np.array(prices) - premium if position == 'long' else premium - np.array(prices)
    return pnl

st.title("📊 Black-Scholes Options Pricing Dashboard")

with st.sidebar:
    st.header("⚙️ Model Parameters")
    
    S = st.number_input("Current Stock Price ($)", min_value=1.0, value=100.0, step=0.5)
    K = st.number_input("Strike Price ($)", min_value=1.0, value=100.0, step=0.5)
    T = st.slider("Time to Expiry (Days)", min_value=1, max_value=365, value=30) / 365
    r = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.1) / 100
    sigma = st.slider("Volatility (%)", min_value=1.0, max_value=100.0, value=20.0, step=1.0) / 100
    
    st.divider()
    option_type = st.radio("Option Type", ['call', 'put'])
    position = st.radio("Position", ['long', 'short'])
    
    st.divider()
    st.header("📈 Heatmap Settings")
    heatmap_var = st.selectbox("Variable for Heatmap", ['Profit/Loss', 'Delta', 'Gamma', 'Theta', 'Vega'])

price, delta, gamma, vega, theta, rho = black_scholes(S, K, T, r, sigma, option_type)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💰 Option Pricing")
    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        st.metric("Option Price", f"${price:.2f}")
        st.metric("Delta", f"{delta:.4f}")
    with metrics_cols[1]:
        st.metric("Gamma", f"{gamma:.4f}")
        st.metric("Vega", f"{vega:.4f}")
    with metrics_cols[2]:
        st.metric("Theta", f"{theta:.4f}")
        st.metric("Rho", f"{rho:.4f}")
    
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["P&L Chart", "Heatmap", "Greeks Surface", "Payoff Diagram"])
    
    with tab1:
        S_range = np.linspace(S*0.5, S*1.5, 100)
        pnl = calculate_pnl(S_range, K, T, r, sigma, option_type, price, position)
        
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(x=S_range, y=pnl, mode='lines', name='P&L', 
                                      line=dict(color='green' if position=='long' else 'red', width=2)))
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_pnl.add_vline(x=S, line_dash="dash", line_color="blue", annotation_text="Current Price")
        fig_pnl.add_vline(x=K, line_dash="dash", line_color="orange", annotation_text="Strike")
        fig_pnl.update_layout(title="Profit/Loss vs Stock Price", xaxis_title="Stock Price ($)", 
                              yaxis_title="P&L ($)", height=400)
        st.plotly_chart(fig_pnl, use_container_width=True)
    
    with tab2:
        spot_range = np.linspace(S*0.7, S*1.3, 30)
        vol_range = np.linspace(sigma*0.5, sigma*1.5, 30)
        
        if heatmap_var == 'Profit/Loss':
            Z = [[calculate_pnl([s], K, T, r, v, option_type, price, position)[0] for s in spot_range] for v in vol_range]
            title = "P&L Heatmap"
        else:
            greek_map = {'Delta': 1, 'Gamma': 2, 'Vega': 3, 'Theta': 4}
            idx = greek_map[heatmap_var]
            Z = [[black_scholes(s, K, T, r, v, option_type)[idx] for s in spot_range] for v in vol_range]
            title = f"{heatmap_var} Heatmap"
        
        fig_heat = go.Figure(data=go.Heatmap(z=Z, x=spot_range, y=vol_range*100, colorscale='RdYlGn'))
        fig_heat.update_layout(title=title, xaxis_title="Stock Price ($)", 
                               yaxis_title="Volatility (%)", height=400)
        st.plotly_chart(fig_heat, use_container_width=True)
    
    with tab3:
        spot_range_3d = np.linspace(S*0.7, S*1.3, 20)
        time_range = np.linspace(0.01, T, 20)
        
        X, Y = np.meshgrid(spot_range_3d, time_range)
        Z_surface = np.array([[black_scholes(s, K, t, r, sigma, option_type)[1] for s in spot_range_3d] for t in time_range])
        
        fig_3d = go.Figure(data=[go.Surface(x=X, y=Y*365, z=Z_surface, colorscale='Viridis')])
        fig_3d.update_layout(title="Delta Surface", scene=dict(xaxis_title="Stock Price ($)", 
                            yaxis_title="Days to Expiry", zaxis_title="Delta"), height=500)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with tab4:
        S_payoff = np.linspace(S*0.5, S*1.5, 100)
        intrinsic = np.maximum(S_payoff - K, 0) if option_type == 'call' else np.maximum(K - S_payoff, 0)
        payoff = intrinsic - price if position == 'long' else price - intrinsic
        
        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Scatter(x=S_payoff, y=payoff, mode='lines', name='At Expiry', 
                                        line=dict(color='blue', width=2)))
        fig_payoff.add_trace(go.Scatter(x=S_payoff, y=calculate_pnl(S_payoff, K, T, r, sigma, option_type, price, position),
                                        mode='lines', name='Current', line=dict(color='green', width=2, dash='dash')))
        fig_payoff.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_payoff.add_vline(x=K, line_dash="dash", line_color="orange", annotation_text="Strike")
        fig_payoff.update_layout(title="Payoff Diagram", xaxis_title="Stock Price ($)", 
                                 yaxis_title="Profit/Loss ($)", height=400)
        st.plotly_chart(fig_payoff, use_container_width=True)

with col2:
    st.subheader("📊 Strategy Builder")
    
    num_legs = st.number_input("Number of Legs", min_value=1, max_value=4, value=1)
    
    strategy_legs = []
    for i in range(num_legs):
        with st.expander(f"Leg {i+1}", expanded=True):
            leg_type = st.selectbox(f"Type", ['call', 'put'], key=f"type_{i}")
            leg_position = st.selectbox(f"Position", ['long', 'short'], key=f"pos_{i}")
            leg_strike = st.number_input(f"Strike", value=K, key=f"strike_{i}")
            leg_qty = st.number_input(f"Quantity", value=1, min_value=1, key=f"qty_{i}")
            strategy_legs.append({'type': leg_type, 'position': leg_position, 'strike': leg_strike, 'qty': leg_qty})
    
    if st.button("Calculate Strategy"):
        S_strategy = np.linspace(S*0.5, S*1.5, 100)
        total_pnl = np.zeros(100)
        
        for leg in strategy_legs:
            leg_price = black_scholes(S, leg['strike'], T, r, sigma, leg['type'])[0]
            leg_pnl = calculate_pnl(S_strategy, leg['strike'], T, r, sigma, leg['type'], leg_price, leg['position'])
            total_pnl += leg_pnl * leg['qty']
        
        fig_strategy = go.Figure()
        fig_strategy.add_trace(go.Scatter(x=S_strategy, y=total_pnl, mode='lines', name='Strategy P&L'))
        fig_strategy.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_strategy.update_layout(title="Strategy P&L", xaxis_title="Stock Price ($)", 
                                   yaxis_title="P&L ($)", height=300)
        st.plotly_chart(fig_strategy, use_container_width=True)
    
    st.divider()
    st.subheader("📈 Implied Volatility Calculator")
    
    market_price = st.number_input("Market Price ($)", min_value=0.01, value=price, step=0.01)
    
    if st.button("Calculate IV"):
        def objective(vol):
            return abs(black_scholes(S, K, T, r, vol, option_type)[0] - market_price)
        
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(objective, bounds=(0.001, 3), method='bounded')
        iv = result.x * 100
        
        st.success(f"Implied Volatility: {iv:.2f}%")
        st.info(f"Model Price at IV: ${black_scholes(S, K, T, r, result.x, option_type)[0]:.2f}")