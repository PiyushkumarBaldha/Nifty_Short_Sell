import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm

# --- Page Config ---
st.set_page_config(page_title="Advanced Nifty Simulator", layout="wide")

st.title("🚀 Advanced Nifty Strategy Simulator (MTM & Expiry)")
st.markdown("""
This tool calculates your **Mark-to-Market (MTM) Profit/Loss** if you exit *before* expiry.
It uses the **Black-Scholes Model** to estimate future premiums.
""")

# --- Sidebar: Entry Inputs ---
st.sidebar.header("1. Entry Details (When you took the trade)")

# Market Data at Entry
entry_spot = st.sidebar.number_input("Nifty Spot Price (at Entry)", value=25088, step=10)
days_to_expiry_total = st.sidebar.number_input("Days to Expiry (Total)", value=5, step=1)
implied_volatility = st.sidebar.number_input("Implied Volatility (IV %)", value=14.0, step=0.1) / 100
risk_free_rate = 0.10  # 10% Risk Free Rate for India

st.sidebar.markdown("---")
st.sidebar.header("2. Your Positions")
lot_size = st.sidebar.number_input("Lot Size", value=65)

# Short Strangle Inputs
c_strike = st.sidebar.number_input("Sell Call Strike", value=25700)
c_entry_prem = st.sidebar.number_input("Sell Call Premium (Entry)", value=50.0)

p_strike = st.sidebar.number_input("Sell Put Strike", value=24500)
p_entry_prem = st.sidebar.number_input("Sell Put Premium (Entry)", value=55.0)

# Hedge Inputs
use_hedge = st.sidebar.checkbox("Active Iron Condor (Hedges)", value=False)
if use_hedge:
    bc_strike = st.sidebar.number_input("Buy Call Strike", value=26000)
    bc_entry_prem = st.sidebar.number_input("Buy Call Premium", value=8.0)
    bp_strike = st.sidebar.number_input("Buy Put Strike", value=24200)
    bp_entry_prem = st.sidebar.number_input("Buy Put Premium", value=10.0)

# --- BLACK-SCHOLES FUNCTION ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    S: Spot Price
    K: Strike Price
    T: Time to Maturity (in years)
    r: Risk-free rate
    sigma: Volatility
    """
    if T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

# --- Main Simulation Area ---
st.header("🔮 Scenario Simulator (What if?)")

col_sim1, col_sim2, col_sim3 = st.columns(3)

with col_sim1:
    sim_spot = st.number_input("Future Nifty Spot Price", value=entry_spot, step=10)
    # Calculate % Change
    pct_change = ((sim_spot - entry_spot) / entry_spot) * 100
    color_pct = "green" if pct_change >= 0 else "red"
    st.markdown(f"Nifty Change: :{color_pct}[**{pct_change:.2f}%**]")

with col_sim2:
    days_passed = st.slider("Days Passed", 0, days_to_expiry_total, 1)
    days_remaining = days_to_expiry_total - days_passed
    st.write(f"Days Remaining: **{days_remaining}**")

with col_sim3:
    sim_iv_change = st.slider("Change in Volatility (IV)", -5.0, 5.0, 0.0, 0.5)
    current_iv = implied_volatility + (sim_iv_change/100)
    st.write(f"New IV: **{current_iv*100:.1f}%**")

# --- Calculations ---
T_remaining = max(0.001, days_remaining / 365.0) # Avoid division by zero

# Calculate Estimated Current Premiums (Exit Price)
est_c_price = black_scholes(sim_spot, c_strike, T_remaining, risk_free_rate, current_iv, 'call')
est_p_price = black_scholes(sim_spot, p_strike, T_remaining, risk_free_rate, current_iv, 'put')

# Calculate P/L for Sell Legs
# Profit = (Entry Price - Current Price) * Lot
pl_call = (c_entry_prem - est_c_price) * lot_size
pl_put = (p_entry_prem - est_p_price) * lot_size

# Calculate P/L for Buy Legs (if any)
pl_hedge = 0
est_bc_price = 0
est_bp_price = 0

if use_hedge:
    est_bc_price = black_scholes(sim_spot, bc_strike, T_remaining, risk_free_rate, current_iv, 'call')
    est_bp_price = black_scholes(sim_spot, bp_strike, T_remaining, risk_free_rate, current_iv, 'put')
    
    # Profit for bought legs = (Current Price - Entry Price)
    pl_bc = (est_bc_price - bc_entry_prem) * lot_size
    pl_bp = (est_bp_price - bp_entry_prem) * lot_size
    pl_hedge = pl_bc + pl_bp

total_mtm = pl_call + pl_put + pl_hedge

# --- Display Results ---
st.divider()
st.subheader("💰 Simulated P/L Report")

res_col1, res_col2, res_col3 = st.columns(3)

with res_col1:
    st.metric(label="Net MTM Profit/Loss", value=f"₹ {total_mtm:,.2f}", delta=f"{total_mtm:.2f}")

with res_col2:
    st.write("#### 📉 Call Leg Status")
    st.write(f"Strike: **{c_strike} CE**")
    st.write(f"Entry: ₹{c_entry_prem:.2f} ➝ New: **₹{est_c_price:.2f}**")
    st.write(f"P/L: ₹{pl_call:,.2f}")

with res_col3:
    st.write("#### 📉 Put Leg Status")
    st.write(f"Strike: **{p_strike} PE**")
    st.write(f"Entry: ₹{p_entry_prem:.2f} ➝ New: **₹{est_p_price:.2f}**")
    st.write(f"P/L: ₹{pl_put:,.2f}")

if use_hedge:
    st.info(f"🛡️ **Hedge Performance:** Your protection legs (Iron Condor) are contributing: **₹ {pl_hedge:,.2f}**")

# --- Warning Logic ---
st.divider()
if days_remaining > 0:
    st.warning("⚠️ **Note:** These are *estimated* premiums based on the Black-Scholes model. Real market prices may differ slightly due to supply/demand.")
else:
    st.success("✅ **Expiry Day:** Prices are calculated based on Intrinsic Value (Settlement Price).")

# --- Charting the Risk Curve ---
# Generate data points for range
x_range = np.linspace(entry_spot * 0.95, entry_spot * 1.05, 50)
y_mtm = []

for x in x_range:
    # Estimate prices at specific spot x
    c_p = black_scholes(x, c_strike, T_remaining, risk_free_rate, current_iv, 'call')
    p_p = black_scholes(x, p_strike, T_remaining, risk_free_rate, current_iv, 'put')
    
    val = (c_entry_prem - c_p) * lot_size + (p_entry_prem - p_p) * lot_size
    
    if use_hedge:
        bc_p = black_scholes(x, bc_strike, T_remaining, risk_free_rate, current_iv, 'call')
        bp_p = black_scholes(x, bp_strike, T_remaining, risk_free_rate, current_iv, 'put')
        val += (bc_p - bc_entry_prem) * lot_size + (bp_p - bp_entry_prem) * lot_size
    
    y_mtm.append(val)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_range, y=y_mtm, mode='lines', name='Projected P/L Line', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=[sim_spot], y=[total_mtm], mode='markers', name='Current Sim', marker=dict(color='red', size=12)))
fig.add_hline(y=0, line_dash="dash", line_color="black")

fig.update_layout(title="Projected P/L vs Nifty Spot", xaxis_title="Nifty Spot", yaxis_title="Profit/Loss (₹)")
st.plotly_chart(fig, use_container_width=True)