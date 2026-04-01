import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="TN Housing Price Predictor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
* { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f172a 0%, #0c1a2e 100%); }
h1,h2,h3,p,label,div { color: #e2e8f0; }
.stButton > button {
  background: linear-gradient(135deg, #2563eb, #06b6d4) !important;
  color: white !important; border: none !important;
  border-radius: 10px !important; padding: 0.6rem 2rem !important;
  font-weight: 600 !important; width: 100% !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }
.stSlider label, .stSelectbox label { color: #94a3b8 !important; }
.stSelectbox > div > div {
  background: rgba(30,41,59,0.8) !important;
  color: #e2e8f0 !important;
  border: 1px solid rgba(6,182,212,0.3) !important;
  border-radius: 8px !important;
}
div[data-testid="metric-container"] {
  background: rgba(30,41,59,0.8);
  border: 1px solid rgba(6,182,212,0.2);
  border-radius: 12px; padding: 1rem;
}
.pred-box {
  background: linear-gradient(135deg, rgba(37,99,235,0.2), rgba(6,182,212,0.2));
  border: 2px solid rgba(6,182,212,0.5);
  border-radius: 16px; padding: 2rem;
  text-align: center; margin: 1rem 0;
}
.city-badge {
  display: inline-block;
  background: rgba(37,99,235,0.2);
  border: 1px solid rgba(37,99,235,0.4);
  border-radius: 20px;
  padding: 0.2rem 0.8rem;
  font-size: 0.85rem;
  color: #93c5fd;
  margin: 0.2rem;
}
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; }
.stTabs [aria-selected="true"] { color: #06b6d4 !important; border-bottom: 3px solid #06b6d4 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAMIL NADU CITY DATA — Real market prices
# ══════════════════════════════════════════
TN_CITIES = {
    "Chennai":       {"base": 8500,  "premium": 1.8, "areas": ["Anna Nagar","T Nagar","Velachery","OMR","Adyar","Tambaram","Porur","Chrompet","Sholinganallur","Perambur"]},
    "Coimbatore":    {"base": 4500,  "premium": 1.3, "areas": ["RS Puram","Gandhipuram","Peelamedu","Saravanampatti","Singanallur","Vadavalli","Race Course","Hopes College"]},
    "Trichy":        {"base": 3200,  "premium": 1.1, "areas": ["Thillai Nagar","Woraiyur","Srirangam","KK Nagar","Ariyamangalam","Puthur","Tennur","Cantonment"]},
    "Madurai":       {"base": 3000,  "premium": 1.1, "areas": ["Anna Nagar","KK Nagar","Goripalayam","Tallakulam","Villapuram","Palanganatham","Bypass Road"]},
    "Salem":         {"base": 2800,  "premium": 1.0, "areas": ["Fairlands","Suramangalam","Hasthampatti","Alagapuram","Kondalampatti","Five Roads"]},
    "Tirunelveli":   {"base": 2500,  "premium": 1.0, "areas": ["Palayamkottai","Melapalayam","Vannarpet","Junction","Pettai","Nanguneri Road"]},
    "Vellore":       {"base": 2600,  "premium": 1.0, "areas": ["Katpadi","Sathuvachari","Gandhi Nagar","Officers Line","Bagayam"]},
    "Erode":         {"base": 2700,  "premium": 1.0, "areas": ["Gandhiji Road","Cauvery Nagar","Perundurai","Bargur","Chithode"]},
    "Tiruppur":      {"base": 3000,  "premium": 1.1, "areas": ["Avinashi Road","Kumaran Nagar","Rayapuram","Palladam Road","Kangeyam Road"]},
    "Kanchipuram":   {"base": 2900,  "premium": 1.0, "areas": ["Gandhi Nagar","Kamatchi Amman Koil","Pillaiyar Palayam","Bus Stand Area"]},
    "Thanjavur":     {"base": 2400,  "premium": 0.9, "areas": ["Medical College Road","Nanjikottai Road","Pettavaithalai","Old Bus Stand"]},
    "Dindigul":      {"base": 2200,  "premium": 0.9, "areas": ["Palani Road","Batlagundu","Natham","Nehru Nagar"]},
    "Hosur":         {"base": 4000,  "premium": 1.2, "areas": ["Sipcot","Mathigiri","Thally Road","Denkanikottai Road","Rayakottah Road"]},
    "Nagercoil":     {"base": 2600,  "premium": 1.0, "areas": ["KK Road","Kottar","Colachel","Marthandam","Thuckalay"]},
    "Cuddalore":     {"base": 2000,  "premium": 0.9, "areas": ["SIPCOT","Panruti","Virudhachalam","Bus Stand Area"]},
}

PROPERTY_TYPES = {
    "Apartment / Flat": 1.0,
    "Independent House": 1.15,
    "Villa": 1.4,
    "Plot / Land": 0.7,
    "Row House": 1.1,
}

AMENITIES = {
    "🚗 Car Parking": 150000,
    "🏊 Swimming Pool": 300000,
    "🏋️ Gym": 200000,
    "🔒 24hr Security": 100000,
    "🌳 Garden/Park": 150000,
    "🏫 Near School": 200000,
    "🏥 Near Hospital": 180000,
    "🚇 Metro/Bus Access": 250000,
    "🏬 Near Shopping Mall": 150000,
    "💧 24hr Water Supply": 100000,
}

# ── Generate realistic TN housing data ──
@st.cache_data
def generate_tn_data():
    np.random.seed(42)
    rows = []
    for city, info in TN_CITIES.items():
        n = 80
        area_sqft  = np.random.randint(400, 3500, n)
        bedrooms   = np.random.choice([1,2,3,4,5], n, p=[0.1,0.3,0.35,0.2,0.05])
        bathrooms  = np.clip(bedrooms - np.random.randint(0,2,n), 1, 4)
        age        = np.random.randint(0, 35, n)
        floor      = np.random.randint(0, 20, n)
        amenity_cnt= np.random.randint(0, 8, n)
        prop_multi = np.random.choice([1.0, 1.15, 1.4, 0.7, 1.1], n)
        noise      = np.random.normal(0, 80000, n)

        base = info["base"]
        price = (
            area_sqft * base * info["premium"] * prop_multi +
            bedrooms * 180000 +
            bathrooms * 90000 +
            amenity_cnt * 120000 -
            age * 12000 +
            floor * 15000 +
            noise
        )
        price = np.clip(price, 500000, 30000000)

        for i in range(n):
            rows.append({
                "city": city,
                "area_sqft": area_sqft[i],
                "bedrooms": bedrooms[i],
                "bathrooms": int(bathrooms[i]),
                "age": age[i],
                "floor": floor[i],
                "amenities": amenity_cnt[i],
                "prop_multi": prop_multi[i],
                "base_rate": base * info["premium"],
                "price": price[i]
            })
    return pd.DataFrame(rows)

@st.cache_resource
def train_models():
    df = generate_tn_data()
    X = df[["area_sqft","bedrooms","bathrooms","age","floor","amenities","prop_multi","base_rate"]]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf  = RandomForestRegressor(n_estimators=200, random_state=42)
    gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
    lr  = LinearRegression()
    rf.fit(X_train, y_train)
    gbr.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    return rf, gbr, lr, \
           r2_score(y_test, rf.predict(X_test)), \
           r2_score(y_test, gbr.predict(X_test)), \
           r2_score(y_test, lr.predict(X_test)), \
           X_test, y_test

rf_model, gbr_model, lr_model, rf_r2, gbr_r2, lr_r2, X_test, y_test = train_models()

# ══════════════════════════════════════════
# UI
# ══════════════════════════════════════════
st.markdown("<h1 style='text-align:center; background:linear-gradient(135deg,#2563eb,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>🏠 Tamil Nadu Housing Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94a3b8;margin-bottom:1.5rem;'>Chennai, Coimbatore, Trichy, Madurai உட்பட 15 Tamil Nadu cities-க்கு house price predict பண்றோம்</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Price Predictor", "📊 Market Analysis", "🏙️ City Comparison"])

# ══════════ TAB 1: PREDICTOR ══════════
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📍 Location Details")
        city = st.selectbox("City தேர்வு பண்ணு", list(TN_CITIES.keys()))
        area_list = TN_CITIES[city]["areas"]
        locality = st.selectbox("Locality / Area", area_list)
        prop_type = st.selectbox("Property Type", list(PROPERTY_TYPES.keys()))

        st.markdown("### 🏡 Property Details")
        area_sqft = st.slider("📐 Area (Square Feet)", 300, 4000, 1200, 50)
        c1, c2 = st.columns(2)
        with c1: bedrooms  = st.selectbox("🛏️ Bedrooms", [1,2,3,4,5], index=1)
        with c2: bathrooms = st.selectbox("🚿 Bathrooms", [1,2,3,4], index=1)
        age   = st.slider("🏗️ Property Age (Years)", 0, 35, 3)
        floor = st.slider("🏢 Floor Number", 0, 25, 2)

    with col_right:
        st.markdown("### ✨ Amenities Select பண்ணு")
        selected_amenities = []
        amenity_value = 0
        cols = st.columns(2)
        for i, (amenity, value) in enumerate(AMENITIES.items()):
            with cols[i % 2]:
                if st.checkbox(amenity, key=amenity):
                    selected_amenities.append(amenity)
                    amenity_value += value

        st.markdown(f"<p style='color:#06b6d4; margin-top:0.5rem;'>✅ {len(selected_amenities)} amenities selected (+₹{amenity_value/100000:.1f}L value)</p>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔮 Price Predict பண்ணு!", use_container_width=True):
        city_info  = TN_CITIES[city]
        base_rate  = city_info["base"] * city_info["premium"]
        prop_multi = PROPERTY_TYPES[prop_type]
        amenity_cnt= len(selected_amenities)

        inp = pd.DataFrame([[area_sqft, bedrooms, bathrooms, age, floor, amenity_cnt, prop_multi, base_rate]],
              columns=["area_sqft","bedrooms","bathrooms","age","floor","amenities","prop_multi","base_rate"])

        rf_pred  = rf_model.predict(inp)[0] + amenity_value
        gbr_pred = gbr_model.predict(inp)[0] + amenity_value
        lr_pred  = lr_model.predict(inp)[0] + amenity_value
        best     = rf_pred  # RF is best

        def fmt(v): return f"₹{v/100000:.2f}L" if v < 10000000 else f"₹{v/10000000:.2f}Cr"

        st.markdown(f"""
        <div class='pred-box'>
          <p style='color:#94a3b8; font-size:0.9rem; margin-bottom:0.5rem;'>
            📍 {locality}, {city} | {prop_type} | {area_sqft} sqft | {bedrooms}BHK
          </p>
          <h1 style='color:#06b6d4; font-size:3rem; margin:0;'>{fmt(best)}</h1>
          <p style='color:#94a3b8; margin-top:0.5rem;'>Estimated Market Price</p>
        </div>
        """, unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("🌲 Random Forest", fmt(rf_pred), f"R²={rf_r2:.3f}")
        c2.metric("📈 Gradient Boost", fmt(gbr_pred), f"R²={gbr_r2:.3f}")
        c3.metric("📉 Linear Regression", fmt(lr_pred), f"R²={lr_r2:.3f}")

        # Price breakdown
        st.markdown("#### 💰 Price Breakdown")
        breakdown = {
            "Base Land Value":    area_sqft * base_rate * prop_multi,
            "Bedroom Premium":    bedrooms * 180000,
            "Bathroom Value":     bathrooms * 90000,
            "Amenities Value":    amenity_value,
            "Floor Premium":      floor * 15000,
            "Age Deduction":      -(age * 12000),
        }
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        colors = ['#2563eb','#06b6d4','#f59e0b','#22c55e','#a855f7','#ef4444']
        vals = list(breakdown.values())
        labels = list(breakdown.keys())
        ax.barh(labels, [abs(v) for v in vals], color=[colors[i] for i in range(len(vals))])
        ax.set_title(f"Price Factors — {locality}, {city}", color='#e2e8f0', pad=10)
        ax.tick_params(colors='#94a3b8', labelsize=9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'₹{x/100000:.0f}L'))
        for spine in ax.spines.values(): spine.set_color('#334155')
        st.pyplot(fig)
        plt.close()

        # Price range
        low  = best * 0.9
        high = best * 1.1
        st.info(f"📊 Market Range: **{fmt(low)}** — **{fmt(high)}** (±10% variation)")

# ══════════ TAB 2: MARKET ANALYSIS ══════════
with tab2:
    st.markdown("### 📊 Tamil Nadu Real Estate Market Overview")
    df = generate_tn_data()

    city_avg = df.groupby("city")["price"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    colors_bar = ['#2563eb' if c == 'Chennai' else '#06b6d4' if c in ['Coimbatore','Hosur'] else '#3b82f6' for c in city_avg.index]
    bars = ax.barh(city_avg.index, city_avg.values/100000, color=colors_bar)
    ax.set_xlabel("Average Price (₹ Lakhs)", color='#94a3b8')
    ax.set_title("Average House Price by City — Tamil Nadu", color='#e2e8f0', pad=15)
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values(): spine.set_color('#334155')
    for bar, val in zip(bars, city_avg.values):
        ax.text(val/100000 + 1, bar.get_y() + bar.get_height()/2,
                f'₹{val/100000:.0f}L', va='center', color='#e2e8f0', fontsize=8)
    st.pyplot(fig)
    plt.close()

    st.markdown("### 📐 Price per Sqft — City wise")
    city_psf = {}
    for city_name, info in TN_CITIES.items():
        city_psf[city_name] = info["base"] * info["premium"]
    psf_df = pd.DataFrame(list(city_psf.items()), columns=["City","Price/sqft (₹)"])
    psf_df = psf_df.sort_values("Price/sqft (₹)", ascending=False)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.patch.set_facecolor('#0f172a')
    ax2.set_facecolor('#1e293b')
    ax2.bar(psf_df["City"], psf_df["Price/sqft (₹)"], color='#06b6d4', alpha=0.8)
    ax2.set_ylabel("Price per Sqft (₹)", color='#94a3b8')
    ax2.set_title("Price per Square Foot by City", color='#e2e8f0')
    ax2.tick_params(axis='x', rotation=45, colors='#94a3b8')
    ax2.tick_params(axis='y', colors='#94a3b8')
    for spine in ax2.spines.values(): spine.set_color('#334155')
    st.pyplot(fig2)
    plt.close()

# ══════════ TAB 3: CITY COMPARISON ══════════
with tab3:
    st.markdown("### 🏙️ City-wise Price Comparison")
    st.markdown("#### உன்னோட budget-க்கு எந்த city-ல வாங்கலாம்?")

    budget = st.slider("💰 உன்னோட Budget (₹ Lakhs)", 10, 500, 60, 5)
    budget_val = budget * 100000

    bhk = st.selectbox("🛏️ BHK Preference", [1, 2, 3, 4], index=1)

    st.markdown(f"#### ₹{budget}L Budget-ல **{bhk}BHK** வாங்க முடியும்:")

    results = []
    for city_name, info in TN_CITIES.items():
        base_rate = info["base"] * info["premium"]
        # Estimate area possible
        approx_cost_per_sqft = base_rate
        fixed_costs = bhk * 180000 + (bhk-1) * 90000
        available_for_area = budget_val - fixed_costs
        possible_sqft = max(0, available_for_area / approx_cost_per_sqft)
        results.append({
            "City": city_name,
            "Possible Area": f"{possible_sqft:.0f} sqft",
            "Price/sqft": f"₹{base_rate:.0f}",
            "Feasibility": "✅ Good" if possible_sqft > 800 else "⚠️ Tight" if possible_sqft > 400 else "❌ Low"
        })

    res_df = pd.DataFrame(results).sort_values("Possible Area", ascending=False)
    st.dataframe(res_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🏙️ Cities in Tamil Nadu")
    for city_name in TN_CITIES:
        st.markdown(f"<span class='city-badge'>📍 {city_name}</span>", unsafe_allow_html=True)

st.markdown("<br><p style='text-align:center;color:#475569;font-size:0.8rem;'>Antony Ruffina N | MCA — St. Joseph's College, Trichy | Housing Price Prediction Project</p>", unsafe_allow_html=True)
