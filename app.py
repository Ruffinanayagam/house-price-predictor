import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="TN Housing Price Predictor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
* { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 100%); }
h1,h2,h3,p,label,div { color: #f1f5f9; }

.stButton > button {
  background: linear-gradient(135deg, #f59e0b, #eab308) !important;
  color: #0a0a1a !important; border: none !important;
  border-radius: 10px !important; padding: 0.65rem 2rem !important;
  font-weight: 700 !important; width: 100% !important;
  font-size: 1rem !important; letter-spacing: 0.5px !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(245,158,11,0.4) !important; }

.stSelectbox > div > div {
  background: rgba(15,30,55,0.9) !important; color: #f1f5f9 !important;
  border: 1px solid rgba(245,158,11,0.4) !important; border-radius: 8px !important;
}
.stSlider label { color: #94a3b8 !important; }
div[data-testid="metric-container"] {
  background: rgba(15,30,55,0.9); border: 1px solid rgba(245,158,11,0.3);
  border-radius: 12px; padding: 1rem;
}
div[data-testid="metric-container"] label { color: #94a3b8 !important; }
div[data-testid="metric-container"] div { color: #f59e0b !important; }
.stCheckbox label { color: #f1f5f9 !important; }
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #f59e0b !important; border-bottom: 3px solid #f59e0b !important; }
.stProgress > div > div { background: linear-gradient(90deg, #1d4ed8, #f59e0b) !important; border-radius: 10px; }
.pred-box {
  background: linear-gradient(135deg, rgba(29,78,216,0.25), rgba(245,158,11,0.15));
  border: 2px solid rgba(245,158,11,0.5); border-radius: 16px;
  padding: 2rem; text-align: center; margin: 1rem 0;
}
.section-header {
  color: #f59e0b !important; font-weight: 700;
  border-bottom: 2px solid rgba(245,158,11,0.3);
  padding-bottom: 0.4rem; margin-bottom: 1rem;
}
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── City Data ──
TN_CITIES = {
    "Chennai":     {"base": 8500, "premium": 1.8, "areas": ["Anna Nagar","T Nagar","Velachery","OMR","Adyar","Tambaram","Porur","Chrompet","Sholinganallur","Perambur","Kodambakkam","Nungambakkam"]},
    "Coimbatore":  {"base": 4500, "premium": 1.3, "areas": ["RS Puram","Gandhipuram","Peelamedu","Saravanampatti","Singanallur","Vadavalli","Race Course","Hopes College","Ondipudur"]},
    "Trichy":      {"base": 3200, "premium": 1.1, "areas": ["Thillai Nagar","Woraiyur","Srirangam","KK Nagar","Ariyamangalam","Puthur","Tennur","Cantonment","Karumandapam"]},
    "Madurai":     {"base": 3000, "premium": 1.1, "areas": ["Anna Nagar","KK Nagar","Goripalayam","Tallakulam","Villapuram","Palanganatham","Bypass Road","Mattuthavani"]},
    "Salem":       {"base": 2800, "premium": 1.0, "areas": ["Fairlands","Suramangalam","Hasthampatti","Alagapuram","Kondalampatti","Five Roads","Swarnapuri"]},
    "Tirunelveli": {"base": 2500, "premium": 1.0, "areas": ["Palayamkottai","Melapalayam","Vannarpet","Junction","Pettai","Nanguneri Road"]},
    "Vellore":     {"base": 2600, "premium": 1.0, "areas": ["Katpadi","Sathuvachari","Gandhi Nagar","Officers Line","Bagayam","CMC Road"]},
    "Erode":       {"base": 2700, "premium": 1.0, "areas": ["Gandhiji Road","Cauvery Nagar","Perundurai","Bargur","Chithode"]},
    "Tiruppur":    {"base": 3000, "premium": 1.1, "areas": ["Avinashi Road","Kumaran Nagar","Rayapuram","Palladam Road","Kangeyam Road"]},
    "Hosur":       {"base": 4000, "premium": 1.2, "areas": ["Sipcot","Mathigiri","Thally Road","Denkanikottai Road","Rayakottah Road"]},
    "Thanjavur":   {"base": 2400, "premium": 0.9, "areas": ["Medical College Road","Nanjikottai Road","Pettavaithalai","Old Bus Stand","Vallam Road"]},
    "Dindigul":    {"base": 2200, "premium": 0.9, "areas": ["Palani Road","Batlagundu","Natham","Nehru Nagar","Gandhiji Road"]},
    "Kanchipuram": {"base": 2900, "premium": 1.0, "areas": ["Gandhi Nagar","Pillaiyar Palayam","Bus Stand Area","Uthiramerur Road"]},
    "Nagercoil":   {"base": 2600, "premium": 1.0, "areas": ["KK Road","Kottar","Colachel","Marthandam","Thuckalay"]},
    "Cuddalore":   {"base": 2000, "premium": 0.9, "areas": ["SIPCOT","Panruti","Virudhachalam","Bus Stand Area","Neyveli Road"]},
}

PROPERTY_TYPES = {
    "Apartment / Flat": 1.0,
    "Independent House": 1.15,
    "Villa": 1.4,
    "Plot / Land": 0.7,
    "Row House": 1.1,
}

AMENITIES = {
    "Car Parking": 150000,
    "Swimming Pool": 300000,
    "Gym / Fitness Center": 200000,
    "24hr Security": 100000,
    "Garden / Park": 150000,
    "Near School (< 1km)": 200000,
    "Near Hospital (< 1km)": 180000,
    "Metro / Bus Access": 250000,
    "Near Shopping Mall": 150000,
    "24hr Water Supply": 100000,
    "Power Backup": 120000,
    "Lift / Elevator": 180000,
}

@st.cache_data
def generate_data():
    np.random.seed(42)
    rows = []
    for city, info in TN_CITIES.items():
        n = 100
        area       = np.random.randint(400, 4000, n)
        bedrooms   = np.random.choice([1,2,3,4,5], n, p=[0.1,0.3,0.35,0.2,0.05])
        bathrooms  = np.clip(bedrooms - np.random.randint(0,2,n), 1, 4)
        age        = np.random.randint(0, 35, n)
        floor      = np.random.randint(0, 20, n)
        amenity_cnt= np.random.randint(0, 8, n)
        prop_multi = np.random.choice([1.0,1.15,1.4,0.7,1.1], n)
        noise      = np.random.normal(0, 80000, n)
        base_rate  = info["base"] * info["premium"]
        price = (area * base_rate * prop_multi + bedrooms*180000 +
                 bathrooms*90000 + amenity_cnt*120000 -
                 age*12000 + floor*15000 + noise)
        price = np.clip(price, 500000, 50000000)
        for i in range(n):
            rows.append({"area":area[i],"bedrooms":bedrooms[i],"bathrooms":int(bathrooms[i]),
                         "age":age[i],"floor":floor[i],"amenities":amenity_cnt[i],
                         "prop_multi":prop_multi[i],"base_rate":base_rate,"price":price[i]})
    return pd.DataFrame(rows)

@st.cache_resource
def train():
    df = generate_data()
    X = df[["area","bedrooms","bathrooms","age","floor","amenities","prop_multi","base_rate"]]
    y = df["price"]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    rf  = RandomForestRegressor(n_estimators=200, random_state=42)
    gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
    lr  = LinearRegression()
    rf.fit(Xtr,ytr); gbr.fit(Xtr,ytr); lr.fit(Xtr,ytr)
    return rf,gbr,lr, r2_score(yte,rf.predict(Xte)), r2_score(yte,gbr.predict(Xte)), r2_score(yte,lr.predict(Xte)), Xte, yte

rf,gbr,lr,rf_r2,gbr_r2,lr_r2,Xte,yte = train()

def fmt(v):
    return f"₹{v/100000:.2f}L" if v < 10000000 else f"₹{v/10000000:.2f}Cr"

# ── Header ──
st.markdown("<h1 style='text-align:center;background:linear-gradient(135deg,#1d4ed8,#f59e0b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.4rem;'>🏠 Tamil Nadu Housing Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94a3b8;'>Covering 15 major Tamil Nadu cities — Chennai, Coimbatore, Trichy, Madurai & more</p>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["  🔮 Price Predictor  ", "  📊 Market Analysis  ", "  💰 Budget Planner  "])

# ═══════════ TAB 1 ═══════════
with tab1:
    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        st.markdown("<h3 class='section-header'>📍 Location</h3>", unsafe_allow_html=True)
        city     = st.selectbox("Select City", list(TN_CITIES.keys()))
        locality = st.selectbox("Select Locality / Area", TN_CITIES[city]["areas"])
        prop_type= st.selectbox("Property Type", list(PROPERTY_TYPES.keys()))

        st.markdown("<h3 class='section-header'>🏡 Property Details</h3>", unsafe_allow_html=True)
        area_sqft = st.slider("Area (Square Feet)", 300, 5000, 1200, 50)
        c1,c2 = st.columns(2)
        with c1: bedrooms  = st.selectbox("Bedrooms", [1,2,3,4,5], index=1)
        with c2: bathrooms = st.selectbox("Bathrooms", [1,2,3,4], index=1)
        age   = st.slider("Property Age (Years)", 0, 40, 3)
        floor = st.slider("Floor Number (0 = Ground)", 0, 30, 2)

    with col2:
        st.markdown("<h3 class='section-header'>✨ Amenities</h3>", unsafe_allow_html=True)
        sel_amenities = []
        amenity_val   = 0
        c1,c2 = st.columns(2)
        for i,(k,v) in enumerate(AMENITIES.items()):
            with (c1 if i%2==0 else c2):
                if st.checkbox(k, key=f"am_{k}"):
                    sel_amenities.append(k)
                    amenity_val += v
        st.markdown(f"<p style='color:#f59e0b;margin-top:0.5rem;'>✅ {len(sel_amenities)} amenities selected &nbsp;|&nbsp; Value: +{fmt(amenity_val)}</p>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔮 Predict House Price"):
        ci        = TN_CITIES[city]
        base_rate = ci["base"] * ci["premium"]
        pm        = PROPERTY_TYPES[prop_type]
        inp = pd.DataFrame([[area_sqft,bedrooms,bathrooms,age,floor,len(sel_amenities),pm,base_rate]],
              columns=["area","bedrooms","bathrooms","age","floor","amenities","prop_multi","base_rate"])
        rf_p  = rf.predict(inp)[0]  + amenity_val
        gbr_p = gbr.predict(inp)[0] + amenity_val
        lr_p  = lr.predict(inp)[0]  + amenity_val

        st.markdown(f"""
        <div class='pred-box'>
          <p style='color:#94a3b8;font-size:0.9rem;margin-bottom:0.5rem;'>
            📍 {locality}, {city} &nbsp;|&nbsp; {prop_type} &nbsp;|&nbsp; {area_sqft} sqft &nbsp;|&nbsp; {bedrooms} BHK
          </p>
          <h1 style='color:#f59e0b;font-size:3.2rem;margin:0;'>{fmt(rf_p)}</h1>
          <p style='color:#94a3b8;margin-top:0.4rem;'>Best Estimated Market Price (Random Forest)</p>
        </div>""", unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("🌲 Random Forest",     fmt(rf_p),  f"R² = {rf_r2:.3f}")
        c2.metric("📈 Gradient Boosting", fmt(gbr_p), f"R² = {gbr_r2:.3f}")
        c3.metric("📉 Linear Regression", fmt(lr_p),  f"R² = {lr_r2:.3f}")

        st.markdown("#### 💰 Price Breakdown")
        breakdown = {
            "Base Land Value":   area_sqft * base_rate * pm,
            "Bedroom Premium":   bedrooms * 180000,
            "Bathroom Value":    bathrooms * 90000,
            "Amenities Value":   amenity_val,
            "Floor Premium":     floor * 15000,
            "Age Deduction":     -(age * 12000),
        }
        fig,ax = plt.subplots(figsize=(9,4))
        fig.patch.set_facecolor('#0d1b2a')
        ax.set_facecolor('#0f2133')
        colors = ['#1d4ed8','#f59e0b','#22c55e','#a855f7','#06b6d4','#ef4444']
        labels = list(breakdown.keys())
        vals   = [abs(v) for v in breakdown.values()]
        ax.barh(labels, vals, color=colors, height=0.5)
        ax.set_title(f"Price Factors — {locality}, {city}", color='#f1f5f9', pad=10)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'₹{x/100000:.0f}L'))
        ax.tick_params(colors='#94a3b8', labelsize=9)
        for s in ax.spines.values(): s.set_color('#1e3a5f')
        for bar,val in zip(ax.patches,vals):
            ax.text(val+50000, bar.get_y()+bar.get_height()/2, f'₹{val/100000:.0f}L', va='center', color='#f1f5f9', fontsize=8)
        st.pyplot(fig); plt.close()

        low,high = rf_p*0.9, rf_p*1.1
        st.info(f"📊 Market Price Range: **{fmt(low)}** — **{fmt(high)}** (±10% market variation)")

# ═══════════ TAB 2 ═══════════
with tab2:
    st.markdown("### 📊 Tamil Nadu Real Estate Market Overview")
    df = generate_data()

    city_avg = {}
    for c,info in TN_CITIES.items():
        city_avg[c] = info["base"] * info["premium"]
    avg_df = pd.DataFrame(list(city_avg.items()), columns=["City","Price/sqft"]).sort_values("Price/sqft", ascending=True)

    fig,ax = plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor('#0d1b2a')
    ax.set_facecolor('#0f2133')
    bar_colors = ['#f59e0b' if c=='Chennai' else '#1d4ed8' for c in avg_df["City"]]
    ax.barh(avg_df["City"], avg_df["Price/sqft"], color=bar_colors, height=0.55)
    ax.set_xlabel("Price per Sqft (₹)", color='#94a3b8')
    ax.set_title("Price per Square Foot by City — Tamil Nadu", color='#f1f5f9', pad=12, fontsize=13)
    ax.tick_params(colors='#94a3b8', labelsize=9)
    for s in ax.spines.values(): s.set_color('#1e3a5f')
    for bar,val in zip(ax.patches, avg_df["Price/sqft"]):
        ax.text(val+50, bar.get_y()+bar.get_height()/2, f'₹{val}', va='center', color='#f1f5f9', fontsize=8)
    st.pyplot(fig); plt.close()

    st.markdown("### 🏙️ City Price Reference Table")
    ref = []
    for c,info in TN_CITIES.items():
        psf = info["base"] * info["premium"]
        ref.append({
            "City": c,
            "Price/sqft (₹)": f"₹{psf:,.0f}",
            "1BHK (600sqft) Est.": fmt(psf*600 + 180000),
            "2BHK (1000sqft) Est.": fmt(psf*1000 + 360000),
            "3BHK (1400sqft) Est.": fmt(psf*1400 + 540000),
        })
    st.dataframe(pd.DataFrame(ref), use_container_width=True, hide_index=True)

    st.markdown("### 📈 Model Performance")
    c1,c2,c3 = st.columns(3)
    c1.metric("🌲 Random Forest R²",     f"{rf_r2:.4f}",  "Best Model")
    c2.metric("📈 Gradient Boosting R²", f"{gbr_r2:.4f}", "2nd Best")
    c3.metric("📉 Linear Regression R²", f"{lr_r2:.4f}",  "Baseline")

# ═══════════ TAB 3 ═══════════
with tab3:
    st.markdown("### 💰 Budget Planner — Find the Right City for Your Budget")
    budget = st.slider("Your Budget (₹ Lakhs)", 10, 700, 80, 5)
    bhk    = st.selectbox("BHK Required", [1,2,3,4], index=1)
    budget_val = budget * 100000

    results = []
    for c,info in TN_CITIES.items():
        psf = info["base"] * info["premium"]
        fixed = bhk*180000 + max(bhk-1,1)*90000
        possible_area = max(0, (budget_val - fixed) / psf)
        feasibility = "✅ Excellent" if possible_area>1200 else "✅ Good" if possible_area>800 else "⚠️ Tight" if possible_area>400 else "❌ Not Feasible"
        results.append({
            "City": c,
            "Approx Area Possible": f"{possible_area:.0f} sqft",
            "Price/sqft": f"₹{psf:,.0f}",
            "Feasibility": feasibility,
            "Sample Localities": ", ".join(info["areas"][:3]),
        })

    res_df = pd.DataFrame(results)
    st.dataframe(res_df, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div style='background:rgba(29,78,216,0.15);border:1px solid rgba(245,158,11,0.3);
                border-radius:12px;padding:1.2rem;margin-top:1rem;'>
      <h4 style='color:#f59e0b;margin:0 0 0.5rem;'>💡 Budget Tip</h4>
      <p style='color:#94a3b8;margin:0;'>
        With ₹{budget}L budget for a {bhk}BHK:<br>
        • <strong style='color:#f1f5f9;'>Best Value:</strong> Salem, Tirunelveli, Dindigul, Thanjavur<br>
        • <strong style='color:#f1f5f9;'>Mid Range:</strong> Trichy, Madurai, Coimbatore, Vellore<br>
        • <strong style='color:#f1f5f9;'>Premium:</strong> Chennai (suburbs), Hosur (IT corridor)
      </p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br><p style='text-align:center;color:#475569;font-size:0.8rem;'>Antony Ruffina N | MCA — St. Joseph's College, Trichy | Housing Price Prediction</p>", unsafe_allow_html=True)
