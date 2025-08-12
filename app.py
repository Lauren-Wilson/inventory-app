################################################################
# 📦 INVENTORY & FINANCIAL DASHBOARD (Executive, Streamlit)
################################################################
# Implements PROMPT FINAL:
# - Pages: Update Inventory, Admin Dashboard (Inventory + P&L)
# - Demo Mode builds DATASET_INV (master inventory) and DATASET_FIN (financials)
# - Google Sheets authentication preserved for future live data
# - Professional visuals (Plotly), titles above visuals, consistent spacing
################################################################

################################################################
# 📦 IMPORTS AND SETUP
################################################################
import os
import json
import pickle
import datetime as dt
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Optional Google Sheets (kept for real data)
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

################################################################
# 🌐 PAGE CONFIG + STYLES
################################################################
st.set_page_config(page_title="📦 Inventory & Financial Dashboard", layout="wide")

st.markdown(
    """
    <style>
      .section-title { font-size: 26px; font-weight: 700; margin: 4px 0 4px 0; }
      .section-subtitle { color: #57606a; font-size: 15px; margin: 0 0 18px 0; }
      .block-sep { margin: 6px 0 12px 0; }
      .kpi .stMetric { background: #f7f9fb; border-radius: 12px; padding: 12px; }
      .small-note { color:#6b7280; font-size: 12px; }
      /* Ensure black text for readability in tables */
      .stDataFrame, .stMarkdown, .stTable, .stText { color: #111 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

################################################################
# 🗂️ PATHS & CONSTANTS
################################################################
CACHE_DIR = ".cache"
INVENTORY_CACHE = os.path.join(CACHE_DIR, "inventory.pkl")
EVENTS_CACHE = os.path.join(CACHE_DIR, "events.pkl")
GALLERY_JSON = "gallery.json"
os.makedirs(CACHE_DIR, exist_ok=True)

IMG_PLACEHOLDER = "https://dummyimage.com/320x200/f4f6f8/556275.png&text=No+Image"

################################################################
# 🔒 GOOGLE SHEETS: CACHE + FETCH
################################################################
def load_cached_data() -> Tuple[List[dict], List[dict]]:
    try:
        with open(INVENTORY_CACHE, "rb") as f:
            inv = pickle.load(f)
        with open(EVENTS_CACHE, "rb") as f:
            ev = pickle.load(f)
        return inv, ev
    except Exception:
        return [], []

def fetch_and_cache_data() -> Tuple[List[dict], List[dict]]:
    try:
        if gspread is None or Credentials is None:
            return [], []
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        sa_info = st.secrets.get("gcp_service_account")
        cfg = st.secrets.get("google_sheets", {})
        if not sa_info or "sheet_id" not in cfg:
            return [], []
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(cfg["sheet_id"])
        inv = sheet.worksheet("Inventory").get_all_records()
        ev = sheet.worksheet("Events").get_all_records()
        with open(INVENTORY_CACHE, "wb") as f:
            pickle.dump(inv, f)
        with open(EVENTS_CACHE, "wb") as f:
            pickle.dump(ev, f)
        return inv, ev
    except Exception as e:
        st.info(f"Google Sheets not available right now: {e}")
        return [], []

# Load data: cache → sheets → empty
inv_records, ev_records = load_cached_data()
if not inv_records or not ev_records:
    inv_records, ev_records = fetch_and_cache_data()

inv_master_df = pd.DataFrame(inv_records)
events_df = pd.DataFrame(ev_records)

################################################################
# 🧼 NORMALIZE INVENTORY SCHEMA (support different sheet headers)
################################################################
if not inv_master_df.empty:
    inv_master_df.columns = [str(c).strip().lower().replace(" ", "_") for c in inv_master_df.columns]

# Required columns for normalized master inventory (we treat base_qty as canonical)
required_cols_defaults = {
    "item_id": None,
    "item_name": "",
    "category": "Uncategorized",
    "base_qty": None,        # canonical (if absent, we try 'quantity')
    "cost_per_item": 0.0,
    "revenue_per_item": 0.0,
}
for col, default in required_cols_defaults.items():
    if col not in inv_master_df.columns:
        # Map fallback from legacy names (quantity -> base_qty)
        if col == "base_qty" and "quantity" in inv_master_df.columns:
            inv_master_df["base_qty"] = inv_master_df["quantity"]
        else:
            inv_master_df[col] = default

# Enforce types and item_id string (to match gallery.json)
if not inv_master_df.empty:
    inv_master_df["item_id"] = inv_master_df["item_id"].astype(str)
    for c in ["base_qty"]:
        inv_master_df[c] = pd.to_numeric(inv_master_df[c], errors="coerce").fillna(0).astype(int)
    for c in ["cost_per_item", "revenue_per_item"]:
        inv_master_df[c] = pd.to_numeric(inv_master_df[c], errors="coerce").fillna(0.0)

################################################################
# 🖼️ GALLERY IMAGE MAP (item_id → URL)
################################################################
try:
    with open(GALLERY_JSON, "r") as f:
        gallery = json.load(f)
    gallery_df = pd.DataFrame(gallery)
    if not gallery_df.empty and "item_id" in gallery_df.columns:
        gallery_df["item_id"] = gallery_df["item_id"].astype(str)
    else:
        gallery_df = pd.DataFrame(columns=["item_id", "image_url"])
except Exception:
    # Minimal placeholder for when gallery.json is missing
    gallery_df = pd.DataFrame(columns=["item_id", "image_url"])

img_lookup = {
    getattr(r, "item_id"): getattr(r, "image_url", IMG_PLACEHOLDER)
    for r in gallery_df.itertuples(index=False)
    if getattr(r, "image_url", None)
}

################################################################
# 🧭 SIDEBAR NAVIGATION + DEMO TOGGLE
################################################################
st.sidebar.title("📋 Navigation")
main_page = st.sidebar.radio("Choose a page:", ["Update Inventory", "Admin Dashboard"])

use_demo = st.sidebar.checkbox("💡 Use Demo Data (toy)", value=False)

################################################################
# 🧪 TOY DATASETS — DATASET_INV & DATASET_FIN
################################################################
def build_demo_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      DATASET_INV: master inventory table with required columns + yearly totals
      inv_usage_logs: monthly inventory usage logs (used for visuals)
      DATASET_FIN: per-event, per-month financials with October spike
    """
    np.random.seed(42)

    # --- DATASET_INV (master inventory base) ---
    items = [
        {"item_id": "P100", "item_name": "Plates",      "category": "Dinnerware", "base_qty": 1000, "cost_per_item": 2.0, "revenue_per_item": 4.0},
        {"item_id": "P200", "item_name": "Cups",        "category": "Dinnerware", "base_qty": 800,  "cost_per_item": 1.0, "revenue_per_item": 2.0},
        {"item_id": "P300", "item_name": "Napkins",     "category": "Consumables","base_qty": 1200, "cost_per_item": 0.5,"revenue_per_item": 1.0},
        {"item_id": "P400", "item_name": "Forks",       "category": "Dinnerware", "base_qty": 500,  "cost_per_item": 1.5,"revenue_per_item": 3.0},
        {"item_id": "P500", "item_name": "Centerpiece", "category": "Decor",      "base_qty": 300,  "cost_per_item": 5.0, "revenue_per_item": 10.0},
    ]
    DATASET_INV = pd.DataFrame(items)

    # Annual totals for demo (drives monthly logs)
    DATASET_INV["quantity_lost"] = (DATASET_INV["base_qty"] * np.random.uniform(0.00, 0.15, size=len(DATASET_INV))).astype(int)
    DATASET_INV["quantity_used"] = (DATASET_INV["base_qty"] * np.random.uniform(0.25, 0.60, size=len(DATASET_INV))).astype(int)

    # --- Monthly usage logs (Inventory) ---
    months = pd.date_range("2024-01-01", periods=12, freq="MS")
    inv_usage_rows = []
    demo_events = ["Spring Gala", "Summer Fest", "Fall Fair", "Winter Wonderland", "Corporate Retreat"]

    for _, it in DATASET_INV.iterrows():
        yearly_used = int(it["quantity_used"])
        yearly_lost = int(it["quantity_lost"])

        raw = np.abs(np.random.normal(1.0, 0.25, size=12))
        raw[9] *= 1.5  # October boost
        month_weights = raw / raw.sum()

        used_monthly = (yearly_used * month_weights).astype(int)
        lost_monthly = (yearly_lost * month_weights).astype(int)

        for idx, m in enumerate(months):
            inv_usage_rows.append({
                "log_id": len(inv_usage_rows) + 1,
                "date": m.date(),
                "event_name": np.random.choice(demo_events),
                "item_id": it["item_id"],
                "item_name": it["item_name"],
                "category": it["category"],
                "used_qty": int(used_monthly[idx]),
                "lost_qty": int(lost_monthly[idx]),
                "cost_per_item": float(it["cost_per_item"]),
                "revenue_per_item": float(it["revenue_per_item"]),
            })

    inv_usage_logs = pd.DataFrame(inv_usage_rows)

    # --- DATASET_FIN (per-event per-month financials) ---
    fin_events = [
        "Wedding Expo", "Corporate Retreat", "Holiday Gala", "Charity Auction",
        "Music Festival", "Food Expo", "Tech Conference", "Spring Gala",
        "Summer Fest", "Winter Wonderland"
    ]
    DATASET_FIN_rows = []
    for m in months:
        season = 2.0 if m.month == 10 else np.random.uniform(0.9, 1.1)  # strong October spike
        for ev in fin_events:
            base_rev = np.random.uniform(5000, 20000)
            revenue = base_rev * season
            expenses = revenue * np.random.uniform(0.25, 0.45)  # 25–45% cost
            DATASET_FIN_rows.append({
                "event_name": ev,
                "month": m,  # datetime for easier filtering
                "revenue": round(float(revenue), 2),
                "expenses": round(float(expenses), 2),
                "net_profit": round(float(revenue - expenses), 2),
            })
    DATASET_FIN = pd.DataFrame(DATASET_FIN_rows)

    return DATASET_INV, inv_usage_logs, DATASET_FIN

################################################################
# 🔧 DEMO vs LIVE: initialize datasets
################################################################
if use_demo:
    # Use our crafted datasets
    DATASET_INV, inv_usage_logs, DATASET_FIN = build_demo_datasets()
    # If inventory master from Sheets is empty, derive from DATASET_INV for master fields
    if inv_master_df.empty:
        inv_master_df = DATASET_INV[["item_id","item_name","category","base_qty","cost_per_item","revenue_per_item"]].copy()
else:
    # Live mode: usage logs empty until user logs; financials empty unless you ingest a real source
    DATASET_INV = inv_master_df[["item_id","item_name","category","base_qty","cost_per_item","revenue_per_item"]].copy() if not inv_master_df.empty else pd.DataFrame(columns=["item_id","item_name","category","base_qty","cost_per_item","revenue_per_item"])
    DATASET_INV["quantity_used"] = 0
    DATASET_INV["quantity_lost"] = 0
    inv_usage_logs = pd.DataFrame(columns=["log_id","date","event_name","item_id","item_name","category","used_qty","lost_qty","cost_per_item","revenue_per_item"])
    DATASET_FIN = pd.DataFrame(columns=["event_name","month","revenue","expenses","net_profit"])

# Keep in session for easy access
st.session_state["DATASET_INV"] = DATASET_INV
st.session_state["usage_data_inventory"] = inv_usage_logs
st.session_state["usage_data_financial"] = DATASET_FIN

################################################################
# 🔍 HELPERS
################################################################
def section(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def search_filter(df: pd.DataFrame, search: str, category: str) -> pd.DataFrame:
    """
    Case-insensitive startswith on item_name; exact category if provided.
    'PL' → 'Plates', not 'apple'.
    """
    work = df.copy()
    if search:
        mask = work["item_name"].str.lower().str.startswith(search.lower())
        work = work[mask]
    if category and category != "All Categories":
        work = work[work["category"] == category]
    return work

################################################################
# 🔧 UPDATE INVENTORY PAGE (images + logging)
################################################################
def update_inventory_page():
    section("🛠️ Update Inventory", "Log usage by item. Use global event, filter by category, and starts-with search.")

    # Global event (optional)
    ev_names = events_df["event_name"].dropna().unique().tolist() if "event_name" in events_df.columns and not events_df.empty else []
    global_event = st.selectbox("Global Event (optional)", options=["— None —"] + ev_names, index=0, key="global_event_select")

    # Filters
    colf1, colf2 = st.columns([2, 2])
    with colf1:
        search_term = st.text_input("Item name starts with", placeholder="e.g., Pla").strip()
    with colf2:
        categories = ["All Categories"] + sorted(inv_master_df["category"].dropna().unique().tolist()) if not inv_master_df.empty else ["All Categories"]
        category_filter = st.selectbox("Category", options=categories, index=0)

    if inv_master_df.empty:
        st.info("No inventory master found (Google Sheets empty). Enable Demo Mode to try the app.")
        return

    subset = search_filter(inv_master_df, search_term, category_filter)
    if subset.empty:
        st.info("No items match your filters.")
        return

    # Render cards (3 per row)
    cols_per_row = 3
    for i, row in enumerate(subset.itertuples(index=False)):
        if i % cols_per_row == 0:
            cols = st.columns(cols_per_row)
        col = cols[i % cols_per_row]
        with col:
            iid = str(row.item_id)
            img_url = img_lookup.get(iid, IMG_PLACEHOLDER)
            st.image(img_url, use_container_width=True)
            st.markdown(f"**{row.item_name}**\n\n_{row.category}_")
            with st.expander("Log usage"):
                # Default to global event if chosen; else picker
                if global_event != "— None —" and global_event in ev_names:
                    selected_event = global_event
                else:
                    selected_event = st.selectbox(
                        "Event", ev_names if ev_names else ["(No events)"], key=f"ev_{iid}")

                used_qty = st.number_input("Used qty", min_value=0, step=1, key=f"u_{iid}")
                lost_qty = st.number_input("Lost qty", min_value=0, step=1, key=f"l_{iid}")
                if st.button("✅ Log", key=f"log_{iid}"):
                    now = dt.datetime.now().date()
                    inv_log = st.session_state.get("usage_data_inventory", pd.DataFrame())
                    new_row = {
                        "log_id": (inv_log["log_id"].max() + 1) if not inv_log.empty else 1,
                        "date": now,
                        "event_name": selected_event if global_event != "— None —" or ev_names else "",
                        "item_id": iid,
                        "item_name": row.item_name,
                        "category": row.category,
                        "used_qty": int(used_qty),
                        "lost_qty": int(lost_qty),
                        "cost_per_item": float(getattr(row, "cost_per_item", 0.0)),
                        "revenue_per_item": float(getattr(row, "revenue_per_item", 0.0)),
                    }
                    inv_log = pd.concat([inv_log, pd.DataFrame([new_row])], ignore_index=True)
                    st.session_state["usage_data_inventory"] = inv_log
                    st.success(f"Logged {used_qty} used / {lost_qty} lost for {row.item_name}.")

################################################################
# 📊 INVENTORY DASHBOARD (Operational View)
################################################################
def render_inventory_dashboard():
    section("📊 Inventory Dashboard", "Stock health, usage trends, and category distribution.")

    inv_usage = st.session_state.get("usage_data_inventory", pd.DataFrame())
    if inv_usage.empty:
        st.info("No inventory usage logs yet. Log some on the Update Inventory page, or enable Demo Mode.")
        return

    # KPIs
    total_used = int(inv_usage["used_qty"].sum())
    total_lost = int(inv_usage["lost_qty"].sum())

    used_per_item = inv_usage.groupby("item_id")[["used_qty","lost_qty"]].sum().reindex(inv_master_df["item_id"], fill_value=0)
    base_by_item = inv_master_df.set_index("item_id")["base_qty"]
    remaining_per_item = (base_by_item - used_per_item["used_qty"] - used_per_item["lost_qty"]).fillna(0)
    # KPI clamps negatives to 0 for the headline only
    total_remaining_kpi = int(remaining_per_item.clip(lower=0).sum())

    # Top used item
    top_series = inv_usage.groupby("item_id")["used_qty"].sum().sort_values(ascending=False)
    top_item_id = top_series.index[0]
    top_item_name = inv_master_df.loc[inv_master_df["item_id"] == top_item_id, "item_name"].iloc[0]
    top_item_used = int(top_series.iloc[0])

    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Items Used", f"{total_used:,}")
    k2.metric("Total Items Lost", f"{total_lost:,}")
    k3.metric("Total Remaining Stock", f"{total_remaining_kpi:,}")
    k4.metric("Top Used Item", f"{top_item_name} ({top_item_used:,})")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)

    # Usage by Category (Pie)
    section("📊 Usage by Category")
    cat_usage = (
        inv_usage.groupby("item_id")["used_qty"].sum().reset_index()
        .merge(inv_master_df[["item_id","category"]], on="item_id", how="left")
        .groupby("category", dropna=False)["used_qty"].sum().reset_index()
        .sort_values("used_qty", ascending=False)
    )
    fig_pie = px.pie(cat_usage, names="category", values="used_qty", hole=0.35)
    fig_pie.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="")
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)

    # Monthly Usage Trend (filterable by event)
    section("📈 Monthly Usage Trend")
    work = inv_usage.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["month"] = work["date"].dt.to_period("M").dt.to_timestamp()
    all_events = sorted(work["event_name"].dropna().unique().tolist())
    chosen_ev = st.multiselect("Filter by event(s)", options=all_events, default=all_events, key="inv_ev_filter")
    if chosen_ev:
        work = work[work["event_name"].isin(chosen_ev)]
    monthly_usage = work.groupby("month")["used_qty"].sum().reset_index()
    fig_line = px.line(monthly_usage, x="month", y="used_qty", markers=True)
    fig_line.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis_title="Units Used", xaxis_title="")
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)

    # Top Performing Items (Bar)
    section("🏆 Top Performing Items")
    top_items = (
        inv_usage.groupby("item_id")["used_qty"].sum().reset_index()
        .merge(inv_master_df[["item_id","item_name"]], on="item_id", how="left")
        .sort_values("used_qty", ascending=False).head(5)
    )
    fig_top = px.bar(top_items, x="item_name", y="used_qty")
    fig_top.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis_title="", yaxis_title="Units Used")
    st.plotly_chart(fig_top, use_container_width=True)
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)

    # Inventory Balance (Table) — % Remaining color-coded; text black
    section("📦 Inventory Balance")
    bal = inv_master_df[["item_id","item_name","category","base_qty"]].copy()
    bal = bal.merge(used_per_item, left_on="item_id", right_index=True, how="left")
    bal[["used_qty","lost_qty"]] = bal[["used_qty","lost_qty"]].fillna(0).astype(int)
    bal["remaining_stock"] = (bal["base_qty"] - bal["used_qty"] - bal["lost_qty"]).astype(int)
    bal["remaining_pct"] = np.where(bal["base_qty"] > 0, (bal["remaining_stock"] / bal["base_qty"]) * 100.0, 0.0)
    bal["status"] = bal["remaining_pct"].apply(lambda v: "🔴" if v <= 0 else ("🟡" if v <= 10 else "🟢"))

    view_cols = ["item_name","category","base_qty","used_qty","lost_qty","remaining_stock","remaining_pct","status"]
    bal_display = bal[view_cols].copy()
    bal_display["remaining_pct"] = bal_display["remaining_pct"].round(1)

    def pct_style(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v <= 0:
            return "background-color: #FDE2E1;"  # red-ish
        elif v <= 10:
            return "background-color: #FFF4CE;"  # yellow-ish
        else:
            return "background-color: #E7F4E8;"  # green-ish

    st.dataframe(
        bal_display.style.applymap(pct_style, subset=["remaining_pct"]).format({
            "base_qty": "{:,.0f}",
            "used_qty": "{:,.0f}",
            "lost_qty": "{:,.0f}",
            "remaining_stock": "{:,.0f}",
            "remaining_pct": "{:.1f}%"
        }),
        use_container_width=True
    )

################################################################
# 💰 PROFIT & LOSS DASHBOARD (Financial View)
################################################################
def render_profit_and_loss_dashboard():
    section("💰 Profit & Loss Dashboard", "Revenue, expenses, and profitability trends with seasonality.")

    fin = st.session_state.get("usage_data_financial", pd.DataFrame()).copy()
    inv_usage = st.session_state.get("usage_data_inventory", pd.DataFrame()).copy()

    # Fallback: If financial logs are empty, derive P&L from inventory usage + per-item pricing
    if fin.empty and not inv_usage.empty:
        tmp = inv_usage.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        agg = tmp.groupby(["date","event_name","item_id"], dropna=False)[["used_qty"]].sum().reset_index()
        agg = agg.merge(inv_master_df[["item_id","item_name","category","cost_per_item","revenue_per_item"]],
                        on="item_id", how="left")
        agg["revenue"] = agg["used_qty"] * agg["revenue_per_item"].fillna(0.0)
        agg["expenses"] = agg["used_qty"] * agg["cost_per_item"].fillna(0.0)
        agg["net_profit"] = agg["revenue"] - agg["expenses"]
        agg.rename(columns={"date":"month"}, inplace=True)
        fin = agg[["event_name","month","revenue","expenses","net_profit","item_id","item_name","category"]].copy()

    if fin.empty:
        st.info("No financial logs yet. Enable Demo Mode to view a realistic example.")
        return

    # Normalize & filters
    fin["month"] = pd.to_datetime(fin["month"], errors="coerce")

    colf1, colf2 = st.columns([2, 2])
    with colf1:
        min_d = fin["month"].min().date()
        max_d = fin["month"].max().date()
        date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d, key="pl_dates")
        if isinstance(date_range, tuple):
            start_d, end_d = date_range
        else:
            start_d, end_d = min_d, date_range
    with colf2:
        events = sorted(fin["event_name"].dropna().unique().tolist())
        chosen_events = st.multiselect("Event(s)", options=events, default=events, key="pl_events")

    mask = (fin["month"].dt.date >= start_d) & (fin["month"].dt.date <= end_d)
    if chosen_events:
        mask &= fin["event_name"].isin(chosen_events)
    f = fin.loc[mask].copy()
    if f.empty:
        st.warning("No records for selected filters.")
        return

    # KPIs
    total_revenue = float(f["revenue"].sum())
    total_expenses = float(f["expenses"].sum())
    total_net = float(f["net_profit"].sum())

    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${total_revenue:,.0f}")
    c2.metric("Total Expenses", f"${total_expenses:,.0f}")
    c3.metric("Total Net Profit", f"${total_net:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)

    # Monthly Revenue, Expenses, Net (Line)
    section("📈 Monthly Revenue, Expenses, and Net")
    f["month_index"] = f["month"].dt.to_period("M").dt.to_timestamp()
    monthly = f.groupby("month_index")[["revenue","expenses","net_profit"]].sum().reset_index()
    fig_trend = px.line(monthly, x="month_index", y=["revenue","expenses","net_profit"], markers=True)
    fig_trend.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="")
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)

    # Revenue by Event (Pie) — aligns with DATASET_FIN spec (per event)
    section("🥧 Revenue by Event")
    rev_event = f.groupby("event_name")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
    fig_event_pie = px.pie(rev_event, names="event_name", values="revenue", hole=0.35)
    fig_event_pie.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="")
    st.plotly_chart(fig_event_pie, use_container_width=True)
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)

    # Event Profitability (Bar)
    section("🏷️ Event Profitability (Net)")
    by_evt = f.groupby("event_name")["net_profit"].sum().reset_index().sort_values("net_profit", ascending=False)
    fig_evt = px.bar(by_evt, x="event_name", y="net_profit")
    fig_evt.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis_title="", yaxis_title="Net Profit")
    st.plotly_chart(fig_evt, use_container_width=True)
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)

    # Net Profit Margin Over Time (Line)
    section("📉 Net Profit Margin Over Time")
    monthly["net_margin_%"] = np.where(monthly["revenue"] > 0, (monthly["net_profit"] / monthly["revenue"]) * 100.0, 0.0)
    fig_margin = px.line(monthly, x="month_index", y="net_margin_%", markers=True)
    fig_margin.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis_title="Margin %", xaxis_title="")
    st.plotly_chart(fig_margin, use_container_width=True)

################################################################
# ▶️ ROUTER (Admin tabs inline)
################################################################
def render_admin():
    section("📊 Admin Dashboard", "Switch between operational and financial views.")
    tab = st.radio("Dashboard", ["Inventory Dashboard", "Profit & Loss Dashboard"], key="admin_tabs", horizontal=True)
    if tab == "Inventory Dashboard":
        render_inventory_dashboard()
    else:
        render_profit_and_loss_dashboard()

if main_page == "Update Inventory":
    update_inventory_page()
else:
    render_admin()
