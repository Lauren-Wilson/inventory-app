#%%
################################################################
# 📦 IMPORTS AND INITIAL SETUP
################################################################
import os
import json
import pickle
import datetime
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

#%%
################################################################
# 🌐 PAGE CONFIGURATION
################################################################
st.set_page_config(page_title="🖼️ Inventory Gallery", layout="centered")
st.title("🖼️ Inventory Image Gallery")

#%%
################################################################
# 🗂️ FILE AND CACHE PATHS
################################################################
CACHE_DIR = ".cache"
INVENTORY_CACHE = os.path.join(CACHE_DIR, "inventory.pkl")
EVENTS_CACHE = os.path.join(CACHE_DIR, "events.pkl")
JSON_PATH = "inventory_data.json"
os.makedirs(CACHE_DIR, exist_ok=True)

#%%
################################################################
# 🧠 FUNCTION: Load Cached Data
################################################################
def load_cached_data():
    try:
        with open(INVENTORY_CACHE, "rb") as f:
            inventory_data = pickle.load(f)
        with open(EVENTS_CACHE, "rb") as f:
            events_data = pickle.load(f)
        st.success("✅ Loaded data from local cache.")
        return inventory_data, events_data
    except Exception as e:
        st.warning(f"⚠️ Cache not found or invalid: {e}")
        return None, None

#%%
################################################################
# 🌐 FUNCTION: Fetch and Cache Data from Google Sheets
################################################################
def fetch_and_cache_data():
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        service_account_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
        client = gspread.authorize(creds)

        SHEET_ID = st.secrets["google_sheets"]["sheet_id"]
        sheet = client.open_by_key(SHEET_ID)

        inventory_ws = sheet.worksheet("Inventory")
        events_ws = sheet.worksheet("Events")
        inventory_data = inventory_ws.get_all_records()
        events_data = events_ws.get_all_records()

        with open(INVENTORY_CACHE, "wb") as f:
            pickle.dump(inventory_data, f)
        with open(EVENTS_CACHE, "wb") as f:
            pickle.dump(events_data, f)

        st.success("✅ Google Sheets data fetched and cached.")
        return inventory_data, events_data
    except Exception as e:
        st.error(f"❌ Google Sheets fetch failed: {e}")
        return [], []

#%%
################################################################
# 📋 LOAD INVENTORY + EVENT DATA
################################################################
inventory_data, events_data = load_cached_data()
if not inventory_data or not events_data:
    inventory_data, events_data = fetch_and_cache_data()

#%%
################################################################
# 🖼️ LOAD GALLERY IMAGE DATA FROM LOCAL JSON
################################################################
try:
    with open("gallery.json", "r") as f:
        gallery_imgs = json.load(f)
        st.success("✅ Loaded image gallery data from JSON.")
except Exception as e:
    st.error(f"❌ Failed to load JSON gallery: {e}")
    gallery_imgs = []

#%%
################################################################
# 🖼️ RENDER IMAGE GALLERY & FORMS
################################################################
if not gallery_imgs or "image_url" not in gallery_imgs[0]:
    st.warning("⚠️ No image URLs found in gallery.")
else:
    seen = set()
    for i, item in enumerate(gallery_imgs):
        item_id = item.get("item_id")
        if item_id in seen:
            continue
        seen.add(item_id)

        name = item.get("item_name", f"Unnamed {i}")
        img_url = item.get("image_url", "").strip()

        if img_url:
            st.image(img_url, caption=name, use_container_width=True)
        else:
            st.caption(f"No image available for {name}")

        with st.expander(f"📋 Submit usage for {name}"):
            try:
                selected_event = st.selectbox(
                    f"Select Event for {name}",
                    [event["event_name"] for event in events_data],
                    key=f"event_{item_id}_{i}"
                )
                used_qty = st.number_input(
                    f"Quantity Used for {name}", min_value=0, step=1, key=f"used_{item_id}_{i}"
                )
                lost_qty = st.number_input(
                    f"Quantity Lost for {name}", min_value=0, step=1, key=f"lost_{item_id}_{i}"
                )

                if st.button(f"✅ Submit Usage for {name}", key=f"submit_{item_id}_{i}"):
                    try:
                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
                        creds = Credentials.from_service_account_info(
                            st.secrets["gcp_service_account"], scopes=scopes
                        )
                        client = gspread.authorize(creds)
                        usage_log_ws = client.open_by_key(st.secrets["google_sheets"]["sheet_id"]).worksheet("UsageLog")

                        usage_log_ws.append_row([
                            now, item_id, name, selected_event, used_qty, lost_qty
                        ])
                        st.success(f"✅ Recorded: {name} | Used: {used_qty}, Lost: {lost_qty}, Event: {selected_event}")
                    except Exception as e:
                        st.warning(f"⚠️ Could not write to UsageLog: {e}")
            except Exception as e:
                st.error(f"❌ Error in form for {name}: {e}")
