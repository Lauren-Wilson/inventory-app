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
from thefuzz import process
from google.oauth2.service_account import Credentials

#%%
################################################################
# 🌐 PAGE CONFIGURATION
################################################################
st.set_page_config(page_title="📦 Inventory App", layout="wide")

#%%
################################################################
# 🗂️ FILE AND CACHE PATHS
################################################################
CACHE_DIR = ".cache"
INVENTORY_CACHE = os.path.join(CACHE_DIR, "inventory.pkl")
EVENTS_CACHE = os.path.join(CACHE_DIR, "events.pkl")
GALLERY_JSON = "gallery.json"
os.makedirs(CACHE_DIR, exist_ok=True)

#%%
################################################################
# 🔒 GOOGLE SHEETS SETUP + CACHING
################################################################
def load_cached_data():
    try:
        with open(INVENTORY_CACHE, "rb") as f:
            inventory_data = pickle.load(f)
        with open(EVENTS_CACHE, "rb") as f:
            events_data = pickle.load(f)
        return inventory_data, events_data
    except:
        return None, None

def fetch_and_cache_data():
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scopes
        )
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

        return inventory_data, events_data
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return [], []

inventory_data, events_data = load_cached_data()
if not inventory_data or not events_data:
    inventory_data, events_data = fetch_and_cache_data()

#%%
################################################################
# 🖼️ LOAD IMAGE DATA FROM JSON
################################################################
try:
    with open(GALLERY_JSON, "r") as f:
        gallery_imgs = json.load(f)
except:
    gallery_imgs = []

#%%
################################################################
# 🧭 SIDEBAR NAVIGATION
################################################################
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Choose a page:", ["Update Inventory", "Admin Dashboard"])

#%%
################################################################
# 🔧 UPDATE INVENTORY PAGE
################################################################
if page == "Update Inventory":
    st.title("🛠️ Update Inventory")

    if not gallery_imgs:
        st.warning("⚠️ No inventory images found.")
    else:
        item_names = [item["item_name"] for item in gallery_imgs]
        search_term = st.text_input("🔍 Search Inventory (start of name only)").lower()

        def starts_with_filter(search, names):
            return [name for name in names if name.lower().startswith(search)]

        matched_names = starts_with_filter(search_term, item_names) if search_term else item_names

        # Filter items based on matched names
        filtered_items = [item for item in gallery_imgs if item["item_name"] in matched_names]

        # Group by category and alphabetize
        categories = sorted(set(item.get("category", "Uncategorized") for item in filtered_items))

        for category in categories:
            st.markdown(f"### 📁 {category}")
            for item in [i for i in filtered_items if i.get("category") == category]:
                name = item.get("item_name", "Unnamed")
                item_id = item.get("item_id", "")
                image_url = item.get("image_url", "")

                if image_url:
                    st.image(image_url, caption=name, use_container_width=True)
                else:
                    st.caption(f"No image for {name}")

                with st.expander(f"📋 Submit usage for {name}"):
                    selected_event = st.selectbox(
                        f"Select Event for {name}",
                        [e["event_name"] for e in events_data],
                        key=f"event_{item_id}"
                    )
                    used_qty = st.number_input(
                        f"Quantity Used for {name}", min_value=0, step=1, key=f"used_{item_id}"
                    )
                    lost_qty = st.number_input(
                        f"Quantity Lost for {name}", min_value=0, step=1, key=f"lost_{item_id}"
                    )
                    if st.button(f"✅ Submit for {name}", key=f"submit_{item_id}"):
                        try:
                            creds = Credentials.from_service_account_info(
                                st.secrets["gcp_service_account"],
                                scopes=["https://www.googleapis.com/auth/spreadsheets"]
                            )
                            client = gspread.authorize(creds)
                            usage_log_ws = client.open_by_key(
                                st.secrets["google_sheets"]["sheet_id"]
                            ).worksheet("UsageLog")
                            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            usage_log_ws.append_row([
                                now, item_id, name, selected_event, used_qty, lost_qty
                            ])
                            st.success(f"✅ Logged usage for {name}")
                        except Exception as e:
                            st.error(f"❌ Could not log usage: {e}")

#%%
################################################################
# 👩‍💻 ADMIN DASHBOARD PAGE
################################################################
elif page == "Admin Dashboard":
    st.title("🛠️ Admin Dashboard")
    st.info("This page is under construction 🚧. Future features will include:")
    st.markdown("- Inventory summary by item and event")
    st.markdown("- Usage trends and analytics")
    st.markdown("- Export options for reports")
    st.markdown("- User management and permissions")
    st.markdown("- and more...")
