import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
client = gspread.authorize(creds)

SHEET_ID = st.secrets["google_sheets"]["sheet_id"]
sheet = client.open_by_key(SHEET_ID)

try:
    inventory_ws = sheet.worksheet("Inventory")
    inventory_data = inventory_ws.get_all_records()
    # st.text("✅ Inventory loaded")
    st.write(inventory_data[:3])
except Exception as e:
    st.error(f"Error loading inventory: {e}")
