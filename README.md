
# 📦 Inventory Tracking App

A mobile-optimized, responsive inventory tracking app built with **Streamlit** and **Google Sheets**, designed for event planners, small business owners, and teams managing shared inventory.

---

## 🚀 Features

- 📱 **Mobile-first gallery UI** with item images and usage forms
- 🔍 **Fuzzy search** to quickly find inventory by name
- 🧾 **Track quantity used and quantity lost**
- 📸 **Image gallery** pulled from a local `gallery.json` file
- 🗃️ **Inventory and event data** loaded from Google Sheets (with caching for performance)
- ✅ **Submit usage logs** directly to the Google Sheet backend
- 🧠 **Cached sheet data** for faster debugging and offline fallback
- 🛠️ **Admin page** (placeholder for future expansion)

---

## 📁 Project Structure

```

.
├── app.py                     # Main Streamlit app
├── gallery.json              # Stores item\_id, item\_name, and image\_url for gallery
├── .cache/                   # Pickled data for faster access
│   ├── inventory.pkl
│   └── events.pkl
├── requirements.txt          # Python dependencies
└── .streamlit/
└── secrets.toml          # Credentials and config (Streamlit Cloud only)

````

---

## 🧩 Data Sources

- **Google Sheets**
  - `Inventory` — master inventory list
  - `Events` — list of events for usage tracking
  - `UsageLog` — submission log for usage entries

- **Local JSON**
  - `gallery.json` — stores image URLs with associated item IDs and names

---

## 🔐 Setup Instructions

1. **Install requirements**
```bash
   pip install -r requirements.txt
````

2. **Set up `.streamlit/secrets.toml`**

   ```toml
   [gcp_service_account]
   type = "service_account"
   project_id = "..."
   private_key_id = "..."
   private_key = "..."
   client_email = "..."
   client_id = "..."
   token_uri = "https://oauth2.googleapis.com/token"

   [google_sheets]
   sheet_id = "your_google_sheet_id"
   ```

3. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deployment

You can deploy this app on [Streamlit Cloud](https://streamlit.io/cloud):

1. Push your project to a GitHub repository
2. Link it via **Streamlit Cloud**
3. Add your secrets under **App Settings → Secrets**
4. Click **Deploy**

---

## 📸 Example Screenshot

Coming Soon
---

## 📦 Future Enhancements

* Custom roles and user authentication
* Admin dashboard for inventory reporting
* Offline entry queuing and sync
* Item reorder tracking or low-stock alerts
* Export reports (CSV/PDF)

---

## 🙌 Credits

Developed with ❤️ by Lauren Wilson
Part of a growing toolkit of tools to simplify inventory and logistics for modern teams.

---


