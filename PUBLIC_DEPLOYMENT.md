# MaydonOs public web version

The new non-Streamlit interface is served by FastAPI from the same origin as its JSON API. The earlier Streamlit prototype remains available at `legacy_streamlit/app.py`; public users open the new web interface.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
$env:FIRMS_MAP_KEY = "your NASA FIRMS map key"
.\.venv\Scripts\python.exe -m uvicorn backend.web_api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Without `FIRMS_MAP_KEY`, the site explains that NASA observations are unavailable; it does not invent demo detections. Open-Meteo weather is public and needs no key.

On Windows, `MaydonOs_Web_Baslat.bat` starts this web version and opens it in the browser.

## Publish on Render

1. Put the project in a GitHub repository. It can remain **private**; the deployed website can still be public.
2. Keep `.streamlit/secrets.toml`, `.env` files, and all API credentials out of Git. `.gitignore` excludes the local secret file and environment files.
3. In Render, create a Blueprint from the repository and use `render.yaml`.
4. Set `FIRMS_MAP_KEY` as a secret environment variable for the web service. Never add it to HTML, JavaScript, or a committed file.
5. Set the deployed service's visibility to public and share its `onrender.com` URL.

Render's free web service sleeps after 15 minutes without inbound traffic and may take about a minute to wake. The free filesystem is temporary. This public interface displays the latest weather and the last 24 hours of satellite detections; it does not promise persistent historical anomaly records or uninterrupted 24/7 monitoring.

The interface is publicly readable and does not ask visitors for coordinates, accounts, location permissions, or personal information. API credentials remain on the server. The map loads Esri World Imagery and the Leaflet library from their public endpoints.
