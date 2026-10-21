(() => {
  const $ = (id) => document.getElementById(id);
  const select = $("district-select"), provinceSelect = $("province-select"), refresh = $("refresh-button"), nearbyButton = $("nearby-button"), live = $("live-status");
  const number = (value, digits = 1) => {
    if (value === null || value === undefined || value === "") return "—";
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toLocaleString("tr-TR", { maximumFractionDigits: digits, minimumFractionDigits: digits }) : "—";
  };
  const hasNumber = (value) => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const normalize = (value) => String(value ?? "").toLocaleLowerCase("tr-TR").normalize("NFD").replace(/[\u0300-\u036f]/g, "").replaceAll("ı", "i");
  const timeFromRow = (row) => {
    const date = String(row?.acq_date ?? "");
    const hhmm = String(row?.acq_time ?? "").padStart(4, "0");
    if (!/^\d{4}$/.test(hhmm) || !date) return null;
    const stamp = new Date(`${date}T${hhmm.slice(0, 2)}:${hhmm.slice(2)}:00Z`);
    return Number.isNaN(stamp.getTime()) ? null : stamp;
  };
  const rowTime = (row) => row?.time ? new Date(row.time) : timeFromRow(row);
  const formatTime = (value) => {
    const date = value instanceof Date ? value : new Date(value ?? "");
    return Number.isNaN(date.getTime()) ? "Saat bilgisi yok" : `${new Intl.DateTimeFormat("tr-TR", { dateStyle: "medium", timeStyle: "short", timeZone: "UTC" }).format(date)} UTC`;
  };
  const relativeTime = (value) => {
    const date = value instanceof Date ? value : new Date(value ?? "");
    if (Number.isNaN(date.getTime())) return "—";
    const minutes = Math.max(0, Math.round((Date.now() - date.getTime()) / 60000));
    return minutes < 60 ? `${minutes} dk önce` : `${Math.floor(minutes / 60)} sa ${minutes % 60} dk önce`;
  };
  function lastObservation(item) {
    if (item?.latest_observation) return item.latest_observation;
    const list = Array.isArray(item?.observations) ? item.observations : [];
    return list.at(-1) || item;
  }
  function confidenceClass(value) {
    const label = normalize(value);
    if (["h", "high", "yuksek"].includes(label)) return "high";
    if (["n", "nominal", "normal"].includes(label)) return "normal";
    if (["l", "low", "dusuk"].includes(label)) return "low";
    if (value === null || value === undefined || String(value).trim() === "") return "unknown";
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric < 0 || numeric > 100) return "unknown";
    // MODIS FIRMS percentage classes: <30 low, 30–<80 nominal, >=80 high.
    return numeric >= 80 ? "high" : numeric >= 30 ? "normal" : "low";
  }
  function landType(item) {
    const code = String(item?.corine_code ?? "");
    const name = normalize(item?.corine_class);
    if (code.startsWith("2") || /tarim|ekilebilir|mera|bahce/.test(name)) return { code: "t", name: item?.corine_class || "Tarım", color: "#45684f" };
    if (code.startsWith("3") || /orman|dogal|calilik/.test(name)) return { code: "o", name: item?.corine_class || "Orman / doğal örtü", color: "#8a7b2e" };
    if (code.startsWith("1") || /sanayi|yerlesim|yapi|maden/.test(name)) return { code: "s", name: item?.corine_class || "Yapı / sanayi", color: "#746d71" };
    if (code.startsWith("4") || code.startsWith("5") || /su|sulak|batak/.test(name)) return { code: "w", name: item?.corine_class || "Su / sulak alan", color: "#43839a" };
    return { code: "x", name: item?.corine_class || "Diğer / bilinmiyor", color: "#9b8d80" };
  }
  function eventType(item) {
    const label = normalize(item?.candidate);
    if (/sanayi|tesis/.test(label)) return { label: "Tesis çevresi", color: "#777477" };
    if (/orman/.test(label)) return { label: "Orman / doğal örtü adayı", color: "#557b55" };
    if (/tarim alaninda yuksek/.test(label)) return { label: "Tarım alanında yüksek öncelik", color: "#a33c32" };
    if (/kirsal termal/.test(label)) return { label: "Kırsal termal aday", color: "#b9802f" };
    return { label: "Genel gözlem", color: "#9b8d80" };
  }
  function scoreInfo(score) {
    const value = score === null || score === undefined ? NaN : Number(score);
    if (!Number.isFinite(value) || value <= 0) return { score: null, label: "Puanlanamadı", color: "#817b75", band: "low" };
    if (value >= 75) return { score: Math.round(value), label: "Öncelikli saha kontrolü", color: "#a33c32", band: "high" };
    if (value >= 50) return { score: Math.round(value), label: "Kontrol değerlendirilebilir", color: "#b9802f", band: "mid" };
    return { score: Math.round(value), label: "Düşük öncelik", color: "#45684f", band: "low" };
  }
  function markerColorBand(item) {
    if (hasNumber(item?.model_rank_percentile)) {
      const rank = Number(item.model_rank_percentile);
      return rank >= 75 ? "high" : rank >= 50 ? "mid" : "low";
    }
    const priority = scoreInfo(item?.priority_score);
    const isRedAlert = priority.band === "high" || eventType(item).color.toLowerCase() === "#a33c32";
    return isRedAlert ? "high" : priority.band;
  }
  function scoreBreakdown(observation) {
    const parts = [];
    const confidenceData = { high: [100, "Yüksek"], normal: [65, "Normal"], low: [35, "Düşük"] }[confidenceClass(observation?.confidence)];
    if (confidenceData) parts.push({ name: "FIRMS güven sınıfı", shown: confidenceData[1], value: confidenceData[0], weight: .45 });
    const frp = observation?.frp === null || observation?.frp === undefined || observation?.frp === "" ? NaN : Number(observation.frp);
    if (Number.isFinite(frp) && frp >= 0) parts.push({ name: "FRP", shown: `${number(frp, 2)} MW`, value: 100 * frp / (frp + 10), weight: .35 });
    const ti4 = observation?.bright_ti4 === null || observation?.bright_ti4 === undefined || observation?.bright_ti4 === "" ? NaN : Number(observation.bright_ti4);
    if (Number.isFinite(ti4)) parts.push({ name: "TI4 termal parlaklığı", shown: `${number(ti4, 2)} K`, value: Math.max(0, Math.min(100, (ti4 - 300) / .6)), weight: .2 });
    const totalWeight = parts.reduce((sum, part) => sum + part.weight, 0);
    if (!totalWeight) return { rows: "<li>Ölçüm bileşeni yok</li>", total: null, missing: true };
    const rows = parts.map((part) => {
      const weight = part.weight / totalWeight;
      return `<li><span>${escapeHtml(part.name)} · ${escapeHtml(part.shown)}</span><b>${number(part.value * weight)} puan <small>(%${number(weight * 100)})</small></b></li>`;
    }).join("");
    return { rows, total: Math.round(parts.reduce((sum, part) => sum + part.value * part.weight / totalWeight, 0)), missing: parts.length < 3 };
  }
  function observations(item) {
    return (Array.isArray(item?.observations) ? item.observations : []).filter((row) => rowTime(row) && rowTime(row) <= new Date());
  }
  function frpHistory(item) {
    const rows = observations(item);
    const withFrp = rows.filter((row) => row.frp !== null && row.frp !== undefined && row.frp !== "" && Number.isFinite(Number(row.frp)));
    const uniqueTimes = withFrp.filter((row, index) => index === 0 || rowTime(row).getTime() !== rowTime(withFrp[index - 1]).getTime());
    const latest = lastObservation(item);
    if (uniqueTimes.length < 2) {
      return `<section class="history"><div class="history-title"><span>FRP değişimi</span><span>Yeterli tekrar ölçümü yok</span></div><p>Bu ilçe içindeki bu koordinat çevresinde farklı zamanlarda alınmış en az iki FRP ölçümü bulunmuyor. Artış veya azalış sonucu çıkarılamaz.</p><small class="history-note">FIRMS yalnızca uydu geçişlerinde ölçüm verir. Aradaki saatler için değer üretilmez. Son gözlem: ${escapeHtml(formatTime(rowTime(latest)))}.</small></section>`;
    }
    const first = uniqueTimes[0], end = uniqueTimes.at(-1);
    const startFrp = Number(first.frp), endFrp = Number(end.frp), delta = endFrp - startFrp;
    const pct = startFrp > 0 ? ` · ${delta > 0 ? "+" : ""}${number(delta / startFrp * 100)}%` : "";
    const trend = Math.abs(delta) < .05 ? "Belirgin değişim yok" : delta > 0 ? "FRP yükselmiş" : "FRP azalmış";
    const changes = `${number(startFrp, 2)} → ${number(endFrp, 2)} MW (${delta > 0 ? "+" : ""}${number(delta, 2)} MW${pct})`;
    const recent = rows.slice(-10);
    const table = recent.map((row) => {
      const hasFrp = row.frp !== null && row.frp !== undefined && row.frp !== "" && Number.isFinite(Number(row.frp));
      const score = scoreInfo(row.priority_score);
      return `<tr><td>${escapeHtml(formatTime(rowTime(row)))}</td><td>${escapeHtml(row.satellite || "—")}</td><td>${hasFrp ? `${number(row.frp, 2)} MW` : "—"}</td><td>${score.score === null ? "—" : `${score.score}/100`}</td></tr>`;
    }).join("");
    return `<section class="history"><div class="history-title"><span>Uydu geçişlerinde FRP</span><span class="trend">${escapeHtml(trend)}</span></div><p>${escapeHtml(changes)}</p><table><thead><tr><th>Zaman (UTC)</th><th>Uydu</th><th>FRP</th><th>Öncelik</th></tr></thead><tbody>${table}</tbody></table><small class="history-note">Son saha kontrolü önceliği en yeni gerçek gözlemin verileriyle hesaplanır. Yakın pikseller farklı olaylara ait olabilir. ${rows.length > recent.length ? `En yeni ${recent.length} / ${rows.length} kayıt gösteriliyor.` : ""}</small></section>`;
  }

  let map, districtGeoJson = null, provinceBoundaryData = null, districtLayer, districtByName = new Map(), markerLayer, nearbyLayer, markerByKey = new Map(), selectedItem = null, dashboard = null, nearbyResults = null, nearbyLocation = null, refreshTimer = null, eumetsatRefreshTimer = null, dashboardRequestId = 0, selectedProvinceCode = "", selectedDayFilter = "", mapFiltersOpen = false, harvestPeriodOnly = false, clusterAnomalyOnly = false, provinceFeatures = [];
  const countryDashboardCache = new Map();
  const dashboardCache = new Map();
  const landFilterSelection = new Set();
  const satelliteFilterSelection = new Set();
  const markerColorSelection = new Set();
  function gibsDateOffset(daysBack) {
    const date = new Date();
    date.setUTCDate(date.getUTCDate() - daysBack);
    return date.toISOString().slice(0, 10);
  }
  function populateImageryDates() {
    const selectDate = $("imagery-date");
    if (!selectDate) return;
    // Include today; the imagery service may publish the daily composite later.
    const dates = Array.from({ length: 7 }, (_, index) => gibsDateOffset(index));
    selectDate.innerHTML = dates.map((value) => {
      const date = new Date(`${value}T00:00:00Z`);
      const label = new Intl.DateTimeFormat("tr-TR", { day: "2-digit", month: "long", year: "numeric", timeZone: "UTC" }).format(date);
      return `<option value="${value}">${escapeHtml(label)}${value === gibsDateOffset(0) ? " · bugün" : ""}</option>`;
    }).join("");
    selectDate.value = dates[0];
  }
  function gibsTileUrl(layer, date, extension) {
    return `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/${layer}/default/${date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.${extension}`;
  }
  function latestCompleteMtgTime() {
    const time = new Date();
    time.setUTCMinutes(Math.floor(time.getUTCMinutes() / 10) * 10 - 10, 0, 0);
    return time.toISOString().replace(".000", "");
  }
  function selectedDistrictName() { return select.value || ""; }
  function pointInRing(lon, lat, ring) {
    if (!Array.isArray(ring) || ring.length < 4) return false;
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i], [xj, yj] = ring[j];
      const crosses = (yi > lat) !== (yj > lat);
      if (crosses && lon < ((xj - xi) * (lat - yi)) / ((yj - yi) || Number.EPSILON) + xi) inside = !inside;
    }
    return inside;
  }
  function pointInProvince(lon, lat, feature) {
    const geometry = feature?.geometry;
    if (!geometry) return false;
    const polygons = geometry.type === "Polygon" ? [geometry.coordinates] : geometry.type === "MultiPolygon" ? geometry.coordinates : [];
    return polygons.some((polygon) => Array.isArray(polygon) && polygon.length && pointInRing(lon, lat, polygon[0]) && !polygon.slice(1).some((hole) => pointInRing(lon, lat, hole)));
  }
  function initializeMap(center = [39, 35]) {
    if (!window.L) {
      $("boundary-status").textContent = "Harita kütüphanesi yüklenemedi.";
      return;
    }
    if (map) return;
    // Allow country-wide users to zoom directly to visible hotspots without
    // first entering a province. Leaflet's +/- controls remain available too.
    map = L.map("map", { zoomControl: true, scrollWheelZoom: true, doubleClickZoom: true, zoomSnap: 0.5, preferCanvas: true }).setView(center, 5.4);
    const osmLayer = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap contributors</a>',
    }).addTo(map);
    districtLayer = L.geoJSON(null, {
      style: (feature) => districtStyle(feature),
      onEachFeature: (feature, layer) => {
        const props = feature?.properties || {};
        const name = props.name || props.shapeName || "İlçe";
        const isProvince = props.shapeType === "ADM1" || Boolean(props.name);
        districtByName.set(normalize(name), layer);
        layer.bindTooltip(escapeHtml(name), { permanent: !isProvince, direction: "center", className: "district-label", opacity: .95 });
        layer.on("mouseover", (event) => {
          event.target.setStyle({ color: "#f4c36d", weight: 3.5, fillOpacity: .2 });
          event.target.bringToFront();
          event.target.openTooltip();
        });
        layer.on("mouseout", (event) => { if (isProvince) event.target.closeTooltip(); updateDistrictStyle(false); });
        layer.on("click", () => {
          if (props.shapeType === "ADM1" || props.name) {
            selectProvince(props.province_code);
            return;
          }
          const option = [...select.options].find((item) => normalize(item.value) === normalize(name));
          if (option) { select.value = option.value; loadDashboard(); }
        });
      },
    }).addTo(map);
    markerLayer = L.layerGroup().addTo(map);
    nearbyLayer = L.layerGroup().addTo(map);
    const imageDate = $("imagery-date")?.value || gibsDateOffset(0);
    const viirsImageLayer = L.tileLayer(gibsTileUrl("VIIRS_SNPP_CorrectedReflectance_TrueColor", imageDate, "jpg"), {
      maxZoom: 18, maxNativeZoom: 9, opacity: .82, updateWhenIdle: true,
      attribution: '&copy; <a href="https://earthdata.nasa.gov/gibs" target="_blank" rel="noopener">NASA GIBS · VIIRS</a>',
    });
    const aerosolLayer = L.tileLayer(gibsTileUrl("OMPS_Aerosol_Index", imageDate, "png"), {
      maxZoom: 18, maxNativeZoom: 9, opacity: .72, updateWhenIdle: true,
      attribution: '&copy; <a href="https://earthdata.nasa.gov/gibs" target="_blank" rel="noopener">NASA GIBS · OMPS</a>',
    });
    const eumetsatLayer = L.tileLayer.wms("https://view.eumetsat.int/geoserver/wms", {
      layers: "mtg_fd:rgb_geocolour", styles: "", format: "image/png", transparent: true,
      version: "1.3.0", uppercase: true, TIME: latestCompleteMtgTime(), opacity: .88,
      attribution: '&copy; <a href="https://view.eumetsat.int/productviewer/productDetails/mtg_fd%3Argb_geocolour?v=default" target="_blank" rel="noopener">EUMETSAT / NASA · MTG GeoColour</a>',
    });
    L.control.layers(
      { "OpenStreetMap": osmLayer },
      {
        "NASA FIRMS · termal noktalar": markerLayer,
        "NASA VIIRS · gerçek renk görüntüsü": viirsImageLayer,
        "NASA OMPS · aerosol indeksi (duman/toz)": aerosolLayer,
        "EUMETSAT MTG · GeoColour (yaklaşık 10 dk)": eumetsatLayer,
        "İl / ilçe sınırları": districtLayer,
        "Yakın çevre FIRMS kayıtları": nearbyLayer,
      },
      { collapsed: true, position: "topright" },
    ).addTo(map);
    addMapFilterControl();
    const imageryStatus = $("imagery-status");
    const refreshImageryStatus = () => {
      if (!imageryStatus) return;
      const date = $("imagery-date")?.value || imageDate;
      const visible = map.hasLayer(viirsImageLayer) || map.hasLayer(aerosolLayer);
      imageryStatus.textContent = visible
        ? `NASA uydu görüntüsü: ${date} · FIRMS ile otomatik eşleştirme yok`
        : "Uydu görüntüleri kapalı · katmanlar bağımsız";
    };
    map.on("overlayadd overlayremove", refreshImageryStatus);
    const eumetsatStatus = $("eumetsat-status");
    const updateEumetsatLayer = () => {
      if (!map.hasLayer(eumetsatLayer)) return;
      const requestTime = latestCompleteMtgTime();
      eumetsatLayer.setParams({ TIME: requestTime }, true);
      if (eumetsatStatus) eumetsatStatus.textContent = `EUMETSAT MTG · görüntü zamanı istenen: ${requestTime.slice(0, 16).replace("T", " ")} UTC`;
    };
    map.on("overlayadd", (event) => {
      if (event.layer !== eumetsatLayer) return;
      updateEumetsatLayer();
      if (eumetsatRefreshTimer) clearInterval(eumetsatRefreshTimer);
      eumetsatRefreshTimer = window.setInterval(updateEumetsatLayer, 10 * 60 * 1000);
    });
    map.on("overlayremove", (event) => {
      if (event.layer !== eumetsatLayer) return;
      if (eumetsatRefreshTimer) clearInterval(eumetsatRefreshTimer);
      eumetsatRefreshTimer = null;
      if (eumetsatStatus) eumetsatStatus.textContent = "EUMETSAT görüntüsü kapalı";
    });
    eumetsatLayer.on("tileerror", () => {
      if (eumetsatStatus && map.hasLayer(eumetsatLayer)) eumetsatStatus.textContent = "EUMETSAT görüntüsü alınamadı; FIRMS katmanı bundan etkilenmez.";
    });
    eumetsatLayer.on("load", () => {
      if (eumetsatStatus && map.hasLayer(eumetsatLayer)) eumetsatStatus.textContent = `EUMETSAT MTG · WMS görüntü isteği: ${latestCompleteMtgTime().slice(0, 16).replace("T", " ")} UTC`;
    });
    [viirsImageLayer, aerosolLayer].forEach((layer) => {
      layer.on("tileerror", () => {
        if (imageryStatus && (map.hasLayer(viirsImageLayer) || map.hasLayer(aerosolLayer))) {
          imageryStatus.textContent = "NASA görüntüsü bu tarih için alınamadı; FIRMS verisi bundan etkilenmez.";
        }
      });
      layer.on("load", refreshImageryStatus);
    });
    $("imagery-date")?.addEventListener("change", () => {
      const date = $("imagery-date").value;
      viirsImageLayer.setUrl(gibsTileUrl("VIIRS_SNPP_CorrectedReflectance_TrueColor", date, "jpg"));
      aerosolLayer.setUrl(gibsTileUrl("OMPS_Aerosol_Index", date, "png"));
      refreshImageryStatus();
    });
    map.on("click", (event) => {
      if (selectedProvinceCode) {
        const province = provinceFeatures.find((feature) => feature.properties.province_code === selectedProvinceCode);
        if (province && !pointInProvince(event.latlng.lng, event.latlng.lat, province)) {
          showAllProvinces();
          return;
        }
      }
      if (selectedItem) { $("det").hidden = true; selectedItem = null; }
    });
  }
  function districtStyle(feature) {
    const props = feature?.properties || {};
    const isProvince = props.shapeType === "ADM1" || Boolean(props.name);
    const active = isProvince
      ? props.province_code === selectedProvinceCode
      : normalize(props.shapeName || props.name) === normalize(selectedDistrictName());
    const ffmc = Number(dashboard?.risk?.ffmc);
    const hasFfmc = Boolean($("lf")?.checked && active && dashboard?.risk && Number.isFinite(ffmc));
    const fill = hasFfmc
      ? (ffmc >= 85 ? "#a33c32" : ffmc >= 70 ? "#b9802f" : "#45684f")
      : (active ? "#a9c39c" : "#82947c");
    return {
      color: active ? "#9b7027" : "#536452",
      weight: active ? 3 : isProvince ? 1.5 : 1.15,
      opacity: active ? .98 : .78,
      fillColor: fill,
      fillOpacity: hasFfmc ? .56 : active ? .3 : isProvince ? .055 : .025,
      dashArray: active || isProvince ? null : "4 3",
    };
  }
  function renderBoundaryData(data, provinceCode = "") {
    districtGeoJson = data;
    if (!Array.isArray(districtGeoJson.features) || !districtGeoJson.features.length) throw new Error("Sınır verisi boş geldi.");
    initializeMap();
    districtLayer.clearLayers(); districtByName.clear();
    districtLayer.addData(districtGeoJson);
    const bounds = districtLayer.getBounds();
    $("boundary-status").textContent = provinceCode
      ? `${districtGeoJson.features.length} ilçe · il dışına tıkla: 81 il`
      : "81 il sınırı · OpenStreetMap";
    if (bounds.isValid()) map.fitBounds(bounds.pad(.04), { maxZoom: provinceCode ? 10 : 6, animate: true, padding: [18, 18] });
    updateDistrictStyle(false);
  }
  async function loadBoundaries(provinceCode) {
    try {
      const response = await fetch(`/api/district-boundaries?province=${encodeURIComponent(provinceCode)}`, { headers: { Accept: "application/geo+json, application/json" } });
      if (!response.ok) throw new Error(`OSM sınır verisi alınamadı (${response.status}).`);
      renderBoundaryData(await response.json(), provinceCode);
    } catch (error) {
      $("boundary-status").textContent = error.message;
      $("boundary-status").classList.add("error-text");
      throw error;
    }
  }
  async function selectProvince(code) {
    if (!code) return;
    selectedProvinceCode = code;
    selectedDayFilter = "";
    nearbyResults = null;
    nearbyLocation = null;
    nearbyLayer?.clearLayers();
    $("nearby-panel").hidden = true;
    provinceSelect.value = code;
    const province = provinceFeatures.find((feature) => feature.properties.province_code === code);
    const provinceName = province?.properties.name || "Seçili il";
    select.disabled = true;
    select.innerHTML = `<option value="">İlçe listesi alınıyor…</option>`;
    dashboard = { hotspots: [], risk: null, weather: null };
    selectedItem = null;
    $("det").hidden = true;
    try {
      const response = await fetch(`/api/districts?province=${encodeURIComponent(code)}`, { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`İlçe listesi alınamadı (${response.status}).`);
      const districts = await response.json();
      select.innerHTML = `<option value="">${escapeHtml(provinceName)} · ilçe seçin</option>` + districts.map(({ name }) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
      select.disabled = false;
      await loadBoundaries(code);
      live.textContent = `${provinceName} · uydu noktaları yükleniyor…`;
      live.dataset.state = "";
      setSourceStatus("Seçili ilin güncel uydu noktaları yükleniyor…", false);
      renderAll();
      await loadDashboard();
    } catch (error) {
      select.innerHTML = `<option value="">İlçe verisi yüklenemedi</option>`;
      select.disabled = true;
      live.textContent = "İlçe sınırları alınamadı";
      setSourceStatus(error.message, true);
    }
  }
  function showAllProvinces() {
    selectedProvinceCode = "";
    selectedDayFilter = "";
    provinceSelect.value = "";
    select.disabled = true;
    select.innerHTML = `<option value="">Önce il seçin</option>`;
    const hours = Number($("ft").value || 24);
    const cachedCountry = countryDashboardCache.get(hours);
    dashboard = cachedCountry || { hotspots: [], risk: null, weather: null };
    selectedItem = null; $("det").hidden = true;
    markerLayer?.clearLayers();
    nearbyResults = null;
    nearbyLocation = null;
    nearbyLayer?.clearLayers();
    $("nearby-panel").hidden = true;
    if (provinceBoundaryData) {
      renderBoundaryData(provinceBoundaryData);
      setSourceStatus(cachedCountry?.firms_status || "Türkiye genelindeki güncel uydu noktaları yükleniyor…", false);
      if (cachedCountry) {
        const displayTime = cachedCountry.updated_at ? new Date(cachedCountry.updated_at) : null;
        live.textContent = `Türkiye geneli · kayıtlı noktalar gösteriliyor${displayTime ? ` · son sorgu ${displayTime.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" })}` : ""} · güncelleniyor…`;
        live.dataset.state = cachedCountry.hotspots?.length ? "live" : "";
      }
      renderAll();
      loadDashboard();
    }
  }
  function updateDistrictStyle(fit = false) {
    if (!districtLayer) return;
    districtLayer.setStyle(districtStyle);
    const active = districtByName.get(normalize(selectedDistrictName()));
    if (active) {
      active.bringToFront();
      if (fit) map.fitBounds(active.getBounds().pad(.08), { maxZoom: 10 });
    }
  }
  function setMapSummary(items) {
    const scores = items.map((item) => Number(item.priority_score)).filter(Number.isFinite);
    const modelRanks = items.map((item) => Number(item.model_rank_percentile)).filter(Number.isFinite);
    const latest = items.map((item) => rowTime(lastObservation(item))).filter(Boolean).sort((a, b) => b - a)[0];
    const total = Number(dashboard?.hotspots_total ?? items.length);
    const countLabel = `${items.length} / ${total}`;
    const hasModelMap = Array.isArray(dashboard?.map_hotspots);
    const modelCoverage = hasModelMap
      ? dashboard.map_hotspots.filter((item) => hasNumber(item.model_rank_percentile)).length
      : modelRanks.length;
    const modelStatus = dashboard?.model_status || "Model durumu bekleniyor.";
    const validation = dashboard?.model_validation;
    const auc = Number(validation?.auc_mean);
    const aucRange = Array.isArray(validation?.auc_range) && validation.auc_range.length === 2
      ? ` · kat aralığı ${number(validation.auc_range[0], 3)}–${number(validation.auc_range[1], 3)}`
      : "";
    const modelMetric = Number.isFinite(auc)
      ? `<small>4 coğrafi kat AUC ${number(auc, 3)}${aucRange} · FIRMS proxy doğrulaması</small>`
      : "";
    const hasRank = hasModelMap ? modelCoverage > 0 : modelRanks.length > 0;
    const modelCard = `<div class="st model-rank-card"><b>${hasRank ? (hasModelMap ? modelCoverage : modelRanks.length).toLocaleString("tr-TR") : "—"}</b><small>Model sıralaması · uygun nokta</small><small class="model-card-status">${escapeHtml(modelStatus)}</small>${modelMetric}</div>`;
    const filterContent = `<div class="map-filter-control"><button class="map-filter-toggle" type="button" aria-expanded="${mapFiltersOpen}" aria-controls="map-filters-panel">☷ Filtreler</button><section id="map-filters-panel" class="map-filters-panel"${mapFiltersOpen ? "" : " hidden"}><div class="map-filter-head"><b>Harita filtreleri</b><button class="map-filter-close" type="button" aria-label="Filtreleri kapat" title="Filtreleri kapat">×</button></div><div class="filter-group"><b>Nokta rengi <small>Birden fazla seçilebilir</small></b><div class="filter-options"><button class="filter-option" data-filter="color" data-value="all">Tümü</button><button class="filter-option" data-filter="color" data-value="low">🟢 Yeşil</button><button class="filter-option" data-filter="color" data-value="mid">🟡 Sarı</button><button class="filter-option" data-filter="color" data-value="high">🔴 Kırmızı</button></div><small>Model sırası varsa: yeşil &lt;50, sarı 50–74, kırmızı ≥75. Model sırası yoksa aynı renkler saha kontrolü puanı bandını gösterir.</small></div><div class="filter-group"><b>Hasat ve geçmiş bağlamı <small>Hasat: yaklaşık kışlık hububat takvimi · kümelenme: 7+ günlük geçmiş</small></b><div class="filter-options"><button class="filter-option" data-filter="harvest" data-value="on">Hububat hasat dönemi</button><button class="filter-option" data-filter="cluster" data-value="on">Geçmişe göre kümelenme</button></div></div><div class="filter-group"><b>Zaman aralığı</b><div class="filter-options"><button class="filter-option" data-filter="hours" data-value="6">6 saat</button><button class="filter-option" data-filter="hours" data-value="24">24 saat</button><button class="filter-option" data-filter="hours" data-value="48">48 saat</button><button class="filter-option" data-filter="hours" data-value="96">4 gün</button></div></div><div class="filter-group"><b>Uydu güveni</b><div class="filter-options"><button class="filter-option" data-filter="confidence" data-value="high">Yüksek</button><button class="filter-option" data-filter="confidence" data-value="hn">Yüksek + normal</button><button class="filter-option" data-filter="confidence" data-value="all">Tümü</button></div></div><div class="filter-group"><b>Arazi türü <small>Birden fazla seçilebilir</small></b><div class="filter-options"><button class="filter-option" data-filter="land" data-value="all">Tümü</button><button class="filter-option" data-filter="land" data-value="t">Tarım</button><button class="filter-option" data-filter="land" data-value="o">Orman / doğal</button><button class="filter-option" data-filter="land" data-value="s">Yapı / sanayi</button></div></div><div class="filter-group"><b>Uydu <small>Birden fazla seçilebilir</small></b><div class="filter-options"><button class="filter-option" data-filter="satellite" data-value="all">Tümü</button><button class="filter-option" data-filter="satellite" data-value="N">S-NPP</button><button class="filter-option" data-filter="satellite" data-value="1">NOAA-20</button><button class="filter-option" data-filter="satellite" data-value="2">NOAA-21</button><button class="filter-option" data-filter="satellite" data-value="A">MODIS Aqua</button><button class="filter-option" data-filter="satellite" data-value="T">MODIS Terra</button></div></div><div class="filter-group"><b>Harita göstergesi</b><div class="filter-options"><button class="filter-option" data-filter="ffmc" data-value="on">FFMC göstergesi</button></div><p class="ffmc-status" aria-live="polite"></p><div class="ffmc-legend"><span><i class="low"></i>Düşük &lt;70</span><span><i class="medium"></i>Orta 70–84</span><span><i class="high"></i>Yüksek ≥85</span></div></div><p class="filter-result" aria-live="polite"></p></section></div>`;
    $("stats").innerHTML = `<div class="st map-count-card"><b>${countLabel}</b><small>Haritada gösterilen / toplam</small>${filterContent}</div><div class="st"><b>${scores.length ? Math.max(...scores) : "—"}</b><small>En yüksek saha kontrolü puanı</small></div>${modelCard}<div class="st"><b>${latest ? escapeHtml(formatTime(latest)) : "—"}</b><small>Son gözlem</small></div>${dashboard?.weather ? `<div class="st"><b>${number(dashboard.weather.temperature)}°</b><small>${number(dashboard.weather.humidity, 0)}% nem · ${number(dashboard.weather.wind_speed)} km/sa</small></div><div class="st"><b>${dashboard.risk ? number(dashboard.risk.ffmc) : "—"}</b><small>FFMC · seçili ilçe hava noktası</small></div>` : ""}`;
    updateMapFilterControl();
  }
  function satelliteName(code) {
    return ({ N: "S-NPP", "1": "NOAA-20", "2": "NOAA-21", A: "MODIS Aqua", T: "MODIS Terra" })[String(code)] || String(code || "Uydu bilinmiyor");
  }
  function addMapFilterControl() {
    const root = $("stats");
    if (!root || root.dataset.filtersBound) return;
    root.dataset.filtersBound = "true";
    root.addEventListener("click", (event) => {
      const close = event.target.closest(".map-filter-close");
      if (close) {
        event.stopPropagation();
        mapFiltersOpen = false;
        const panel = root.querySelector(".map-filters-panel");
        const toggle = root.querySelector(".map-filter-toggle");
        if (panel) panel.hidden = true;
        if (toggle) {
          toggle.setAttribute("aria-expanded", "false");
          toggle.focus();
        }
        return;
      }
      const toggle = event.target.closest(".map-filter-toggle");
      if (toggle) {
        event.stopPropagation();
        mapFiltersOpen = !mapFiltersOpen;
        const panel = root.querySelector(".map-filters-panel");
        if (panel) panel.hidden = !mapFiltersOpen;
        toggle.setAttribute("aria-expanded", String(mapFiltersOpen));
        return;
      }
      const button = event.target.closest(".filter-option");
      if (!button) return;
      event.stopPropagation();
      const group = button.dataset.filter, value = button.dataset.value;
      if (group === "hours") {
        $("ft").value = value;
        selectedDayFilter = "";
        if (nearbyLocation && !selectedProvinceCode) loadNearbyAt(nearbyLocation.lat, nearbyLocation.lon);
        else loadDashboard();
      } else if (group === "confidence") {
        $("fc").value = value;
        renderAll();
      } else if (group === "land") {
        if (value === "all") landFilterSelection.clear();
        else landFilterSelection.has(value) ? landFilterSelection.delete(value) : landFilterSelection.add(value);
        renderAll();
      } else if (group === "satellite") {
        if (value === "all") satelliteFilterSelection.clear();
        else satelliteFilterSelection.has(value) ? satelliteFilterSelection.delete(value) : satelliteFilterSelection.add(value);
        renderAll();
      } else if (group === "color") {
        if (value === "all") markerColorSelection.clear();
        else markerColorSelection.has(value) ? markerColorSelection.delete(value) : markerColorSelection.add(value);
        renderAll();
      } else if (group === "harvest") {
        harvestPeriodOnly = !harvestPeriodOnly;
        renderAll();
      } else if (group === "cluster") {
        clusterAnomalyOnly = !clusterAnomalyOnly;
        renderAll();
      } else if (group === "ffmc") {
        $("lf").checked = !$("lf").checked;
        updateDistrictStyle(false);
      }
      updateMapFilterControl(root);
    });
    root.addEventListener("wheel", (event) => {
      if (event.target.closest(".map-filters-panel")) event.stopPropagation();
    }, { passive: true });
  }
  function updateMapFilterControl(root = document.querySelector(".map-filter-control")) {
    if (!root) return;
    root.querySelectorAll(".filter-option").forEach((button) => {
      const { filter, value } = button.dataset;
      let active = false;
      if (filter === "hours") active = $("ft").value === value;
      if (filter === "confidence") active = $("fc").value === value;
      if (filter === "land") active = value === "all" ? landFilterSelection.size === 0 : landFilterSelection.has(value);
      if (filter === "satellite") active = value === "all" ? satelliteFilterSelection.size === 0 : satelliteFilterSelection.has(value);
      if (filter === "color") active = value === "all" ? markerColorSelection.size === 0 : markerColorSelection.has(value);
      if (filter === "harvest") active = harvestPeriodOnly;
      if (filter === "cluster") active = clusterAnomalyOnly;
      if (filter === "ffmc") active = $("lf").checked;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    const activeCount = landFilterSelection.size + satelliteFilterSelection.size + markerColorSelection.size
      + (harvestPeriodOnly ? 1 : 0) + (clusterAnomalyOnly ? 1 : 0)
      + ($("fc").value === "all" ? 0 : 1) + ($("ft").value === "24" ? 0 : 1) + ($("lf").checked ? 1 : 0);
    const toggle = root.querySelector(".map-filter-toggle");
    toggle.innerHTML = `☷ Filtreler${activeCount ? ` <span>${activeCount}</span>` : ""}`;
    const mapRows = Array.isArray(dashboard?.map_hotspots) ? dashboard.map_hotspots : dashboard?.hotspots;
    root.querySelector(".filter-result").textContent = `${filteredItems(mapRows).length.toLocaleString("tr-TR")} bildirim haritada gösteriliyor`;
    const ffmcStatus = root.querySelector(".ffmc-status");
    const ffmc = Number(dashboard?.risk?.ffmc);
    if (!$("lf").checked) ffmcStatus.textContent = "Katman kapalı.";
    else if (!selectedDistrictName()) ffmcStatus.textContent = "Renkli gösterim için il ve ilçe seçin. Ülke veya il görünümüne tek bir FFMC değeri uygulanmaz.";
    else if (!dashboard?.risk || !Number.isFinite(ffmc)) ffmcStatus.textContent = "Seçili ilçe için FFMC hava verisi alınamadı; sınır renklendirilmedi.";
    else ffmcStatus.textContent = `${selectedDistrictName()} · FFMC ${number(ffmc)} · ${ffmc >= 85 ? "Yüksek" : ffmc >= 70 ? "Orta" : "Düşük"}`;
  }
  function filteredItems(source = dashboard?.hotspots) {
    let rows = Array.isArray(source) ? source : [];
    const confidenceFilter = $("fc").value;
    const satelliteAliases = { N: ["n", "snpp"], "1": ["n20", "noaa-20", "1"], "2": ["n21", "noaa-21", "2"], A: ["aqua", "a"], T: ["terra", "t"] };
    const query = normalize($("q").value.trim());
    if (confidenceFilter === "high") rows = rows.filter((item) => confidenceClass(lastObservation(item)?.confidence ?? item.confidence) === "high");
    if (confidenceFilter === "hn") rows = rows.filter((item) => ["high", "normal"].includes(confidenceClass(lastObservation(item)?.confidence ?? item.confidence)));
    if (markerColorSelection.size) rows = rows.filter((item) => markerColorSelection.has(markerColorBand(item)));
    if (harvestPeriodOnly) rows = rows.filter((item) => item.harvest_context?.in_period === true);
    if (clusterAnomalyOnly) rows = rows.filter((item) => item.cluster_anomaly === true);
    if (landFilterSelection.size) rows = rows.filter((item) => landFilterSelection.has(landType(item).code));
    if (satelliteFilterSelection.size) rows = rows.filter((item) => [...satelliteFilterSelection].some((satelliteFilter) => {
      const aliases = satelliteAliases[satelliteFilter] || [satelliteFilter.toLowerCase()];
      return aliases.includes(normalize(item.satellite)) || observations(item).some((ob) => aliases.includes(normalize(ob.satellite)));
    }));
    if (query) rows = rows.filter((item) => normalize(`${item.district || selectedDistrictName()} ${item.province || ""} ${item.candidate} ${item.corine_class} ${item.latitude} ${item.longitude}`).includes(query));
    return rows.sort((a, b) => {
      const rankA = hasNumber(a.model_rank_percentile) ? Number(a.model_rank_percentile) : null;
      const rankB = hasNumber(b.model_rank_percentile) ? Number(b.model_rank_percentile) : null;
      if (rankA !== null || rankB !== null) {
        if (rankA === null) return 1;
        if (rankB === null) return -1;
        if (rankA !== rankB) return rankB - rankA;
      }
      const timeA = timeFromRow(a)?.getTime() ?? -Infinity;
      const timeB = timeFromRow(b)?.getTime() ?? -Infinity;
      return timeB - timeA || (Number(b.priority_score) || 0) - (Number(a.priority_score) || 0);
    });
  }
  function renderList(items) {
    $("count").textContent = String(items.length);
    if (!items.length) {
      const area = selectedDistrictName() || (selectedProvinceCode ? provinceSelect.selectedOptions[0]?.textContent : "Türkiye geneli");
      $("list").innerHTML = `<p class="empty-note">${escapeHtml(area)} için seçili zaman aralığında FIRMS kaydı bulunamadı. Bu, bölgede yangın olmadığı anlamına gelmez.</p>${loadMoreMarkup()}`;
      bindLoadMore();
      return;
    }
    $("list").innerHTML = items.map((item, index) => {
      const obs = lastObservation(item), priority = scoreInfo(item.priority_score), type = eventType(item);
      const stamp = timeFromRow(item) || rowTime(obs);
      const frp = obs?.frp === null || obs?.frp === undefined ? "—" : `${number(obs.frp)} MW`;
      const area = [item.district, item.province].filter(Boolean).join(" · ");
      const contextBadge = `${item.cluster_anomaly ? " · Geçmişe göre kümelenme" : ""}${hasNumber(item.model_rank_percentile) ? ` · Model sırası ${number(item.model_rank_percentile, 0)}/100` : ""}`;
      return `<button class="alert-row ${selectedItem === item ? "selected" : ""}" type="button" data-key="${hotspotKey(item)}" data-index="${index}" aria-label="${escapeHtml(type.label)} · ${escapeHtml(formatTime(stamp))}"><i class="alert-indicator" style="background:${type.color}" aria-hidden="true"></i><span><span class="alert-title">${escapeHtml(type.label)}</span><span class="alert-meta">${escapeHtml(formatTime(stamp))} · ${escapeHtml(satelliteName(obs?.satellite ?? item.satellite))}${area ? ` · ${escapeHtml(area)}` : ""} · ${escapeHtml(priority.label)}${escapeHtml(contextBadge)}</span></span><span class="alert-value">${escapeHtml(frp)}<small>${priority.score === null ? "" : `${priority.score}/100`}</small></span></button>`;
    }).join("") + loadMoreMarkup();
    $("list").querySelectorAll(".alert-row").forEach((button) => {
      const item = items[Number(button.dataset.index)];
      button.addEventListener("click", () => showDetail(item, true));
      button.addEventListener("mouseenter", () => setHotspotHover(button.dataset.key, true));
      button.addEventListener("mouseleave", () => setHotspotHover(button.dataset.key, false));
    });
    bindLoadMore();
  }
  function loadMoreMarkup() {
    if (!dashboard?.hotspots_has_more) return "";
    const pageSize = Number(dashboard.hotspots_page_size || 100);
    return `<button class="load-more-button" type="button" id="load-more">Daha eski ${pageSize} bildirimi yükle</button>`;
  }
  function bindLoadMore() {
    $("load-more")?.addEventListener("click", () => loadDashboard({ offset: (dashboard?.hotspots || []).length }));
  }
  function hotspotKey(item) { return `${Number(item.latitude).toFixed(5)},${Number(item.longitude).toFixed(5)}`; }
  function setHotspotHover(key, active) {
    if (!key) return;
    document.querySelector(`.alert-row[data-key="${key}"]`)?.classList.toggle("hover-linked", active);
    const marker = markerByKey.get(key);
    if (marker) {
      marker.setZIndexOffset(active ? 1000 : 0);
      marker.getElement()?.classList.toggle("hovered", active);
    }
  }
  function renderMarkers(items) {
    if (!markerLayer) return;
    markerLayer.clearLayers();
    markerByKey.clear();
    items.forEach((item) => {
      const lat = Number(item.latitude), lon = Number(item.longitude);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
      const priority = scoreInfo(item.priority_score), land = landType(item), type = eventType(item);
      const markerBand = markerColorBand(item);
      const isRedAlert = markerBand === "high";
      const iconSize = isRedAlert ? 28 : 20;
      const historyClass = item.cluster_anomaly ? " history-cluster" : "";
      const icon = L.divIcon({ className: `hotspot-icon priority-${markerBand}${historyClass}`, html: `<span class="hotspot-dot" style="--land-color:${land.color};--event-color:${type.color}"></span>`, iconSize: [iconSize + (item.cluster_anomaly ? 10 : 0), iconSize + (item.cluster_anomaly ? 10 : 0)], iconAnchor: [(iconSize + (item.cluster_anomaly ? 10 : 0)) / 2, (iconSize + (item.cluster_anomaly ? 10 : 0)) / 2] });
      const zIndexOffset = markerBand === "high" ? 1400 : markerBand === "mid" ? 500 : 0;
      const markerTitle = hasNumber(item.model_rank_percentile)
        ? `Model göreli sırası ${number(item.model_rank_percentile, 0)}/100`
        : `${isRedAlert ? "Yüksek saha kontrolü önceliği" : "Uydu bildirimi"}`;
      const marker = L.marker([lat, lon], { icon, keyboard: true, riseOnHover: true, zIndexOffset, title: `${markerTitle} · ${land.name} · ${type.label}` }).addTo(markerLayer);
      const latest = lastObservation(item), score = scoreInfo(item.priority_score);
      const frpLabel = latest?.frp === null || latest?.frp === undefined ? "FRP ölçümü yok" : `FRP ${number(latest.frp, 2)} MW`;
      marker.bindTooltip(`<b>${escapeHtml(type.label)}</b><br>${escapeHtml(land.name)} · ${escapeHtml(frpLabel)}<br>${hasNumber(item.model_rank_percentile) ? `Türkiye model göreli sırası ${number(item.model_rank_percentile, 0)}/100<br>` : ""}${score.score === null ? "Saha kontrolü puanı yok" : `Saha kontrolü ${score.score}/100 · ${escapeHtml(score.label)}`}${item.cluster_anomaly ? `<br>Tarihsel kümelenme: ${number(item.cluster_anomaly_score, 0)}/100` : ""}`, { direction: "top", offset: [0, -8], opacity: .98, className: "hotspot-tooltip" });
      const key = hotspotKey(item);
      markerByKey.set(key, marker);
      marker.on("mouseover", () => setHotspotHover(key, true));
      marker.on("mouseout", () => setHotspotHover(key, false));
      // A hotspot click zooms to the point and opens its details, while keeping
      // the country scope intact; only province polygons change the scope.
      marker.on("click", () => showDetail(item, true));
    });
  }
  function detailMarkup(item, address = "Adres sorgulanıyor…") {
    const latest = lastObservation(item), priority = scoreInfo(item.priority_score), breakdown = scoreBreakdown(latest), land = landType(item), type = eventType(item);
    const first = item.acq_date && item.acq_time !== null ? formatTime(timeFromRow(item)) : "—";
    const currentTime = formatTime(rowTime(latest));
    const frp = latest?.frp === null || latest?.frp === undefined ? "Ölçüm yok" : `${number(latest.frp, 2)} MW`;
    const lat = Number(item.latitude), lon = Number(item.longitude);
    const coords = Number.isFinite(lat) && Number.isFinite(lon) ? `${lat.toFixed(5)}, ${lon.toFixed(5)}` : "Koordinat yok";
    const season = item.harvest_context || {};
    const modelRank = hasNumber(item.model_rank_percentile) ? `${number(item.model_rank_percentile, 0)}/100 · gösterilen uygun noktalar içindeki sıra` : "Hesaplanmadı";
    const harvestWindow = season.window_start && season.window_end ? `${harvestDateLabel(season.window_start.slice(5))}–${harvestDateLabel(season.window_end.slice(5))}` : "bölge takvimi yok";
    const clusterValue = hasNumber(item.cluster_anomaly_score)
      ? `${number(item.cluster_anomaly_score, 0)}/100 · ${number(item.cluster_daily_count, 0)} piksel / 5 km hücre`
      : "Yeterli günlük geçmiş yok veya bu olay için hesaplanmadı";
    return `<div class="detail-head"><div><h3>${escapeHtml(type.label)}</h3><small>${escapeHtml(satelliteName(item.satellite))} · FIRMS termal gözlemi</small></div><span class="priority-pill">${priority.score === null ? "—" : `${priority.score}/100`} · ${escapeHtml(priority.label)}</span></div>
      <dl class="detail-grid"><div><dt>İlk tespit</dt><dd>${escapeHtml(first)}</dd></div><div><dt>Son uydu gözlemi</dt><dd>${escapeHtml(currentTime)}<small>${escapeHtml(relativeTime(rowTime(latest)))} · güncelleme sorgusu ${escapeHtml(dashboard?.updated_at ? formatTime(dashboard.updated_at) : "—")}</small></dd></div><div><dt>Adres</dt><dd id="detail-address">${escapeHtml(address)}<small>${escapeHtml(coords)}</small></dd></div><div><dt>Arazi türü</dt><dd>${escapeHtml(land.name)}</dd></div><div><dt>Son FRP</dt><dd>${escapeHtml(frp)}<small>Ateş ışıma gücü; alan ölçümü değildir.</small></dd></div><div><dt>Güven / TI4</dt><dd>${escapeHtml(latest?.confidence ?? "—")} / ${latest?.bright_ti4 === null || latest?.bright_ti4 === undefined ? "—" : `${number(latest.bright_ti4, 1)} K`}</dd></div></dl>
      <details class="score-breakdown"><summary>Son öncelik puanının hesabı</summary><p>En yeni gerçek uydu gözleminin değerleri kullanıldı.</p><ul>${breakdown.rows}</ul><p>Sonuç: <b>${breakdown.total === null ? "Puanlanamadı" : `${breakdown.total}/100`}</b></p><small>Güven sınıfı: yüksek 100, normal 65, düşük 35. FRP: 100 × FRP / (FRP + 10). TI4: (TI4 − 300) / 0,6; 0–100 arasında sınırlandırılır.${breakdown.missing ? " Eksik bileşenlerde kalan ağırlık yeniden ölçeklendirildi." : ""} Bu puan yangın olasılığı değildir.</small></details>
      <section class="evidence-context"><h4>Mevsim ve geçmiş bağlamı</h4><p><b>${escapeHtml(season.label || "Takvim bilgisi yok")} · yaklaşık dönem ${escapeHtml(harvestWindow)}</b><small>${escapeHtml(season.note || "Mevsimsel bağlam, olay nedenini doğrulamaz.")}</small></p><p><b>Tarihsel kümelenme: ${escapeHtml(clusterValue)}</b><small>5 km hücredeki günlük FIRMS sayısı, önceki kapsama günlerinin medyanıyla karşılaştırılır. En az 7 günlük geçmiş gerekir. Kesikli mavi halka kümelenmeyi belirtir; yangın doğrulaması değildir.</small></p><p><b>Model sırası: ${escapeHtml(modelRank)}</b><small>Model, ülkenin bölgesel hububat hasat pencerelerine denk gelen tarihli FIRMS kayıtlarıyla eğitilmiştir. İl/bölge bilgisi ve hasat penceresindeki ilerleme model girdisidir; bu nedenle yalnızca noktanın bulunduğu ilin yaklaşık hububat sezonunda, S-NPP/NOAA-20 ve tam hava, CORINE, geçmiş girdileri eşleştiğinde sıralama üretir. Hava girdisi ilçe merkezinden alınır. Eğitim etiketi FIRMS sıcak nokta vekilidir; anız yakımı, yangın olasılığı veya yasallık sonucu değildir.</small></p>${season.sources?.length ? `<small>Takvim kaynakları: ${season.sources.map((source) => `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener">${escapeHtml(source.label)}</a>`).join(" · ")}</small>` : ""}</section>
      ${frpHistory(item)}
      <div class="detail-actions"><a target="_blank" rel="noopener" href="https://www.openstreetmap.org/?mlat=${encodeURIComponent(lat)}&mlon=${encodeURIComponent(lon)}#map=15/${encodeURIComponent(lat)}/${encodeURIComponent(lon)}">OSM’de aç ↗</a><button type="button" class="copy-coordinates">Koordinatı kopyala</button></div>
      <p class="history-note">Termal bildirim yangın ya da anız yakımı doğrulaması değildir. Haritadaki arazi sınıfı bölgesel bağlam sağlar.</p>`;
  }
  async function showDetail(item, zoom) {
    selectedItem = item;
    const panel = $("det"); panel.hidden = false; panel.innerHTML = detailMarkup(item);
    if (zoom && map) map.setView([Number(item.latitude), Number(item.longitude)], Math.max(map.getZoom(), 12));
    renderList(filteredItems());
    panel.querySelector(".copy-coordinates")?.addEventListener("click", async (event) => {
      const text = `${item.latitude}, ${item.longitude}`;
      try { await navigator.clipboard.writeText(text); event.currentTarget.textContent = "Kopyalandı"; }
      catch (_) { event.currentTarget.textContent = text; }
    });
    const lat = Number(item.latitude), lon = Number(item.longitude);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
    try {
      const response = await fetch(`/api/reverse-geocode?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`, { headers: { Accept: "application/json" } });
      if (!response.ok) return;
      const data = await response.json();
      const address = panel.querySelector("#detail-address");
      if (address && selectedItem === item) address.innerHTML = `${escapeHtml(data.address || "Yakın adres bulunamadı")}<small>${Number(lat).toFixed(5)}, ${Number(lon).toFixed(5)}</small>`;
    } catch (_) { /* Keep coordinates visible if reverse geocoding is unavailable. */ }
  }
  function setSourceStatus(message, isError) {
    const node = $("firms-status"); node.textContent = message || "NASA FIRMS yanıtı yok."; node.dataset.state = isError ? "error" : "";
  }
  function setWeatherStatus(message, isError) {
    const node = $("weather-status");
    if (!node) return;
    node.textContent = message || "Hava durumu yanıtı yok.";
    node.dataset.state = isError ? "error" : "";
  }
  function setPeriodSummary(data) {
    const node = $("period-summary");
    if (!node) return;
    const days = Array.isArray(data?.hotspots_by_day) ? data.hotspots_by_day : [];
    const total = days.reduce((sum, day) => sum + Number(day.count || 0), 0);
    const hours = Number(data?.hours) || 6;
    const label = hours === 96 ? "Son 4 gün" : hours === 48 ? "Son 48 saat" : hours === 24 ? "Son 24 saat" : `Son ${hours} saat`;
    if (!total || !days.length) {
      node.textContent = `${label}: kayıt yok. Eksik uydu geçişleri sıfır tespit anlamına gelmez.`;
      return;
    }
    const daily = days.map((day) => {
      const date = new Date(`${day.date}T00:00:00Z`);
      const shortDate = new Intl.DateTimeFormat("tr-TR", { day: "2-digit", month: "2-digit", timeZone: "UTC" }).format(date);
      return `<button class="day-filter${selectedDayFilter === day.date ? " active" : ""}" type="button" data-day="${escapeHtml(day.date)}">${shortDate}<b>${Number(day.count).toLocaleString("tr-TR")}</b></button>`;
    }).join("");
    node.innerHTML = `<span class="period-total">${label}: ${total.toLocaleString("tr-TR")} tespit</span><span class="day-filters"><button class="day-filter${selectedDayFilter ? "" : " active"}" type="button" data-day="">Tümü</button>${daily}</span>`;
    node.querySelectorAll(".day-filter").forEach((button) => button.addEventListener("click", () => {
      selectedDayFilter = button.dataset.day || "";
      loadDashboard({ offset: 0, day: selectedDayFilter });
    }));
  }
  function setAnalysisStatus(data) {
    const node = $("analysis-status");
    if (!node) return;
    const history = data?.history_status || "Tarihsel kümelenme verisi henüz yok.";
    const model = data?.model_status || "Model sıralaması henüz hesaplanmadı.";
    node.textContent = `${history} · ${model}`;
    node.dataset.state = /gizli|eksik|başlangıç|güncellenmedi|üretilmedi|eğitim kapsam|eğitimindeki|seçin|eşleşmedi|bekleniyor/i.test(`${history} ${model}`) ? "" : "live";
  }
  function setNearbyPanelMessage(message, isError = false) {
    const panel = $("nearby-panel");
    panel.hidden = false;
    panel.innerHTML = `<div class="nearby-head"><b>Yakın çevre</b><button type="button" class="nearby-close" aria-label="Yakın çevre panelini kapat">×</button></div><p class="nearby-message${isError ? " is-error" : ""}">${escapeHtml(message)}</p>`;
    panel.querySelector(".nearby-close")?.addEventListener("click", () => { panel.hidden = true; nearbyLayer?.clearLayers(); nearbyResults = null; nearbyLocation = null; });
  }
  function renderNearbyPanel(result) {
    const panel = $("nearby-panel");
    panel.hidden = false;
    const counts = result.counts || {};
    const rows = (result.hotspots || []).slice(0, 6).map((item) => {
      const time = formatTime(rowTime(item));
      const frp = item.frp === null || item.frp === undefined || item.frp === "" ? "FRP yok" : `${number(item.frp, 2)} MW`;
      return `<li><span><b>${number(item.distance_km, 1)} km</b><small>${escapeHtml(time)} · ${escapeHtml(satelliteName(item.satellite))}</small></span><span>${escapeHtml(frp)}</span></li>`;
    }).join("");
    const message = !result.available
      ? `<p class="nearby-message is-error">${escapeHtml(result.source_status || "NASA FIRMS verisi alınamadı.")}</p>`
      : result.hotspots_total
        ? `<p class="nearby-message">Son ${result.hours} saatteki termal uydu kayıtları. Bunlar teyit edilmiş yangın değildir.</p>`
        : `<p class="nearby-message">Bu süre ve çevrede FIRMS kaydı yok. Bu, bölgede yangın olmadığını kanıtlamaz.</p>`;
    const radiusStats = result.available ? `<div class="nearby-radii"><div><b>${Number(counts["10"] || 0)}</b><small>10 km</small></div><div><b>${Number(counts["25"] || 0)}</b><small>25 km</small></div><div><b>${Number(counts["50"] || 0)}</b><small>50 km</small></div></div>` : "";
    panel.innerHTML = `<div class="nearby-head"><b>Yakın çevre · ${result.available ? "FIRMS" : "veri yok"}</b><button type="button" class="nearby-close" aria-label="Yakın çevre panelini kapat">×</button></div>${message}${radiusStats}${rows ? `<ol class="nearby-events">${rows}</ol>` : ""}<small class="nearby-source">${escapeHtml(result.source_status || "NASA FIRMS")} · sorgu ${escapeHtml(formatTime(result.checked_at_utc))}</small><small class="nearby-source">${escapeHtml(result.notice || "Termal gözlemler resmî olay kaydı değildir.")}</small>`;
    panel.querySelector(".nearby-close")?.addEventListener("click", () => { panel.hidden = true; nearbyLayer?.clearLayers(); nearbyResults = null; nearbyLocation = null; });
  }
  function renderNearbyMap(result) {
    if (!map || !nearbyLayer) return;
    nearbyLayer.clearLayers();
    const lat = Number(result.center.latitude), lon = Number(result.center.longitude);
    [[50000, "#526b57"], [25000, "#a87931"], [10000, "#a33c32"]].forEach(([radius, color]) => {
      L.circle([lat, lon], { radius, color, weight: 1.4, opacity: .75, fillColor: color, fillOpacity: .035, dashArray: "5 5", interactive: false }).addTo(nearbyLayer);
    });
    L.circleMarker([lat, lon], { radius: 6, color: "#263b2b", weight: 2, fillColor: "#f2c46e", fillOpacity: 1, bubblingMouseEvents: false }).bindTooltip("Seçilen konum").addTo(nearbyLayer);
    (result.hotspots || []).forEach((item) => {
      const point = L.circleMarker([Number(item.latitude), Number(item.longitude)], {
        radius: 5, color: item.priority_color || "#a33c32", weight: 1.5,
        fillColor: item.priority_color || "#a33c32", fillOpacity: .85, bubblingMouseEvents: false,
      });
      const time = formatTime(rowTime(item));
      const frp = item.frp === null || item.frp === undefined || item.frp === "" ? "FRP yok" : `${number(item.frp, 2)} MW`;
      point.bindPopup(`<b>FIRMS termal gözlemi</b><br>${escapeHtml(number(item.distance_km, 1))} km · ${escapeHtml(frp)}<br>${escapeHtml(time)}<br>Yangın doğrulaması değildir.`).addTo(nearbyLayer);
    });
    map.setView([lat, lon], 8, { animate: true });
  }
  async function loadNearbyAt(lat, lon) {
    setNearbyPanelMessage("Son uydu gözlemleri sorgulanıyor…");
    map.setView([lat, lon], 8, { animate: true });
    try {
      const hours = Number($("ft").value || 24);
      const response = await fetch(`/api/nearby?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&hours=${hours}`, { headers: { Accept: "application/json" } });
      if (!response.ok) { const issue = await response.json().catch(() => ({})); throw new Error(issue.detail || `Sunucu yanıtı ${response.status}`); }
      nearbyResults = await response.json();
      renderNearbyMap(nearbyResults);
      renderNearbyPanel(nearbyResults);
      live.textContent = nearbyResults.available ? `Yakın çevre · ${nearbyResults.hotspots_total} kayıt` : "Yakın çevre verisi alınamadı";
      live.dataset.state = nearbyResults.available ? "live" : "error";
    } catch (error) {
      setNearbyPanelMessage(error.message || "Yakın çevre sorgusu tamamlanamadı.", true);
      live.textContent = "Yakın çevre verisi alınamadı";
      live.dataset.state = "error";
    }
  }
  async function searchNearby() {
    if (!navigator.geolocation) {
      setNearbyPanelMessage("Bu tarayıcı konum paylaşımını desteklemiyor.", true);
      return;
    }
    nearbyButton.disabled = true;
    nearbyButton.setAttribute("aria-busy", "true");
    setNearbyPanelMessage("Konum izni bekleniyor. Konum yalnızca yakın çevre sorgusu için kullanılır.");
    try {
      const position = await new Promise((resolve, reject) => navigator.geolocation.getCurrentPosition(resolve, reject, { enableHighAccuracy: false, timeout: 12000, maximumAge: 60000 }));
      const lat = Number(position.coords.latitude), lon = Number(position.coords.longitude);
      if (!(lat >= 35 && lat <= 43 && lon >= 25 && lon <= 46)) throw new Error("Yakın çevre taraması şu anda Türkiye sınırları içinde çalışıyor.");
      showAllProvinces();
      nearbyLocation = { lat, lon };
      await loadNearbyAt(lat, lon);
    } catch (error) {
      const message = error.code === 1
        ? "Konum izni verilmedi. Tarayıcı ayarlarından izin verip yeniden deneyebilirsiniz."
        : error.code === 3
          ? "Konum alınamadı; bağlantı zaman aşımına uğradı. Yeniden deneyin."
          : error.message || "Yakın çevre sorgusu tamamlanamadı.";
      setNearbyPanelMessage(message, true);
      live.textContent = "Yakın çevre verisi alınamadı";
      live.dataset.state = "error";
    } finally {
      nearbyButton.disabled = false;
      nearbyButton.removeAttribute("aria-busy");
    }
  }
  function renderAll() {
    const items = filteredItems();
    const mapRows = Array.isArray(dashboard?.map_hotspots) ? dashboard.map_hotspots : dashboard?.hotspots;
    const mapItems = filteredItems(mapRows);
    setMapSummary(mapItems); renderList(items); renderMarkers(mapItems); updateDistrictStyle(false);
    updateMapFilterControl();
    renderStats(items); renderArchive(items);
    renderHarvestCalendar(dashboard?.harvest_calendar, dashboard?.harvest_calendar_sources);
    if (selectedItem && !items.includes(selectedItem) && !mapItems.includes(selectedItem)) { $("det").hidden = true; selectedItem = null; }
    else if (selectedItem && !$("det").hidden) $("det").innerHTML = detailMarkup(selectedItem, $("det").querySelector("#detail-address")?.textContent || "Adres sorgulanıyor…");
  }
  function harvestDateLabel(value) {
    const parts = String(value || "").split("-").map(Number);
    const months = ["", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"];
    if (parts.length !== 2 || !months[parts[0]] || !parts[1]) return "—";
    return `${parts[1]} ${months[parts[0]]}`;
  }
  function renderHarvestCalendar(rows, sources) {
    const node = $("harvest-calendar");
    if (!node) return;
    if (!Array.isArray(rows) || !rows.length) {
      node.innerHTML = `<p class="empty-note">Takvim verisi şu anda alınamadı.</p>`;
      return;
    }
    node.innerHTML = `<div class="calendar-table-wrap"><table class="harvest-table"><thead><tr><th>Bölge</th><th>Yaklaşık başlangıç</th><th>Yaklaşık bitiş</th></tr></thead><tbody>${rows.map((row) => `<tr><td>${escapeHtml(row.region)}</td><td>${escapeHtml(harvestDateLabel(row.start_mm_dd))}</td><td>${escapeHtml(harvestDateLabel(row.end_mm_dd))}</td></tr>`).join("")}</tbody></table></div>${Array.isArray(sources) && sources.length ? `<p class="history-note">${sources.map((source) => `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener">${escapeHtml(source.label)}</a>`).join(" · ")}</p>` : ""}`;
  }
  function summaryCard(title, entries) {
    return `<article class="summary-card"><h2>${escapeHtml(title)}</h2>${entries.length ? entries.map(([name, count]) => `<div class="summary-row"><span>${escapeHtml(name)}</span><b>${count}</b></div>`).join("") : `<p class="empty-note">Bu aralıkta kayıt yok.</p>`}</article>`;
  }
  function renderStats(items) {
    const scopeLabel = selectedDistrictName() || (selectedProvinceCode ? provinceSelect.selectedOptions[0]?.textContent : "Türkiye geneli");
    if (!items.length) {
      $("stats-content").innerHTML = `<p class="empty-note">${escapeHtml(scopeLabel)} için seçili zaman aralığında özetlenecek gerçek FIRMS kaydı yok.</p>`;
      return;
    }
    const bySat = new Map(), byLand = new Map(), byPriority = new Map(), bySeason = new Map(), byCluster = new Map();
    items.forEach((item) => {
      const obs = lastObservation(item), sat = satelliteName(obs?.satellite ?? item.satellite), land = landType(item).name, pri = scoreInfo(item.priority_score).label;
      bySat.set(sat, (bySat.get(sat) || 0) + 1); byLand.set(land, (byLand.get(land) || 0) + 1); byPriority.set(pri, (byPriority.get(pri) || 0) + 1);
      const season = item.harvest_context?.label || "Takvim bilgisi yok";
      bySeason.set(season, (bySeason.get(season) || 0) + 1);
      const cluster = hasNumber(item.cluster_anomaly_score) ? (item.cluster_anomaly ? "Geçmişe göre yüksek kümelenme" : "Geçmiş karşılaştırması") : "Yeterli geçmiş yok";
      byCluster.set(cluster, (byCluster.get(cluster) || 0) + 1);
    });
    const order = (map) => [...map.entries()].sort((a, b) => b[1] - a[1]);
    const validation = dashboard?.model_validation;
    const modelNote = validation ? `Model AUC: ortalama ${number(validation.auc_mean, 3)}; coğrafi katlar aralığı ${Array.isArray(validation.auc_range) ? validation.auc_range.map((value) => number(value, 3)).join("–") : "—"}. Bu ölçüm FIRMS proxy/control örneklemine aittir (${Number(validation.evaluation_rows || 0).toLocaleString("tr-TR")} kayıt); saha doğrulaması veya gerçek prevalans başarısı değildir.` : (dashboard?.model_status || "Model sıralaması hesaplanmadı.");
    const historyNote = dashboard?.history_status || "Tarihsel kümelenme durumu yok.";
    $("stats-content").innerHTML = `<div class="stats-grid">${summaryCard("Uyduya göre kayıt", order(bySat))}${summaryCard("Arazi bağlamı", order(byLand))}${summaryCard("Saha kontrolü sırası", order(byPriority))}${summaryCard("Hasat takvimi bağlamı", order(bySeason))}${summaryCard("Günlük tarihsel kümelenme", order(byCluster))}</div><p class="history-note">${escapeHtml(historyNote)} Tarihsel kayıt, sunucunun erişebildiği FIRMS sorgularından oluşur; ücretsiz barındırma yeniden başlatıldığında yerel geçmiş sıfırlanabilir.</p><p class="history-note">${escapeHtml(modelNote)} Sıra göstergesi yangın olasılığı değildir. ${escapeHtml(dashboard?.harvest_calendar_status || "Hasat dönemi bölgeye ve yıla göre değişir.")}</p>`;
  }
  function renderArchive(items) {
    const rows = [...items].sort((a, b) => (rowTime(lastObservation(b))?.getTime() || 0) - (rowTime(lastObservation(a))?.getTime() || 0));
    $("archive-content").innerHTML = rows.length ? `<table class="archive-table"><thead><tr><th>Tespit zamanı (UTC)</th><th>Uydu</th><th>Arazi türü</th><th>FRP</th><th>Güven</th><th>Kontrol puanı</th></tr></thead><tbody>${rows.map((item) => { const obs = lastObservation(item), score = scoreInfo(item.priority_score); return `<tr><td>${escapeHtml(formatTime(rowTime(obs)))}</td><td>${escapeHtml(satelliteName(obs?.satellite ?? item.satellite))}</td><td>${escapeHtml(landType(item).name)}</td><td>${obs?.frp === null || obs?.frp === undefined ? "—" : `${number(obs.frp, 2)} MW`}</td><td>${escapeHtml(obs?.confidence ?? "—")}</td><td>${score.score === null ? "—" : `${score.score}/100`}</td></tr>`; }).join("")}</tbody></table>` : `<p class="empty-note">Seçili aralıkta kayıt yok. Bu arayüz kalıcı arşiv tutmuyor.</p>`;
  }
  function setView(view) {
    document.querySelectorAll(".view").forEach((node) => { node.hidden = node.id !== `v-${view}`; });
    document.querySelectorAll(".tab").forEach((node) => node.classList.toggle("on", node.dataset.v === view));
    if (view === "map" && map) setTimeout(() => map.invalidateSize(), 50);
  }
  async function loadDashboard({ force = false, offset = 0, day = selectedDayFilter } = {}) {
    const requestId = ++dashboardRequestId;
    const district = selectedDistrictName(), hours = Number($("ft").value || 24);
    const scopeLabel = district || (selectedProvinceCode ? provinceSelect.selectedOptions[0]?.textContent : "Türkiye geneli");
    const cacheKey = `${selectedProvinceCode || "country"}|${district || ""}|${hours}|${day}|${offset}`;
    const cached = dashboardCache.get(cacheKey);
    if (!force && cached && dashboardDataUsable(cached.data) && Date.now() - cached.savedAt < 5 * 60 * 1000) {
      dashboard = mergeDashboardPage(cached.data, offset, dashboard);
      if (!district && !selectedProvinceCode && !day && !offset && dashboardDataUsable(dashboard)) countryDashboardCache.set(hours, dashboard);
      setSourceStatus(dashboard.firms_status, /alınamadı|gir|geçersiz|hata|timeout|doğrulanamadı|sertifika|bağlantı/i.test(dashboard.firms_status || ""));
      setWeatherStatus(dashboard.weather?.source ? `${dashboard.weather.source} · önbellek` : "Hava özeti yalnız ilçe seçildiğinde hesaplanır.", false);
      setPeriodSummary(dashboard);
      setAnalysisStatus(dashboard);
      live.textContent = `${scopeLabel} · kayıtlı veriler gösteriliyor`;
      live.dataset.state = dashboard.hotspots?.length ? "live" : "";
      renderAll(); updateDistrictStyle(true);
      return;
    }
    const cachedCountry = !district && !selectedProvinceCode ? countryDashboardCache.get(hours) : null;
    refresh.disabled = true; refresh.setAttribute("aria-busy", "true");
    if (!cachedCountry) { live.dataset.state = ""; live.textContent = "Veri alınıyor…"; }
    try {
      const params = new URLSearchParams({ hours: String(hours) });
      if (offset > 0) params.set("offset", String(offset));
      if (day) params.set("day", day);
      if (selectedProvinceCode) params.set("province", selectedProvinceCode);
      if (district) params.set("district", district);
      const response = await fetch(`/api/dashboard?${params.toString()}`, { headers: { Accept: "application/json" } });
      if (!response.ok) { const issue = await response.json().catch(() => ({})); throw new Error(issue.detail || `Sunucu yanıtı ${response.status}`); }
      const pageData = await response.json();
      if (requestId !== dashboardRequestId) return;
      dashboard = mergeDashboardPage(pageData, offset, dashboard);
      if (dashboardDataUsable(pageData)) dashboardCache.set(cacheKey, { savedAt: Date.now(), data: pageData });
      if (!district && !selectedProvinceCode && !day && !offset && dashboardDataUsable(dashboard)) countryDashboardCache.set(hours, dashboard);
      setSourceStatus(dashboard.firms_status, /alınamadı|gir|geçersiz|hata|timeout|doğrulanamadı|sertifika|bağlantı/i.test(dashboard.firms_status || ""));
      setWeatherStatus(dashboard.weather?.source ? `${dashboard.weather.source} · güncel` : "Hava özeti yalnız ilçe seçildiğinde hesaplanır.", false);
      setPeriodSummary(dashboard);
      setAnalysisStatus(dashboard);
      const displayTime = dashboard.updated_at ? new Date(dashboard.updated_at) : new Date();
      live.textContent = `${scopeLabel} · son sorgu ${displayTime.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" })}`;
      live.dataset.state = dashboard.hotspots?.length ? "live" : "";
      document.title = district ? `MaydonOs · ${dashboard.province} · ${district}` : `MaydonOs · ${scopeLabel}`;
      renderAll(); updateDistrictStyle(true);
    } catch (error) {
      if (requestId !== dashboardRequestId) return;
      dashboard = cached?.data || cachedCountry || { hotspots: [], risk: null, weather: null };
      live.textContent = cachedCountry ? "Güncelleme başarısız · son kayıtlı noktalar gösteriliyor" : "Veri alınamadı";
      live.dataset.state = cachedCountry?.hotspots?.length ? "live" : "error";
      setSourceStatus(`Veriler alınamadı: ${error.message}`, true);
      setWeatherStatus("Hava özeti alınamadı.", true);
      setPeriodSummary(dashboard);
      setAnalysisStatus(dashboard);
      renderAll();
    } finally {
      if (requestId === dashboardRequestId) {
        refresh.disabled = false; refresh.removeAttribute("aria-busy");
      }
    }
  }
  function dashboardDataUsable(data) {
    const status = String(data?.firms_status || "");
    return status.startsWith("NASA FIRMS") && !/verisi alınamadı|zaman aşımına uğradı|MAP_KEY.*reddedildi|TLS sertifikası doğrulanamadı/i.test(status);
  }
  function mergeDashboardPage(page, offset, previous) {
    if (!offset || !previous || previous.scope !== page.scope || previous.province_code !== page.province_code || previous.district !== page.district || previous.hours !== page.hours || previous.selected_day !== page.selected_day) return page;
    const rows = [...(previous.hotspots || []), ...(page.hotspots || [])];
    const seen = new Set();
    const hotspots = rows.filter((item) => {
      const key = `${hotspotKey(item)}|${item.acq_date || ""}|${item.acq_time || ""}|${item.satellite || ""}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    return { ...page, hotspots };
  }
  async function init() {
    populateImageryDates();
    initializeMap();
    try {
      const response = await fetch("/api/district-boundaries", { headers: { Accept: "application/geo+json, application/json" } });
      if (!response.ok) throw new Error("İl sınırları alınamadı.");
      provinceBoundaryData = await response.json();
      provinceFeatures = provinceBoundaryData.features || [];
      provinceSelect.innerHTML = `<option value="">İl seçin</option>` + provinceFeatures.map((feature) => `<option value="${escapeHtml(feature.properties.province_code)}">${escapeHtml(feature.properties.name)}</option>`).join("");
      renderBoundaryData(provinceBoundaryData);
    } catch (error) { setSourceStatus(error.message, true); }
    await loadDashboard();
    provinceSelect.addEventListener("change", () => selectProvince(provinceSelect.value));
    select.addEventListener("change", loadDashboard);
    $("back-to-provinces").addEventListener("click", showAllProvinces);
    refresh.addEventListener("click", () => nearbyLocation && !selectedProvinceCode ? loadNearbyAt(nearbyLocation.lat, nearbyLocation.lon) : loadDashboard({ force: true }));
    nearbyButton.addEventListener("click", searchNearby);
    ["ft", "fc", "fl", "fs", "q", "lf"].forEach((id) => $(id).addEventListener(id === "q" ? "input" : "change", () => {
      if (id === "ft") {
        selectedDayFilter = "";
        if (nearbyLocation && !selectedProvinceCode) loadNearbyAt(nearbyLocation.lat, nearbyLocation.lon);
        else loadDashboard();
      } else renderAll();
    }));
    document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => setView(button.dataset.v)));
    refreshTimer = window.setInterval(() => { if (!nearbyLocation) loadDashboard({ force: true }); }, 15 * 60 * 1000);
  }
  window.addEventListener("beforeunload", () => { if (refreshTimer) clearInterval(refreshTimer); if (eumetsatRefreshTimer) clearInterval(eumetsatRefreshTimer); });
  init();
})();
