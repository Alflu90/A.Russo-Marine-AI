// ---------- Classificatore ----------

const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const browseBtn = document.getElementById("browse-btn");
const previewImg = document.getElementById("preview-img");
const uploadHint = document.getElementById("upload-hint");
const classifyBtn = document.getElementById("classify-btn");
const resultBox = document.getElementById("classify-result");
const predictionsList = document.getElementById("predictions-list");
const modelAccuracyNote = document.getElementById("model-accuracy-note");
const errorBox = document.getElementById("classify-error");
const resetBtn = document.getElementById("reset-btn");

let selectedFile = null;

resetBtn.addEventListener("click", () => {
  selectedFile = null;
  previewImg.hidden = true;
  previewImg.src = "";
  uploadHint.hidden = false;
  classifyBtn.disabled = true;
  resultBox.hidden = true;
  errorBox.hidden = true;
  fileInput.value = "";
});

browseBtn.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("click", (e) => {
  if (e.target === browseBtn) return;
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.hidden = false;
  uploadHint.hidden = true;
  classifyBtn.disabled = false;
  resultBox.hidden = true;
  errorBox.hidden = true;
}

classifyBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  classifyBtn.disabled = true;
  classifyBtn.textContent = "Classificazione in corso…";
  errorBox.hidden = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const resp = await fetch("/api/classify", { method: "POST", body: formData });
    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.detail || "Errore durante la classificazione.");
    }

    renderPredictions(data.predictions, data.model_val_accuracy);
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.hidden = false;
  } finally {
    classifyBtn.disabled = false;
    classifyBtn.textContent = "Classifica";
  }
});

function renderPredictions(predictions, modelValAccuracy) {
  predictionsList.innerHTML = "";
  predictions.forEach((p) => {
    const li = document.createElement("li");
    li.className = "prediction-row";
    const pct = Math.round(p.confidence * 100);
    li.innerHTML = `
      <span class="prediction-label">${p.label}</span>
      <span class="prediction-bar-track"><span class="prediction-bar-fill" style="width:${pct}%"></span></span>
      <span class="prediction-confidence">${pct}%</span>
    `;
    predictionsList.appendChild(li);
  });

  if (modelValAccuracy != null) {
    modelAccuracyNote.textContent = `Accuratezza del modello sul set di validazione: ${Math.round(modelValAccuracy * 100)}%.`;
  } else {
    modelAccuracyNote.textContent = "";
  }

  resultBox.hidden = false;
}

// ---------- Analisi ambientale ----------

const speciesSelect = document.getElementById("species-select");
const summaryGrid = document.getElementById("species-summary");
const statOccurrences = document.getElementById("stat-occurrences");
const statAphia = document.getElementById("stat-aphia");
const statReliable = document.getElementById("stat-reliable");

let trendChart = null;

// Mappa di distribuzione: inizializzata vuota, centrata sull'oceano globale.
const map = L.map("map", { worldCopyJump: true }).setView([15, 0], 2);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
  maxZoom: 10,
}).addTo(map);
let occurrenceLayer = L.layerGroup().addTo(map);

async function loadOccurrenceMap(label) {
  occurrenceLayer.clearLayers();
  try {
    const resp = await fetch(`/api/environmental/occurrences/${encodeURIComponent(label)}`);
    if (!resp.ok) return; // specie senza occorrenze geolocalizzate: mappa resta vuota, nessun errore bloccante

    const data = await resp.json();
    const markers = data.points.map((p) =>
      L.circleMarker([p.lat, p.lon], {
        radius: 3,
        color: "#F4A261",
        fillColor: "#F4A261",
        fillOpacity: 0.6,
        weight: 0,
      })
    );
    markers.forEach((m) => occurrenceLayer.addLayer(m));

    if (markers.length > 0) {
      map.fitBounds(L.featureGroup(markers).getBounds(), { maxZoom: 6, padding: [20, 20] });
    }
  } catch {
    // rete non disponibile o simile: la mappa resta semplicemente vuota
  }
}

async function loadSpeciesList() {
  try {
    const resp = await fetch("/api/environmental/species");
    const data = await resp.json();
    speciesSelect.innerHTML = '<option value="">Scegli una specie…</option>';
    data.species.forEach((label) => {
      const opt = document.createElement("option");
      opt.value = label;
      opt.textContent = label;
      speciesSelect.appendChild(opt);
    });
  } catch {
    speciesSelect.innerHTML = '<option value="">Impossibile caricare l\'elenco specie</option>';
  }
}

speciesSelect.addEventListener("change", async () => {
  const label = speciesSelect.value;
  if (!label) return;

  loadOccurrenceMap(label);

  const [trendResp, summaryResp] = await Promise.all([
    fetch(`/api/environmental/trend/${encodeURIComponent(label)}`),
    fetch(`/api/environmental/summary/${encodeURIComponent(label)}`),
  ]);

  if (trendResp.ok) {
    const trend = await trendResp.json();
    renderTrendChart(trend);
  }

  if (summaryResp.ok) {
    const summary = await summaryResp.json();
    statOccurrences.textContent = summary.obis_occurrence_count ?? "n/d";
    statAphia.textContent = summary.aphia_id ?? "n/d";
    statReliable.textContent = summary.taxonomy_reliable ? "Sì" : "No";
    summaryGrid.hidden = false;
  }
});

function renderTrendChart(trend) {
  const ctx = document.getElementById("trend-chart").getContext("2d");
  const pointColors = trend.years.map((y) =>
    trend.anomaly_years.includes(y) ? "#F4A261" : "#1C7293"
  );
  const pointRadii = trend.years.map((y) => (trend.anomaly_years.includes(y) ? 6 : 3));

  if (trendChart) trendChart.destroy();

  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: trend.years,
      datasets: [{
        label: `Osservazioni per anno — ${trend.canonical_label}`,
        data: trend.n_occurrences,
        borderColor: "#065A82",
        backgroundColor: "rgba(6,90,130,0.08)",
        pointBackgroundColor: pointColors,
        pointRadius: pointRadii,
        tension: 0.15,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, title: { display: true, text: "Osservazioni (campione OBIS)" } },
      },
    },
  });
}

loadSpeciesList();

// ---------- Assistente (chat in streaming) ----------

const assistantForm = document.getElementById("assistant-form");
const assistantInput = document.getElementById("assistant-input");
const assistantStatus = document.getElementById("assistant-status");
const assistantOutput = document.getElementById("assistant-output");

let currentEventSource = null;

assistantForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = assistantInput.value.trim();
  if (!query) return;

  if (currentEventSource) currentEventSource.close();

  assistantOutput.textContent = "";
  assistantStatus.hidden = false;
  assistantStatus.textContent = "Invio della domanda…";

  const url = `/api/assistant/ask?query=${encodeURIComponent(query)}`;
  const source = new EventSource(url);
  currentEventSource = source;

  source.addEventListener("status", (e) => {
    assistantStatus.textContent = e.data;
  });

  source.addEventListener("token", (e) => {
    // EventSource ricompone automaticamente le righe "data:" multiple con \n,
    // ma i delta dei token sono in genere brevi frammenti senza a-capo interni.
    assistantOutput.textContent += e.data;
  });

  source.addEventListener("llm_error", (e) => {
    assistantOutput.textContent = e.data;
    assistantOutput.classList.add("assistant-output-error");
  });

  source.addEventListener("done", () => {
    assistantStatus.hidden = true;
    source.close();
  });

  source.onerror = () => {
    assistantStatus.textContent = "Connessione con l'assistente interrotta. Verifica che LM Studio e Chroma siano attivi.";
    source.close();
  };
});
