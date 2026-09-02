/**
 * app.js — AI Career Intelligence Platform
 * =========================================
 * Handles:
 *   - Tab switching (4 input modes)
 *   - Drag-and-drop & click file uploads
 *   - Async API calls to /api/analyze and /api/transcribe
 *   - Dynamic rendering of results (scores, charts, tokens, sentences)
 *   - Chart.js visualisations (doughnut + bar)
 */

"use strict";

// ── DOM references ────────────────────────────────────────────────────────
const tabBtns         = document.querySelectorAll(".tab-btn");
const tabPanels       = document.querySelectorAll(".tab-panel");
const analyseBtn      = document.getElementById("analyse-btn");
const btnLoader       = document.getElementById("btn-loader");
const btnText         = analyseBtn.querySelector(".btn-text");
const actionHint      = document.getElementById("action-hint");
const errorBox        = document.getElementById("error-box");
const errorText       = document.getElementById("error-text");
const resultsSection  = document.getElementById("results-section");

// Input elements
const manualText      = document.getElementById("manual-text");
const charCount       = document.getElementById("char-count");
const clearTextBtn    = document.getElementById("clear-text-btn");
const txtFileInput    = document.getElementById("txt-file-input");
const csvFileInput    = document.getElementById("csv-file-input");
const audioFileInput  = document.getElementById("audio-file-input");
const txtFileName     = document.getElementById("txt-file-name");
const csvFileName     = document.getElementById("csv-file-name");
const audioFileName   = document.getElementById("audio-file-name");

// Result elements
const transcriptCard  = document.getElementById("transcript-card");
const transcriptText  = document.getElementById("transcript-text");
const scoreLabel      = document.getElementById("score-label");
const scoreCompound   = document.getElementById("score-compound");
const scorePos        = document.getElementById("score-pos");
const scoreNeg        = document.getElementById("score-neg");
const scoreNeu        = document.getElementById("score-neu");
const statsGrid       = document.getElementById("stats-grid");
const pipelineSteps   = document.getElementById("pipeline-steps");
const emotionTags     = document.getElementById("emotion-tags");
const sentenceList    = document.getElementById("sentence-list");
const reportMeta      = document.getElementById("report-meta");

// Chart instances (kept for destruction on re-render)
let doughnutChart = null;
let barChart      = null;

// Current active tab
let activeTab = "text";


// ══════════════════════════════════════════════════════════════════════════
//  TAB SWITCHING
// ══════════════════════════════════════════════════════════════════════════

tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;
    switchTab(target);
  });
});

function switchTab(tab) {
  activeTab = tab;

  tabBtns.forEach(b => {
    const isActive = b.dataset.tab === tab;
    b.classList.toggle("active", isActive);
    b.setAttribute("aria-selected", String(isActive));
  });

  tabPanels.forEach(p => {
    const isActive = p.id === `panel-${tab}`;
    p.classList.toggle("active", isActive);
  });

  hideError();
}


// ══════════════════════════════════════════════════════════════════════════
//  CHARACTER COUNTER
// ══════════════════════════════════════════════════════════════════════════

manualText.addEventListener("input", () => {
  const len = manualText.value.length;
  charCount.textContent = `${len.toLocaleString()} character${len !== 1 ? "s" : ""}`;
});

clearTextBtn.addEventListener("click", () => {
  manualText.value  = "";
  charCount.textContent = "0 characters";
  manualText.focus();
});


// ══════════════════════════════════════════════════════════════════════════
//  FILE UPLOAD — display filename on selection
// ══════════════════════════════════════════════════════════════════════════

txtFileInput.addEventListener("change",   () => showFileName(txtFileInput,   txtFileName));
csvFileInput.addEventListener("change",   () => showFileName(csvFileInput,   csvFileName));
audioFileInput.addEventListener("change", () => showFileName(audioFileInput, audioFileName));

function showFileName(input, display) {
  if (input.files && input.files[0]) {
    display.textContent = `✓  ${input.files[0].name}`;
  }
}

// Drag-and-drop for upload zones
setupDropZone("txt-drop-zone",   txtFileInput,   txtFileName);
setupDropZone("csv-drop-zone",   csvFileInput,   csvFileName);
setupDropZone("audio-drop-zone", audioFileInput, audioFileName);

function setupDropZone(zoneId, fileInput, displayEl) {
  const zone = document.getElementById(zoneId);
  if (!zone) return;

  zone.addEventListener("dragover", e => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });

  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));

  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) {
      // Assign to hidden input via DataTransfer
      const dt  = new DataTransfer();
      dt.items.add(file);
      fileInput.files   = dt.files;
      displayEl.textContent = `✓  ${file.name}`;
    }
  });

  // Click anywhere on zone triggers file picker
  zone.addEventListener("click", e => {
    if (e.target !== fileInput) fileInput.click();
  });

  zone.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
}


// ══════════════════════════════════════════════════════════════════════════
//  ANALYSE BUTTON
// ══════════════════════════════════════════════════════════════════════════

analyseBtn.addEventListener("click", async () => {
  hideError();
  setLoading(true);
  resultsSection.classList.add("hidden");

  try {
    if (activeTab === "audio") {
      await runTranscribeAndAnalyze();
    } else {
      await runAnalyze();
    }
  } catch (err) {
    showError("Unexpected error: " + err.message);
  } finally {
    setLoading(false);
  }
});


// ── Text / TXT / CSV ────────────────────────────────────────────────────

async function runAnalyze() {
  const formData = new FormData();

  if (activeTab === "text") {
    const text = manualText.value.trim();
    if (!text) { showError("Please enter some text to analyse."); return; }
    formData.append("text", text);

  } else if (activeTab === "txt") {
    if (!txtFileInput.files || !txtFileInput.files[0]) {
      showError("Please select a .txt file."); return;
    }
    formData.append("txt_file", txtFileInput.files[0]);

  } else if (activeTab === "csv") {
    if (!csvFileInput.files || !csvFileInput.files[0]) {
      showError("Please select a .csv file."); return;
    }
    formData.append("csv_file", csvFileInput.files[0]);
  }

  actionHint.textContent = "Running pipeline…";
  const res  = await fetch("/api/analyze", { method: "POST", body: formData });
  const data = await res.json();
  actionHint.textContent = "";

  if (!res.ok || data.status === "error") {
    showError(data.message || "Analysis failed."); return;
  }

  // Hide transcript (not applicable here)
  transcriptCard.classList.add("hidden");
  renderResults(data, null);
}


// ── Audio / Video ────────────────────────────────────────────────────────

async function runTranscribeAndAnalyze() {
  if (!audioFileInput.files || !audioFileInput.files[0]) {
    showError("Please select an audio or video file."); return;
  }

  const formData = new FormData();
  formData.append("media_file", audioFileInput.files[0]);
  formData.append("analyze", "true");

  actionHint.textContent = "Transcribing…  (this may take a moment)";
  const res  = await fetch("/api/transcribe", { method: "POST", body: formData });
  const data = await res.json();
  actionHint.textContent = "";

  if (!res.ok || data.status === "error") {
    showError(data.message || "Transcription failed."); return;
  }

  // Show transcript
  transcriptCard.classList.remove("hidden");
  transcriptText.textContent = data.transcript || "(empty transcript)";

  if (data.pipeline && data.pipeline.report) {
    renderResults(data.pipeline, data.transcript);
  } else {
    showError("Transcript received but pipeline did not return results.");
  }
}


// ══════════════════════════════════════════════════════════════════════════
//  RENDER RESULTS
// ══════════════════════════════════════════════════════════════════════════

function renderResults(data, transcript) {
  const sd = data.sentiment_detail || data.report?.sentiment_summary || {};
  const pd = data.preprocessing_detail || {};
  const rp = data.report || {};

  // ── Sentiment score cards ───────────────────────────────────────────
  const label    = sd.label    || rp.sentiment_summary?.overall_label || "Neutral";
  const compound = sd.compound ?? rp.sentiment_summary?.compound_score ?? 0;
  const pos      = sd.pos      ?? rp.sentiment_summary?.positive_score ?? 0;
  const neg      = sd.neg      ?? rp.sentiment_summary?.negative_score ?? 0;
  const neu      = sd.neu      ?? rp.sentiment_summary?.neutral_score  ?? 0;

  scoreLabel.textContent    = label;
  scoreLabel.className      = `score-card-value ${label}`;
  scoreCompound.textContent = compound.toFixed(4);
  scorePos.textContent      = pos.toFixed(4);
  scoreNeg.textContent      = neg.toFixed(4);
  scoreNeu.textContent      = neu.toFixed(4);

  // ── Doughnut chart ──────────────────────────────────────────────────
  renderDoughnut(pos, neg, neu);

  // ── Top words bar chart ─────────────────────────────────────────────
  const topWords = rp.preprocessing_summary?.top_words || [];
  renderBarChart(topWords);

  // ── Preprocessing stats ─────────────────────────────────────────────
  const ps = rp.preprocessing_summary || {};
  renderStats([
    { val: ps.sentence_count   ?? "—", lbl: "Sentences"    },
    { val: ps.word_count_raw   ?? "—", lbl: "Raw Words"    },
    { val: ps.word_count_clean ?? "—", lbl: "Clean Words"  },
    { val: ps.unique_words     ?? "—", lbl: "Unique Words"  },
  ]);

  // ── Pipeline token steps ────────────────────────────────────────────
  if (data.preprocessing_detail) {
    renderPipelineSteps(data.preprocessing_detail);
  } else {
    pipelineSteps.innerHTML = "";
  }

  // ── Emotion tags ────────────────────────────────────────────────────
  const tags = rp.emotion_tags || [];
  emotionTags.innerHTML = tags.map(t =>
    `<span class="emotion-tag" role="listitem">${escHtml(t)}</span>`
  ).join("");

  // ── Per-sentence breakdown ──────────────────────────────────────────
  const sentences = sd.per_sentence || rp.sentiment_summary?.per_sentence || [];
  renderSentences(sentences);

  // ── Report meta ─────────────────────────────────────────────────────
  renderReportMeta(rp);

  // ── Show results ────────────────────────────────────────────────────
  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}


// ══════════════════════════════════════════════════════════════════════════
//  CHART RENDERING
// ══════════════════════════════════════════════════════════════════════════

function renderDoughnut(pos, neg, neu) {
  if (doughnutChart) doughnutChart.destroy();
  const ctx = document.getElementById("sentiment-doughnut").getContext("2d");

  doughnutChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels:   ["Positive", "Negative", "Neutral"],
      datasets: [{
        data:            [pos, neg, neu],
        backgroundColor: ["rgba(74,222,128,0.8)", "rgba(248,113,113,0.8)", "rgba(148,163,184,0.4)"],
        borderColor:     ["#4ade80", "#f87171", "#64748b"],
        borderWidth:     2,
        hoverOffset:     6,
      }],
    },
    options: {
      responsive:         true,
      maintainAspectRatio:false,
      cutout:             "68%",
      plugins: {
        legend: {
          position: "bottom",
          labels:   { color: "#94a3b8", font: { size: 12, family: "Inter" }, padding: 16 },
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${(ctx.parsed * 100).toFixed(1)}%`,
          },
        },
      },
    },
  });
}

function renderBarChart(topWords) {
  if (barChart) barChart.destroy();
  if (!topWords || topWords.length === 0) return;

  const ctx    = document.getElementById("top-words-bar").getContext("2d");
  const labels = topWords.map(w => w.word  || w[0]);
  const counts = topWords.map(w => w.count || w[1]);

  barChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label:           "Frequency",
        data:            counts,
        backgroundColor: "rgba(99,102,241,0.7)",
        borderColor:     "#6366f1",
        borderWidth:     1,
        borderRadius:    6,
        hoverBackgroundColor: "rgba(139,92,246,0.9)",
      }],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      indexAxis:           "y",
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `  Count: ${ctx.parsed.x}`,
          },
        },
      },
      scales: {
        x: {
          grid:   { color: "rgba(255,255,255,0.05)" },
          ticks:  { color: "#64748b", font: { size: 11 } },
        },
        y: {
          grid:   { display: false },
          ticks:  { color: "#94a3b8", font: { size: 12, family: "JetBrains Mono" } },
        },
      },
    },
  });
}


// ══════════════════════════════════════════════════════════════════════════
//  DOM RENDERING HELPERS
// ══════════════════════════════════════════════════════════════════════════

function renderStats(items) {
  statsGrid.innerHTML = items.map(({ val, lbl }) => `
    <div class="stat-item">
      <span class="stat-val">${val}</span>
      <span class="stat-lbl">${escHtml(lbl)}</span>
    </div>
  `).join("");
}

function renderPipelineSteps(pd) {
  const steps = [
    { label: "After Tokenization",      tokens: pd.tokens          || [] },
    { label: "After Stop-word Removal", tokens: pd.filtered_tokens || [] },
    { label: "After Lemmatization",     tokens: pd.lemmas          || [] },
  ];

  pipelineSteps.innerHTML = steps.map(({ label, tokens }) => {
    const chips = tokens.slice(0, 30).map(t =>
      `<span class="token-chip">${escHtml(t)}</span>`
    ).join("");
    const more  = tokens.length > 30 ? `<span class="token-chip">+${tokens.length - 30} more</span>` : "";
    return `
      <div class="pipeline-step">
        <div class="step-label">${label}</div>
        <div class="step-tokens">${chips}${more}</div>
      </div>
    `;
  }).join("");
}

function renderSentences(sentences) {
  if (!sentences || sentences.length === 0) {
    sentenceList.innerHTML = `<p style="color:var(--text-muted);font-size:0.88rem;">No sentence-level data available.</p>`;
    return;
  }

  sentenceList.innerHTML = sentences.map(s => `
    <div class="sentence-item">
      <span class="sentence-text">${escHtml(s.sentence)}</span>
      <span class="sentence-badge">
        <span class="s-label ${escHtml(s.label)}">${escHtml(s.label)}</span>
        <span class="s-score">${s.compound >= 0 ? "+" : ""}${(s.compound).toFixed(3)}</span>
      </span>
    </div>
  `).join("");
}

function renderReportMeta(rp) {
  const is    = rp.input_summary         || {};
  const items = [
    { key: "Report Generated",   val: rp.generated_at || "—" },
    { key: "Input Source",       val: is.source        || "—" },
    { key: "Characters",         val: is.char_count    ?? "—" },
    { key: "Rows / Lines",       val: is.row_count     ?? "—" },
    { key: "Milestone",          val: rp.milestone     || "—" },
  ];

  reportMeta.innerHTML = items.map(({ key, val }) => `
    <div class="meta-item">
      <span class="meta-key">${escHtml(key)}</span>
      <span class="meta-val">${escHtml(String(val))}</span>
    </div>
  `).join("");
}


// ══════════════════════════════════════════════════════════════════════════
//  UI STATE HELPERS
// ══════════════════════════════════════════════════════════════════════════

function setLoading(on) {
  analyseBtn.disabled     = on;
  btnLoader.classList.toggle("active", on);
  btnText.textContent     = on ? "Analysing…" : "Analyse";
}

function showError(msg) {
  errorText.textContent = msg;
  errorBox.classList.remove("hidden");
}

function hideError() {
  errorBox.classList.add("hidden");
  errorText.textContent = "";
}

function escHtml(str) {
  const div       = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
