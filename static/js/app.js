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


}


// ══════════════════════════════════════════════════════════════════════════
//  SECTION NAVIGATION & MEETING INTELLIGENCE (MILESTONE 2)
// ══════════════════════════════════════════════════════════════════════════

const navBtns = document.querySelectorAll(".nav-btn");
const sectionContainers = document.querySelectorAll(".section-container");

navBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    const targetSectionId = btn.dataset.section;

    navBtns.forEach(b => b.classList.toggle("active", b === btn));
    sectionContainers.forEach(sec => {
      sec.classList.toggle("active", sec.id === targetSectionId);
    });
  });
});

// Meeting Input Tabs
const mtabBtns = document.querySelectorAll("[data-mtab]");
let activeMTab = "maudio";

mtabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    activeMTab = btn.dataset.mtab;
    mtabBtns.forEach(b => b.classList.toggle("active", b === btn));
    document.querySelectorAll("#section-meeting .tab-panel").forEach(p => {
      p.classList.toggle("active", p.id === `mpanel-${activeMTab}`);
    });
    hideMeetingError();
  });
});

// File input & Drag drop setup for Meeting Recording
const meetingFileInput = document.getElementById("meeting-file-input");
const meetingFileName = document.getElementById("meeting-file-name");
if (meetingFileInput) {
  meetingFileInput.addEventListener("change", () => {
    if (meetingFileInput.files && meetingFileInput.files[0]) {
      meetingFileName.textContent = `✓  ${meetingFileInput.files[0].name}`;
    }
  });
  setupDropZone("meeting-drop-zone", meetingFileInput, meetingFileName);
}

// History Select
const historySelect = document.getElementById("history-select");
if (historySelect) {
  loadMeetingHistory();
  historySelect.addEventListener("change", (e) => {
    const selectedId = e.target.value;
    if (selectedId) {
      fetchMeetingById(selectedId);
    }
  });
}

async function loadMeetingHistory() {
  try {
    const res = await fetch("/api/meetings");
    if (!res.ok) return;
    const data = await res.json();
    if (data.status === "ok" && Array.isArray(data.meetings)) {
      historySelect.innerHTML = `<option value="">-- Load Past Meeting --</option>` +
        data.meetings.map(m => `<option value="${escHtml(m.id)}">${escHtml(m.title)} (${escHtml(m.created_at.slice(0,10))})</option>`).join("");
    }
  } catch (err) {
    console.warn("Could not fetch meeting history", err);
  }
}

async function fetchMeetingById(meetingId) {
  try {
    setMeetingLoading(true);
    hideMeetingError();
    const res = await fetch(`/api/meetings/${meetingId}`);
    const data = await res.json();
    if (res.ok && data.status === "ok") {
      renderMeetingDashboard(data);
    } else {
      showMeetingError(data.message || "Failed to retrieve meeting details.");
    }
  } catch (err) {
    showMeetingError(`Network error: ${err.message}`);
  } finally {
    setMeetingLoading(false);
  }
}

// Process Meeting Button Handler
const processMeetingBtn = document.getElementById("process-meeting-btn");
const meetingBtnLoader = document.getElementById("meeting-btn-loader");
const meetingErrorBox = document.getElementById("meeting-error-box");
const meetingErrorText = document.getElementById("meeting-error-text");
const meetingProgressCard = document.getElementById("meeting-progress-card");
const meetingDashboard = document.getElementById("meeting-dashboard");

if (processMeetingBtn) {
  processMeetingBtn.addEventListener("click", handleProcessMeeting);
}

async function handleProcessMeeting() {
  hideMeetingError();

  const titleInput = document.getElementById("meeting-title-input").value.trim();
  const transcriptInput = document.getElementById("meeting-transcript-text").value.trim();
  const title = titleInput || "Meeting Recording";

  const formData = new FormData();
  formData.append("title", title);

  if (activeMTab === "maudio") {
    if (!meetingFileInput.files || !meetingFileInput.files[0]) {
      showMeetingError("Please select or drop an audio/video recording file.");
      return;
    }
    formData.append("media_file", meetingFileInput.files[0]);
  } else {
    if (!transcriptInput) {
      showMeetingError("Please paste the raw meeting transcript text.");
      return;
    }
    formData.append("transcript", transcriptInput);
  }

  setMeetingLoading(true);
  showStepperProgress(true);

  try {
    updateStep("step-transcribe", "active");
    await new Promise(r => setTimeout(r, 400));
    updateStep("step-transcribe", "completed");

    updateStep("step-llm", "active");
    
    const res = await fetch("/api/meetings/process", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok || data.status !== "ok") {
      showMeetingError(data.message || "Meeting processing failed.");
      showStepperProgress(false);
      return;
    }

    updateStep("step-llm", "completed");
    updateStep("step-validation", "active");
    await new Promise(r => setTimeout(r, 300));
    updateStep("step-validation", "completed");

    updateStep("step-db", "active");
    await new Promise(r => setTimeout(r, 300));
    updateStep("step-db", "completed");

    renderMeetingDashboard(data);
    loadMeetingHistory(); // Refresh history dropdown

  } catch (err) {
    showMeetingError(`Processing failed: ${err.message}`);
    showStepperProgress(false);
  } finally {
    setMeetingLoading(false);
  }
}

function renderMeetingDashboard(data) {
  document.getElementById("dash-title").textContent = data.title || "Meeting Intelligence Report";
  document.getElementById("dash-id").textContent = `ID: ${data.meeting_id}`;
  document.getElementById("dash-summary").textContent = data.summary || "No executive summary available.";

  // Render Key Points
  const kpUl = document.getElementById("dash-key-points");
  if (data.key_points && data.key_points.length > 0) {
    kpUl.innerHTML = data.key_points.map(kp => `<li>${escHtml(kp)}</li>`).join("");
  } else {
    kpUl.innerHTML = `<li style="color:var(--text-muted)">No key points extracted.</li>`;
  }

  // Render Decisions
  const decUl = document.getElementById("dash-decisions");
  if (data.decisions && data.decisions.length > 0) {
    decUl.innerHTML = data.decisions.map(d => `<li>${escHtml(d)}</li>`).join("");
  } else {
    decUl.innerHTML = `<li style="color:var(--text-muted)">No formal decisions recorded.</li>`;
  }

  // Render Action Items
  const tbody = document.getElementById("dash-action-tbody");
  if (data.action_items && data.action_items.length > 0) {
    tbody.innerHTML = data.action_items.map(item => `
      <tr>
        <td><strong>${escHtml(item.task)}</strong></td>
        <td>${item.assigned_to ? escHtml(item.assigned_to) : '<span style="color:var(--text-muted)">Unassigned</span>'}</td>
        <td>${item.deadline ? escHtml(item.deadline) : '<span style="color:var(--text-muted)">—</span>'}</td>
        <td><span class="badge-priority badge-priority-${escHtml(item.priority || 'Unknown')}">${escHtml(item.priority || 'Unknown')}</span></td>
        <td><span class="badge-status badge-status-${escHtml((item.status || 'Pending').replace(/\s+/g, '-'))}">${escHtml(item.status || 'Pending')}</span></td>
      </tr>
    `).join("");
  } else {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--text-muted)">No action items extracted.</td></tr>`;
  }

  // Render Participants & Responsibilities
  const pGrid = document.getElementById("dash-participants-grid");
  if (data.participants && data.participants.length > 0) {
    pGrid.innerHTML = data.participants.map(p => {
      const initial = p.name ? p.name.charAt(0).toUpperCase() : "?";
      const resps = p.responsibilities && p.responsibilities.length > 0
        ? p.responsibilities.map(r => `<li>${escHtml(r)}</li>`).join("")
        : `<li style="color:var(--text-muted)">No assigned responsibilities</li>`;

      return `
        <div class="participant-card-item">
          <div class="participant-header">
            <div class="avatar-circle">${initial}</div>
            <div class="participant-name-title">${escHtml(p.name)}</div>
          </div>
          <div class="responsibility-subheading">Assigned Responsibilities:</div>
          <ul class="bullet-list">${resps}</ul>
        </div>
      `;
    }).join("");
  } else {
    pGrid.innerHTML = `<p style="color:var(--text-muted);font-size:0.88rem;">No participants mapped.</p>`;
  }

  // Render Raw Transcript
  document.getElementById("dash-raw-transcript").textContent = data.raw_transcript || "Transcript unavailable.";

  // Reveal Dashboard
  meetingDashboard.classList.remove("hidden");
  meetingDashboard.scrollIntoView({ behavior: "smooth" });
}

function setMeetingLoading(on) {
  processMeetingBtn.disabled = on;
  meetingBtnLoader.classList.toggle("active", on);
  processMeetingBtn.querySelector(".btn-text").textContent = on ? "Processing…" : "Process Meeting";
}

function showMeetingError(msg) {
  meetingErrorText.textContent = msg;
  meetingErrorBox.classList.remove("hidden");
}

function hideMeetingError() {
  meetingErrorBox.classList.add("hidden");
  meetingErrorText.textContent = "";
}

function showStepperProgress(show) {
  meetingProgressCard.classList.toggle("hidden", !show);
  if (show) {
    ["step-transcribe", "step-llm", "step-validation", "step-db"].forEach(id => {
      const el = document.getElementById(id);
      el.classList.remove("active", "completed");
    });
  }
}

function updateStep(stepId, state) {
  const el = document.getElementById(stepId);
  if (!el) return;
  if (state === "active") {
    el.classList.remove("completed");
    el.classList.add("active");
  } else if (state === "completed") {
    el.classList.remove("active");
    el.classList.add("completed");
  }
}

// DOM Rendering Helpers & State
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


// ══════════════════════════════════════════════════════════════════════════
//  MAIN SECTION NAVIGATION (Meeting Intelligence / Semantic Search / NLP)
// ══════════════════════════════════════════════════════════════════════════
document.querySelectorAll(".main-nav .nav-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const targetSection = btn.dataset.section;
    if (!targetSection) return;

    document.querySelectorAll(".main-nav .nav-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");

    document.querySelectorAll(".section-container").forEach(sec => {
      sec.classList.toggle("active", sec.id === targetSection);
    });
  });
});


// ══════════════════════════════════════════════════════════════════════════
//  SEMANTIC SEARCH FRONTEND LOGIC (MILESTONE 3 TASK 4)
// ══════════════════════════════════════════════════════════════════════════
const semanticSearchInput  = document.getElementById("semantic-search-input");
const semanticSearchBtn    = document.getElementById("semantic-search-btn");
const searchMetaBar        = document.getElementById("search-meta-bar");
const searchMetaInfo       = document.getElementById("search-meta-info");
const searchLatencyBadge   = document.getElementById("search-latency-badge");
const searchResultsList    = document.getElementById("search-results-list");
const searchFilterType     = document.getElementById("search-filter-type");
const searchFilterTopK     = document.getElementById("search-filter-topk");

if (semanticSearchBtn) {
  semanticSearchBtn.addEventListener("click", performSemanticSearch);

  if (semanticSearchInput) {
    semanticSearchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") performSemanticSearch();
    });
  }

  document.querySelectorAll(".sample-chips-row .chip-btn").forEach(chip => {
    chip.addEventListener("click", () => {
      if (semanticSearchInput) {
        semanticSearchInput.value = chip.dataset.query;
        performSemanticSearch();
      }
    });
  });
}

async function performSemanticSearch() {
  if (!semanticSearchInput) return;
  const query = semanticSearchInput.value.trim();
  if (!query) return;

  // Set Loading state
  semanticSearchBtn.disabled = true;
  semanticSearchBtn.innerHTML = `
    <span class="spinner" aria-hidden="true"></span>
    Searching…
  `;

  searchResultsList.innerHTML = `
    <div class="card" style="text-align: center; padding: 40px; color: var(--text-muted);">
      <p>Searching meeting knowledge repository…</p>
    </div>
  `;
  searchMetaBar.classList.add("hidden");

  try {
    const contentType = searchFilterType ? searchFilterType.value : "";
    const topK = searchFilterTopK ? searchFilterTopK.value : 5;

    const res = await fetch("/api/search/semantic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        top_k: parseInt(top_k, 10),
        content_type: contentType || null
      })
    });

    if (!res.ok) {
      throw new Error(`Search request failed with status ${res.status}`);
    }

    const data = await res.json();
    if (data.status === "ok") {
      renderSemanticSearchResults(data);
    } else {
      throw new Error(data.message || "Semantic search failed.");
    }
  } catch (err) {
    console.error("Semantic search error:", err);
    searchResultsList.innerHTML = `
      <div class="card" style="border-color: rgba(248,113,113,0.3); background: rgba(248,113,113,0.05); color: var(--red-400);">
        <p>⚠️ <strong>Search Error:</strong> ${escHtml(err.message)}</p>
      </div>
    `;
  } finally {
    semanticSearchBtn.disabled = false;
    semanticSearchBtn.innerHTML = `
      <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
      Search
    `;
  }
}

function renderSemanticSearchResults(data) {
  const results = data.results || [];
  const latencyMs = data.latency_ms || 0;

  // Show Performance Latency Badge
  searchLatencyBadge.textContent = `⚡ Latency: ${latencyMs} ms`;
  searchMetaInfo.textContent = `Found ${results.length} relevant meeting result${results.length !== 1 ? "s" : ""} for "${data.query}"`;
  searchMetaBar.classList.remove("hidden");

  if (results.length === 0) {
    searchResultsList.innerHTML = `
      <div class="card" style="text-align: center; padding: 40px; color: var(--text-muted);">
        <p style="font-size: 1.1rem; margin-bottom: 8px;">🔍 No matching meeting records found.</p>
        <p style="font-size: 0.85rem;">Try rephrasing your search query or selecting a different content type filter.</p>
      </div>
    `;
    return;
  }

  searchResultsList.innerHTML = results.map(item => {
    const simPercent = Math.round((item.similarity || item.score || 0) * 100);
    const dateFormatted = item.date ? item.date.slice(0, 10) : "Recent";
    const typeLabel = (item.content_type || "transcript").replace("_", " ");

    return `
      <div class="result-card">
        <div class="result-card-header">
          <div class="result-title-wrap">
            <h3 class="result-title">${escHtml(item.title || "Meeting")}</h3>
            <span class="result-date">📅 ${escHtml(dateFormatted)}</span>
          </div>
          <div class="result-badges-wrap">
            <span class="badge-type badge-type-${escHtml(item.content_type)}">${escHtml(typeLabel)}</span>
            <span class="badge-similarity">${simPercent}% Match</span>
          </div>
        </div>

        <div class="result-snippet-box">
          "${escHtml(item.relevant_snippet || item.text || "")}"
        </div>

        <div class="result-card-actions">
          <span class="result-meeting-id">Meeting ID: ${escHtml(item.meeting_id)}</span>
          <button class="btn-view-meeting" onclick="viewMeetingFromSearch('${escHtml(item.meeting_id)}')">
            View Complete Meeting →
          </button>
        </div>
      </div>
    `;
  }).join("");
}

function viewMeetingFromSearch(meetingId) {
  // Switch to Meeting Intelligence section
  const meetingNavBtn = document.getElementById("nav-meeting");
  if (meetingNavBtn) meetingNavBtn.click();

  // Load meeting data
  if (typeof fetchMeetingById === "function") {
    fetchMeetingById(meetingId);
  }
}


// ══════════════════════════════════════════════════════════════════════════
//  SEARCH MODE TABS (Semantic Search vs Ask Your Meetings RAG)
// ══════════════════════════════════════════════════════════════════════════
document.querySelectorAll(".search-mode-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    const smode = tab.dataset.smode;
    document.querySelectorAll(".search-mode-tab").forEach(t => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".smode-panel").forEach(p => p.classList.toggle("active", p.id === `smode-panel-${smode}`));
  });
});


// ══════════════════════════════════════════════════════════════════════════
//  RAG QUESTION ANSWERING LOGIC (MILESTONE 3 TASK 5)
// ══════════════════════════════════════════════════════════════════════════
const ragQuestionInput   = document.getElementById("rag-question-input");
const ragAskBtn          = document.getElementById("rag-ask-btn");
const ragOutputContainer = document.getElementById("rag-output-container");
const ragAnswerText      = document.getElementById("rag-answer-text");
const ragLatencyBadge    = document.getElementById("rag-latency-badge");
const ragSourcesList     = document.getElementById("rag-sources-list");

if (ragAskBtn) {
  ragAskBtn.addEventListener("click", performRAGQuestionAnswering);

  if (ragQuestionInput) {
    ragQuestionInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") performRAGQuestionAnswering();
    });
  }

  document.querySelectorAll(".rag-chip-btn").forEach(chip => {
    chip.addEventListener("click", () => {
      if (ragQuestionInput) {
        ragQuestionInput.value = chip.dataset.question;
        performRAGQuestionAnswering();
      }
    });
  });
}

async function performRAGQuestionAnswering() {
  if (!ragQuestionInput) return;
  const question = ragQuestionInput.value.trim();
  if (!question) return;

  ragAskBtn.disabled = true;
  ragAskBtn.innerHTML = `
    <span class="spinner" aria-hidden="true"></span>
    Thinking…
  `;

  ragOutputContainer.classList.remove("hidden");
  ragAnswerText.textContent = "Retrieving meeting evidence and generating grounded response…";
  ragLatencyBadge.textContent = "⚡ Processing…";
  ragSourcesList.innerHTML = "";

  try {
    const res = await fetch("/api/search/rag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, top_k: 5 })
    });

    if (!res.ok) {
      throw new Error(`RAG Q&A failed with status ${res.status}`);
    }

    const data = await res.json();
    if (data.status === "ok") {
      renderRAGResponse(data);
    } else {
      throw new Error(data.message || "RAG Q&A failed.");
    }
  } catch (err) {
    console.error("RAG Error:", err);
    ragAnswerText.textContent = `⚠️ Error: ${err.message}`;
    ragLatencyBadge.textContent = "⚡ Error";
    ragSourcesList.innerHTML = "";
  } finally {
    ragAskBtn.disabled = false;
    ragAskBtn.innerHTML = `
      <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/></svg>
      Ask Question
    `;
  }
}

function renderRAGResponse(data) {
  const answer = data.answer || "I couldn't find enough information in the available meeting records to answer this question.";
  const sources = data.sources || [];
  const latencyMs = data.latency_ms || 0;

  ragAnswerText.textContent = answer;
  ragLatencyBadge.textContent = `⚡ Latency: ${latencyMs} ms (${sources.length} sources)`;

  if (sources.length === 0) {
    ragSourcesList.innerHTML = `
      <p style="font-size: 0.85rem; color: var(--text-muted); font-style: italic;">
        No supporting meeting context records were retrieved for this query.
      </p>
    `;
    return;
  }

  ragSourcesList.innerHTML = sources.map(src => {
    const simPercent = Math.round((src.similarity || 0) * 100);
    const dateStr = src.date ? src.date.slice(0, 10) : "Recent";
    const typeLabel = (src.content_type || "transcript").replace("_", " ");

    return `
      <div class="rag-source-item">
        <div class="rag-source-header">
          <span class="rag-source-title">📌 ${escHtml(src.title || "Meeting")} (${escHtml(dateStr)})</span>
          <div class="result-badges-wrap">
            <span class="badge-type badge-type-${escHtml(src.content_type)}">${escHtml(typeLabel)}</span>
            <span class="badge-similarity">${simPercent}% Match</span>
          </div>
        </div>
        <div class="rag-source-snippet">
          "${escHtml(src.relevant_snippet || "")}"
        </div>
        <div class="result-card-actions">
          <span class="result-meeting-id">ID: ${escHtml(src.meeting_id)}</span>
          <button class="btn-view-meeting" onclick="viewMeetingFromSearch('${escHtml(src.meeting_id)}')">
            Open Meeting Details →
          </button>
        </div>
      </div>
    `;
  }).join("");
}

