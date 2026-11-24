"use strict";
// Sparrow fine-tuning UI: plain DOM over the /v1 API. Session cookie auth; every
// state-changing request carries the CSRF header the API requires.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  me: null,
  studio: { configured: false, url: "" },
  baseModel: null,
  pipelines: [],
  stockModels: [],
  workspaces: [],
  datasets: [],
  workspaceId: null,
  datasetId: null,
  dataset: null,
  items: [],
  images: [],
  assistSummary: null,
  unlabelled: [],
  snapshot: null,
  jobs: [],
  jobId: null,
  jobEvents: [],
  stream: null,
  models: [],
  modelId: null,
  renamingModel: null,
  page: "data",
};

// ------------------------------------------------------------------ errors

function showError(message) {
  $("#error-text").textContent = message;
  const banner = $("#error-banner");
  banner.hidden = false;
  // The banner is sticky, but before the page scrolls past its slot it sits at the top
  // of the content; bring it into view so an error is never off screen.
  const box = banner.getBoundingClientRect();
  if (box.top < 0 || box.bottom > window.innerHeight) banner.scrollIntoView({ block: "nearest" });
}

function clearError() {
  $("#error-banner").hidden = true;
  $("#error-text").textContent = "";
}

// Browsers report ResizeObserver loop notices (often caused by extensions resizing the
// page) as error events without an Error object; they are not application failures.
window.addEventListener("error", (e) => {
  if (!e.error && /^ResizeObserver loop/.test(e.message || "")) return;
  showError(`Unexpected error: ${e.message}`);
});
window.addEventListener("unhandledrejection", (e) => {
  const r = e.reason;
  showError(r && r.message ? r.message : String(r));
});

let toastTimer = null;
function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3500);
}

// Modal choice. options: [{ value, label, detail }]. Resolves to the chosen value, or
// null when the user cancels (button, Esc or closing the dialog).
function chooseDialog(title, message, options, cancelLabel = "Cancel") {
  const dlg = $("#choice-dialog");
  $("#choice-title").textContent = title;
  $("#choice-message").textContent = message;
  $("#choice-cancel").textContent = cancelLabel;
  return new Promise((resolve) => {
    let picked = null;
    $("#choice-options").replaceChildren(...options.map((o) => {
      const btn = el("button", { class: "choice-option", type: "button" },
        el("strong", { text: o.label }), o.detail ? el("span", { text: o.detail }) : null);
      btn.addEventListener("click", () => { picked = o.value; dlg.close(); });
      return btn;
    }));
    $("#choice-cancel").onclick = () => dlg.close();
    dlg.addEventListener("close", () => resolve(picked), { once: true });
    dlg.showModal();
  });
}

// ------------------------------------------------------------------ api

class ApiError extends Error {
  constructor(status, message, detail) {
    super(message);
    this.status = status;
    this.detail = detail;  // the response's parsed ``detail`` (string or object), if any
  }
}

function detailText(detail) {
  if (detail == null) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => (typeof d === "string" ? d : d.msg || JSON.stringify(d))).join("; ");
  }
  return JSON.stringify(detail);
}

async function api(method, path, body, opts = {}) {
  const headers = {};
  if (method !== "GET" && method !== "HEAD") headers["X-Sparrow-CSRF"] = "1";
  let payload;
  if (body instanceof FormData) {
    payload = body; // the browser sets the multipart Content-Type with its boundary
  } else if (body instanceof Blob || body instanceof ArrayBuffer) {
    payload = body;
    headers["Content-Type"] = opts.contentType || "application/octet-stream";
  } else if (body !== undefined) {
    payload = JSON.stringify(body);
    headers["Content-Type"] = "application/json";
  }
  const resp = await fetch(path, { method, headers, body: payload, credentials: "same-origin" });
  if (resp.status === 401 && !opts.allow401) {
    state.me = null;
    showLogin();
    throw new ApiError(401, "Your session has ended. Sign in again.");
  }
  if (!resp.ok) {
    let msg = `${method} ${path} failed (${resp.status})`;
    let detail;
    try {
      const data = await resp.json();
      if (data && data.detail !== undefined) {
        detail = data.detail;
        msg = `${msg}: ${detail && detail.message ? detail.message : detailText(detail)}`;
      }
    } catch (_) { /* non-JSON error body */ }
    throw new ApiError(resp.status, msg, detail);
  }
  if (resp.status === 204) return null;
  const type = resp.headers.get("content-type") || "";
  return type.includes("application/json") ? resp.json() : resp.text();
}

// Runs an async UI action; any failure lands in the banner instead of the console.
function guarded(fn) {
  return async (...args) => {
    try {
      clearError();
      await fn(...args);
    } catch (err) {
      showError(err && err.message ? err.message : String(err));
    }
  };
}

// ------------------------------------------------------------------ helpers

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("data-")) node.setAttribute(k, v);
    else if (k === "style") node.setAttribute("style", v);
    else node[k] = v;
  }
  for (const c of children) {
    if (c === null || c === undefined) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString();
}

function fmtNum(v) {
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v)) return String(v);
  return Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(4);
}

function numericMetrics(metrics) {
  const out = {};
  for (const [k, v] of Object.entries(metrics || {})) {
    if (typeof v === "number" && Number.isFinite(v)) out[k] = v;
  }
  return out;
}

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);
const STATE_BADGE = {
  queued: "status-info",
  running: "status-running",
  succeeded: "status-complete",
  failed: "status-cancelled",
  cancelled: "status-cancelled",
};

function pipelineName(id) {
  const p = state.pipelines.find((x) => x.id === id);
  return p ? p.name : id;
}

// ------------------------------------------------------------------ navigation and auth

const WORKSPACE_FOLD_KEY = "sparrow-ft.workspace-folded";
const WORKSPACE_KEY = "sparrow-ft.workspace";
const DATASET_KEY = "sparrow-ft.dataset";

function remembered(key) {
  try { return Number(localStorage.getItem(key)) || null; } catch (_) { return null; }
}

function remember(key, id) {
  try {
    if (id) localStorage.setItem(key, String(id));
    else localStorage.removeItem(key);
  } catch (_) { /* storage unavailable: selection is simply not kept */ }
}

function showPage(page) {
  state.page = page;
  $$(".view").forEach((v) => { v.hidden = v.dataset.view !== page; });
  $$(".sidebar-nav .nav-item").forEach((b) => b.classList.toggle("active", b.dataset.page === page));
  if (page === "jobs") return guarded(loadJobs)();
  if (page === "models") {
    loadCatalog().catch(() => {});
    return guarded(loadModels)();
  }
  return Promise.resolve();
}

function setNavOpen(open) {
  $(".sidebar").classList.toggle("open", open);
  $("#sidebar-toggle").setAttribute("aria-expanded", String(open));
}

function showLogin() {
  closeStream();
  setNavOpen(false);
  document.body.classList.add("signed-out");
  $(".sidebar").hidden = true;
  $(".topbar").hidden = true;
  $$(".view").forEach((v) => { v.hidden = v.dataset.view !== "login"; });
  $("#login-username").focus();
}

function renderMe() {
  const me = state.me;
  document.body.classList.remove("signed-out");
  $(".sidebar").hidden = false;
  $(".topbar").hidden = false;
  $("#user-name").textContent = me.username || (me.method === "open" ? "open access" : "API token");
  $("#auth-method").textContent = me.method === "open" ? "Authentication is not enabled" : `via ${me.method}`;
  $("#logout-btn").hidden = me.method !== "session";
}

async function loadMe() {
  state.me = await api("GET", "/v1/auth/me", undefined, { allow401: true });
  if (state.me.auth_required && !state.me.method) {
    showLogin();
    return false;
  }
  renderMe();
  return true;
}

async function login(event) {
  event.preventDefault();
  const username = $("#login-username").value.trim();
  const password = $("#login-password").value;
  const resp = await fetch("/v1/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Sparrow-CSRF": "1" },
    credentials: "same-origin",
    body: JSON.stringify({ username, password }),
  });
  if (!resp.ok) {
    let msg = "Sign-in failed";
    try { msg = detailText((await resp.json()).detail) || msg; } catch (_) { /* keep default */ }
    throw new ApiError(resp.status, msg);
  }
  $("#login-password").value = "";
  await boot();
}

async function logout() {
  await api("POST", "/v1/auth/logout");
  state.me = null;
  await boot();
}

// ------------------------------------------------------------------ data page

async function loadWorkspaces(selectId) {
  const data = await api("GET", "/v1/workspaces");
  state.workspaces = data.workspaces;
  const sel = $("#workspace-select");
  sel.replaceChildren(...state.workspaces.map((w) => el("option", { value: w.id, text: w.name })));
  if (!state.workspaces.length) sel.append(el("option", { value: "", text: "(none yet)" }));
  const exists = (id) => id && state.workspaces.some((w) => w.id === Number(id));
  const first = state.workspaces[0] && state.workspaces[0].id;
  const pick = [selectId, state.workspaceId, remembered(WORKSPACE_KEY), first].find(exists);
  state.workspaceId = pick ? Number(pick) : null;
  if (state.workspaceId) sel.value = String(state.workspaceId);
  remember(WORKSPACE_KEY, state.workspaceId);
  await loadDatasets();
}

async function loadDatasets(selectId) {
  const sel = $("#dataset-select");
  if (!state.workspaceId) {
    state.datasets = [];
  } else {
    state.datasets = (await api("GET", `/v1/workspaces/${state.workspaceId}/datasets`)).datasets;
  }
  const label = (d) => (d.name.toLowerCase().includes(`(${d.task})`) ? d.name : `${d.name} (${d.task})`);
  sel.replaceChildren(...state.datasets.map((d) => el("option", { value: d.id, text: label(d) })));
  if (!state.datasets.length) sel.append(el("option", { value: "", text: "(none yet)" }));
  const exists = (id) => id && state.datasets.some((d) => d.id === Number(id));
  const first = state.datasets[0] && state.datasets[0].id;
  const pick = [selectId, state.datasetId, remembered(DATASET_KEY), first].find(exists);
  state.datasetId = pick ? Number(pick) : null;
  if (state.datasetId) sel.value = String(state.datasetId);
  if (state.datasetId) remember(DATASET_KEY, state.datasetId);
  const ws = state.workspaces.find((w) => w.id === state.workspaceId);
  $("#topbar-workspace").textContent = ws ? ws.name : "No workspace selected";
  if (state.listedWorkspaceId !== state.workspaceId) {
    state.listedWorkspaceId = state.workspaceId;
    state.jobId = null;
    state.modelId = null;
    if (state.page === "jobs" || state.page === "models") {
      showPage(state.page); // switched from the sidebar while that page is open: reload it in place
    } else {
      // Refresh the in-progress count for the new workspace; the pages reload on open.
      workspaceList("/v1/jobs?limit=200", "jobs").then((jobs) => { state.jobs = jobs; renderActiveChip(); }).catch(() => {});
    }
  }
  renderWorkspacePanel();
  await loadDataset();
}

async function loadDataset() {
  state.snapshot = null;
  if (!state.datasetId) {
    state.dataset = null;
    state.items = [];
    state.images = [];
    state.unlabelled = [];
  } else {
    state.dataset = await api("GET", `/v1/datasets/${state.datasetId}`);
    const items = (await api("GET", `/v1/datasets/${state.datasetId}/items`)).items;
    state.images = (await api("GET", `/v1/datasets/${state.datasetId}/images`)).images;
    // Uploaded images with no imported annotations have no item yet; show them so the
    // user sees the upload landed. They stay out of snapshots until labelled.
    const labelled = new Set(items.map((i) => i.image));
    state.unlabelled = state.images
      .filter((img) => !labelled.has(img.name))
      .map((img) => ({ image: img.name, width: img.width, height: img.height, review_status: "unlabelled", annotations: [], unlabelled: true }));
    const filter = $("#example-filter").value;
    const all = [...items, ...state.unlabelled].sort((a, b) => (a.image < b.image ? -1 : a.image > b.image ? 1 : 0));
    const only = state.onlyImages && state.onlyImagesDataset === state.datasetId ? state.onlyImages : null;
    const shown = only ? all.filter((i) => only.has(i.image)) : all;
    state.items = filter ? shown.filter((i) => i.review_status === filter) : shown;
    const snaps = (await api("GET", `/v1/datasets/${state.datasetId}/snapshots`)).snapshots;
    state.snapshot = snaps.length ? snaps.reduce((a, b) => (a.id > b.id ? a : b)) : null;
  }
  await loadCropPrep();
  $("#topbar-dataset").textContent = state.dataset ? `${state.dataset.name} · ${state.dataset.task}` : "Model fine-tuning";
  renderDatasetPanel();
  renderPipelines();
  renderItems();
  renderSnapshot();
  renderLabelFormat();
  await renderDetectionTools();
  renderCropPrep();
  await loadParentModels();
}

function renderPipelines() {
  const sel = $("#pipeline-select");
  const task = state.dataset ? state.dataset.task : null;
  const prev = sel.value;
  const usable = state.pipelines.filter((p) => !task || p.task === task);
  sel.replaceChildren(...usable.map((p) => el("option", {
    value: p.id,
    text: p.status === "available" ? p.name : `${p.name} (planned)`,
    disabled: p.status !== "available",
  })));
  const firstAvailable = usable.find((p) => p.status === "available");
  if (usable.some((p) => p.id === prev && p.status === "available")) sel.value = prev;
  else if (firstAvailable) sel.value = firstAvailable.id;
  renderPipelineSummary();
}

// What each pipeline does, written from the trainer code (src/sparrow_finetune/pipelines/*.py).
const PIPELINE_GUIDES = {
  dinov3_linear: {
    summary: "Trains only a small linear classifier on top of a frozen image encoder. The encoder is never changed, so training is fast and works with a few dozen labelled crops per species.",
    steps: [
      "Prepare: crops every labelled box (or whole image), splits train/validation by group so photos of the same event stay together, and embeds each crop once with the encoder (cached, so retraining skips this).",
      "Train: fits a one-vs-rest logistic-regression head (SGD, L2) one pass per epoch and keeps the epoch with the best validation macro-F1. It must score within 0.02 macro-F1 of a full-batch logistic regression on the same features, or the run fails (with fewer than 100 validation crops it only warns).",
      "Calibrate: picks the lowest confidence threshold that keeps validation accuracy at or above the target (default 95%); predictions below it are labelled unknown and sent to review.",
      "Export: writes the head as ONNX and checks it gives the same scores as scikit-learn before it can be activated.",
    ],
    needs: "Classification labels, at least 2 examples per class. Runs on CPU; no GPU needed.",
    outputs: "Accuracy, macro-F1, per-class scores, confusion matrix, calibrated threshold, near-duplicate count between train and validation.",
    when: "First choice for species classification on camera-trap crops, especially with small or medium datasets.",
  },
  conventional_classifier: {
    summary: "Fine-tunes a whole image classifier (an architecture from timm, PyTorch Image Models: resnet50 unless a base model or finished model sets another) end to end on your crops. Slower than the linear probe and needs more data, but every layer adapts to your images.",
    steps: [
      "Prepare: crops every labelled box (or whole image) and makes the same group-aware train/validation split.",
      "Train: warms up the new classifier head on a frozen backbone, then fine-tunes the whole network with AdamW (backbone at a lower learning rate, cosine decay, label smoothing, augmentation). Keeps the epoch with the best validation macro-F1; stops early when it stops improving.",
      "Calibrate: same confidence threshold and unknown label as the linear probe.",
      "Export: one ONNX graph from RGB crop to class probabilities, checked against PyTorch before activation.",
    ],
    needs: "Classification labels; a few hundred examples per class work best. A GPU worker is strongly recommended.",
    outputs: "Accuracy, macro-F1, per-class scores, confusion matrix, calibrated threshold. Can warm-start from a model you trained earlier.",
    when: "When the linear probe plateaus and you have enough labelled data and a GPU.",
  },
  yolo_detector: {
    summary: "Fine-tunes an Ultralytics YOLO detector to draw and label boxes on whole photos, starting from YOLO11n or MegaDetector v6 weights.",
    steps: [
      "Prepare: writes a YOLO dataset (images, one label file per image, data.yaml) with a group-aware train/validation split.",
      "Train: runs YOLO training (default 50 epochs, 640 px images); batch size can be sized automatically from free GPU memory. Reports progress every epoch.",
      "Evaluate: validates the best checkpoint on the validation split.",
      "Export: writes a fixed-size ONNX model and checks it against the PyTorch model before activation.",
    ],
    needs: "Detection boxes on whole images (draw them, import COCO / Camera Trap DP / Pascal VOC, or pre-draw with MegaDetector and review). A GPU worker is strongly recommended.",
    outputs: "Precision, recall, mAP@50 and mAP@50-95, overall and per class. Can warm-start from a detector you trained earlier.",
    when: "When you need box locations, or want a detector that recognises your species directly.",
  },
};

function renderPipelineGuide(p) {
  const box = $("#pipeline-guide");
  const guide = p ? PIPELINE_GUIDES[p.id] : null;
  box.hidden = !guide;
  if (!guide) { box.replaceChildren(); return; }
  const row = (term, text) => el("div", {}, el("dt", { text: term }), el("dd", { text }));
  box.replaceChildren(
    el("p", { className: "pipeline-guide-summary", text: guide.summary }),
    el("strong", { text: "What happens when you train" }),
    el("ol", {}, ...guide.steps.map((s) => el("li", { text: s }))),
    el("dl", {}, row("Data needed", guide.needs), row("You get", guide.outputs), row("Use it", guide.when)),
  );
}

function renderPipelineSummary() {
  const p = state.pipelines.find((x) => x.id === $("#pipeline-select").value);
  $("#pipeline-name").textContent = p ? p.name : "No pipeline available for this dataset";
  $("#pipeline-icon").textContent = p ? p.name.charAt(0).toUpperCase() : "-";
  const badge = $("#pipeline-status");
  badge.hidden = !p;
  if (p) {
    badge.textContent = p.status === "available" ? "Available" : "Planned";
    badge.className = `status-badge ${p.status === "available" ? "status-ready" : "status-planned"}`;
    const req = p.requires && p.requires.length ? `Needs a worker with: ${p.requires.join(", ")}.` : "";
    $("#pipeline-description").textContent = `Task: ${p.task}. ${req}`;
  } else {
    $("#pipeline-description").textContent = "";
  }
  renderPipelineGuide(p);
  updateTrainButton();
}

function updateTrainButton() {
  const p = state.pipelines.find((x) => x.id === $("#pipeline-select").value);
  $("#start-training").disabled = !(state.snapshot && p && p.status === "available");
  $("#base-model-field").hidden = !(p && WARM_START_PIPELINES.has(p.id));
  applyBatchField(p);
  $("#target-accuracy-field").hidden = !(p && CLASSIFIER_PIPELINES.has(p.id));
  $("#stock-model-field").hidden = !(p && CLASSIFIER_PIPELINES.has(p.id) && state.stockModels.length);
  applyTrainingSettings(p);
  applyBaseModelGate();
}

// Defaults of the "Training settings" fields, per pipeline (each pipeline's _PARAMS; pinned by
// tests/finetune/test_training_settings.py). Strict JSON so the test can read it.
const TRAINING_SETTINGS = {
  "yolo_detector": {"epochs": {"param": "epochs", "value": 50, "max": 1000}, "patience": 10, "val_fraction": 0.2, "lr": {"param": "lr0", "value": 0.01}},
  "conventional_classifier": {"epochs": {"param": "epochs", "value": 30, "max": 1000}, "patience": 5, "val_fraction": 0.2, "augment": true, "lr": {"param": "lr", "value": 0.001}},
  "dinov3_linear": {"epochs": {"param": "max_epochs", "value": 50, "max": 500}, "val_fraction": 0.2}
};
let settingsPipeline = null;

function applyTrainingSettings(p) {
  const spec = p ? TRAINING_SETTINGS[p.id] : undefined;
  const has = (key) => Boolean(spec && spec[key] !== undefined);
  $("#epochs-field").hidden = !has("epochs");
  $("#patience-field").hidden = !has("patience");
  $("#val-fraction-field").hidden = !has("val_fraction");
  $("#augment-field").hidden = !has("augment");
  $("#learning-speed-field").hidden = !has("lr");
  $("#training-settings").hidden = !(p && (spec || BATCH_FIELDS[p.id] || CLASSIFIER_PIPELINES.has(p.id)));
  if (!spec || settingsPipeline === p.id) return;
  // A different pipeline starts from its own defaults.
  settingsPipeline = p.id;
  if (spec.epochs) {
    $("#epochs").max = String(spec.epochs.max);
    $("#epochs").value = String(spec.epochs.value);
    $("#epochs-help").textContent = `Default ${spec.epochs.value}. More passes can learn more but take longer.`;
  }
  if (spec.patience !== undefined) {
    $("#early-stop").checked = true;
    $("#patience").disabled = false;
    $("#patience").value = String(spec.patience);
    $("#patience-help").textContent = `Default: on, after ${spec.patience} passes. Saves time; the best model so far is kept.`;
  }
  $("#val-fraction").value = String(spec.val_fraction);
  $("#augment").checked = true;
  $("#learning-speed").value = "1";
}

// Only settings the user changed are sent; the pipeline's own defaults apply to the rest.
function trainingSettingsParams(pipeline) {
  const spec = TRAINING_SETTINGS[pipeline];
  const params = {};
  if (!spec) return params;
  const whole = (selector, max, what) => {
    const n = Number($(selector).value);
    if (!Number.isInteger(n) || n < 1 || n > max) throw new Error(`${what} must be a whole number from 1 to ${max}.`);
    return n;
  };
  if (spec.epochs) {
    const n = whole("#epochs", spec.epochs.max, "Training length");
    if (n !== spec.epochs.value) params[spec.epochs.param] = n;
  }
  if (spec.patience !== undefined) {
    const n = $("#early-stop").checked ? whole("#patience", 1000, "Passes without improvement") : 0;
    if (n !== spec.patience) params.patience = n;
  }
  const fraction = Number($("#val-fraction").value);
  if (fraction !== spec.val_fraction) params.val_fraction = fraction;
  if (spec.augment !== undefined && !$("#augment").checked) params.augment = false;
  const factor = Number($("#learning-speed").value);
  if (spec.lr && factor !== 1) params[spec.lr.param] = spec.lr.value * factor;
  return params;
}

// Pipelines with a batch-size parameter: default and largest value (yolo_detector / conventional_classifier _PARAMS).
const BATCH_FIELDS = {
  yolo_detector: { value: 16, max: 256 },
  conventional_classifier: { value: 32, max: 1024 },
};
let batchFieldPipeline = null;

function applyBatchField(p) {
  const spec = p ? BATCH_FIELDS[p.id] : undefined;
  $("#batch-field").hidden = !spec;
  if (!spec || batchFieldPipeline === p.id) return;
  batchFieldPipeline = p.id;
  const input = $("#batch-size");
  input.max = String(spec.max);
  input.value = String(spec.value);
  $("#batch-help").textContent = `Default ${spec.value}. Automatic falls back to ${p.id === "yolo_detector" ? "a small batch" : "the default"} on a CPU.`;
}

function batchParam(pipeline) {
  const spec = BATCH_FIELDS[pipeline];
  if ($("#batch-auto").checked) return "auto";
  const n = Number($("#batch-size").value);
  if (!Number.isInteger(n) || n < 1 || n > spec.max) throw new Error(`Batch size must be a whole number from 1 to ${spec.max}.`);
  return n;
}

// Starting weights for yolo_detector (yolo_detector.MEGADETECTOR_BASES plus the stock YOLO11n).
const START_WEIGHTS_HELP = {
  "yolo11n.pt": "Ultralytics YOLO11n trained on everyday photos (COCO). Small; good when your objects are not wildlife.",
  "mdv6-yolov10-c": "MegaDetector v6 compact (YOLOv10-c, 6 MB): already finds animals, people and vehicles in camera-trap photos. Fast; works on a CPU.",
  "mdv6-yolov10-e": "MegaDetector v6 extra (YOLOv10-e): more accurate, but a 128 MB download and slow without a GPU. It was trained on 1280-pixel images, so training uses 1280 by default.",
};
// The size each starting weight was trained at (yolo_detector.MEGADETECTOR_BASES[...].imgsz).
const START_WEIGHTS_IMGSZ = { "yolo11n.pt": 640, "mdv6-yolov10-c": 640, "mdv6-yolov10-e": 1280 };

function renderStartWeights() {
  const yolo = $("#pipeline-select").value === "yolo_detector";
  const fromBase = Boolean(state.baseModel && state.baseModel.row.pipeline === "yolo_detector");
  $("#start-weights-field").hidden = !yolo || fromBase;
  $("#start-weights-help").textContent = START_WEIGHTS_HELP[$("#start-weights").value] || "";
  $("#imgsz-field").hidden = !yolo;
}

// FT-31: picking starting weights resets the image size to the size those weights were trained at.
function syncImgszToWeights() {
  $("#imgsz").value = String(START_WEIGHTS_IMGSZ[$("#start-weights").value] || 640);
}

// Pipelines that can start from a base model's weights (base_models.WARM_START_PIPELINES).
const WARM_START_PIPELINES = new Set(["conventional_classifier", "yolo_detector", "dinov3_linear"]);
// The pipeline a local classifier package (POST /v1/base-models/local) warm-starts.
const LOCAL_BASE_PIPELINE = "conventional_classifier";

function baseModelAllowed() {
  return WARM_START_PIPELINES.has($("#pipeline-select").value);
}

// A Studio model can be the base only for a job of its own pipeline.
function usableAsBase(m) {
  return Boolean(m && m.importable && m.pipeline === $("#pipeline-select").value);
}

function applyBaseModelGate() {
  const allowed = baseModelAllowed();
  const p = state.pipelines.find((x) => x.id === $("#pipeline-select").value);
  const note = $("#base-model-note");
  note.hidden = allowed;
  if (!allowed) {
    const name = p ? p.name : "This pipeline";
    note.textContent = `${name} does not use a base model. Switch to the image classifier, DINOv3 or YOLO11 detector pipeline to start from a Studio model.`;
  }
  $("#studio-models-block").classList.toggle("gated", !allowed);
  $("#local-base-model-block").hidden = $("#pipeline-select").value !== LOCAL_BASE_PIPELINE;
  // Keep the choice while a pipeline without base models is shown; drop it for another warm-start pipeline.
  if (allowed && state.baseModel && state.baseModel.row.pipeline !== $("#pipeline-select").value) state.baseModel = null;
  if (state.studioModels) fillStudioModels(state.studioModels);
  else $("#studio-model-select").disabled = true;
  renderStudioModelChoice();
  renderParentModels();
  renderBaseModel();
}

function itemLabel(item) {
  const labels = [...new Set((item.annotations || []).map((a) => a.label))];
  return labels.length ? labels.join(", ") : "(no label)";
}

function renderItems() {
  const grid = $("#example-grid");
  const counts = (state.dataset && state.dataset.review_counts) || { pending: 0, confirmed: 0, excluded: 0 };
  $("#count-pending").textContent = counts.pending;
  $("#count-confirmed").textContent = counts.confirmed;
  $("#count-excluded").textContent = counts.excluded;
  $("#dataset-image-count").textContent = state.dataset
    ? `${plural(state.images.length, "image")} in this dataset · ${state.unlabelled.length} not labelled`
    : "";
  $("#example-empty").hidden = state.items.length > 0;
  grid.replaceChildren(...state.items.map(renderCard));
}

const STATE_TEXT = { pending: "Pending review", confirmed: "Confirmed", excluded: "Excluded", unlabelled: "Not labelled" };
const STATE_CLASS = { pending: "attention", confirmed: "confirmed", excluded: "excluded", unlabelled: "attention" };

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function renderCard(item) {
  const status = item.review_status;
  const src = `/v1/datasets/${state.datasetId}/images/${encodeURIComponent(item.image)}/content`;
  const wrap = el("div", { class: "photo-wrap", "data-view-image": "1" },
    el("img", { src, alt: `Training image ${item.image}`, loading: "lazy" }),
    item.group ? el("span", { class: "camera-label", text: item.group }) : null,
    el("span", { class: `confidence-label ${status === "pending" || status === "unlabelled" ? "attention" : status === "excluded" ? "excluded" : ""}`, text: STATE_TEXT[status] }),
  );
  wrap.append(...boxSpans(item));
  wrap.setAttribute("role", "button");
  wrap.setAttribute("tabindex", "0");
  wrap.setAttribute("aria-label", `Enlarge ${item.image}`);
  const labelInput = el("input", { type: "text", value: itemLabel(item) === "(no label)" ? "" : itemLabel(item), "data-label-input": "1" });
  labelInput.setAttribute("aria-label", `Label for ${item.image}`);
  const btn = (action, text) => el("button", {
    type: "button",
    "data-example-action": action,
    text,
    class: (action === "confirm" && status === "confirmed") || (action === "exclude" && status === "excluded") ? "active" : "",
  });
  const detection = Boolean(state.dataset && state.dataset.task === "detection");
  const boxesBtn = boxable() && item.width && item.height
    ? el("div", { class: "example-actions" }, btn("boxes", item.unlabelled ? "Draw boxes" : "Edit boxes"))
    : null;
  let controls;
  if (!item.unlabelled) {
    controls = [
      el("label", {}, "Training label", labelInput),
      el("div", { class: "example-actions" }, btn("confirm", "Confirm"), btn("correct", "Correct"), btn("exclude", "Exclude")),
      boxesBtn,
    ];
  } else if (!detection) {
    controls = [
      el("label", {}, "Training label", labelInput),
      el("div", { class: "example-actions" }, btn("correct", "Save label")),
      boxesBtn,
    ];
  } else {
    controls = [el("p", { class: "empty-note", text: "No boxes yet. Draw them here or import annotations." }), boxesBtn];
  }
  const card = el("article", { class: "example-card", "data-image": item.image, "data-status": status === "pending" || status === "unlabelled" ? "attention" : status },
    wrap,
    el("div", { class: "example-body" },
      el("span", { class: `example-state ${STATE_CLASS[status]}`, text: STATE_TEXT[status] }),
      el("h3", { text: itemLabel(item) }),
      el("p", { text: item.image }),
      ...controls,
    ),
  );
  return card;
}

const viewer = { index: -1 };

function boxSpans(item) {
  // Classification items carry boxes too; the classifier trains on each box crop. A dataset
  // trained on whole images has no boxes to show.
  if (!(state.dataset && item.width && item.height) || wholeImage()) return [];
  return (item.annotations || []).filter((a) => a.bbox).map((a) => {
    const [x, y, w, h] = a.bbox;
    return el("span", {
      class: "box bbox",
      style: `left:${(100 * x) / item.width}%;top:${(100 * y) / item.height}%;width:${(100 * w) / item.width}%;height:${(100 * h) / item.height}%`,
      text: a.label,
    });
  });
}

function fitViewerImage() {
  const img = $("#image-viewer-img");
  if (!img.naturalWidth) return;
  const maxW = Math.min(1200, window.innerWidth * 0.96) - 26;
  const maxH = window.innerHeight * 0.96 - 110;
  const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight);
  img.style.width = `${Math.floor(img.naturalWidth * scale)}px`;
  img.style.height = `${Math.floor(img.naturalHeight * scale)}px`;
}

function showViewerItem(index) {
  const items = state.items;
  if (index < 0 || index >= items.length) return;
  viewer.index = index;
  const item = items[index];
  const img = $("#image-viewer-img");
  img.src = `/v1/datasets/${state.datasetId}/images/${encodeURIComponent(item.image)}/content`;
  img.alt = `Training image ${item.image}`;
  $("#image-viewer-boxes").replaceChildren(...boxSpans(item));
  $("#image-viewer-caption").textContent =
    `${itemLabel(item)} · ${STATE_TEXT[item.review_status]} · ${item.image} · ${index + 1} of ${items.length}`;
  $("#image-viewer-prev").disabled = index === 0;
  $("#image-viewer-next").disabled = index === items.length - 1;
}

function openImageViewer(image) {
  const index = state.items.findIndex((i) => i.image === image);
  if (index < 0) return;
  showViewerItem(index);
  const dlg = $("#image-viewer");
  if (!dlg.open) dlg.showModal();
  $("#image-viewer-close").focus();
}

function onPhotoOpen(event) {
  if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
  const photo = event.target.closest("[data-view-image]");
  if (!photo) return;
  event.preventDefault();
  openImageViewer(photo.closest(".example-card").dataset.image);
}

async function onExampleAction(event) {
  const button = event.target.closest("[data-example-action]");
  if (!button) return;
  const card = button.closest(".example-card");
  const image = card.dataset.image;
  const item = state.items.find((i) => i.image === image);
  const action = button.dataset.exampleAction;
  if (action === "boxes") {
    openBoxEditor(item);
    return;
  }
  const path = `/v1/datasets/${state.datasetId}/items/${encodeURIComponent(image)}`;
  if (item && item.unlabelled) {
    const label = card.querySelector("[data-label-input]").value.trim();
    if (!label) throw new Error("Enter a label before saving.");
    await api("POST", `/v1/datasets/${state.datasetId}/annotations`, { items: [{ image, annotations: [{ label }] }] });
    await api("PATCH", path, { review_status: "confirmed" });
  } else if (action === "confirm") {
    await api("PATCH", path, { review_status: "confirmed" });
  } else if (action === "exclude") {
    await api("PATCH", path, { review_status: "excluded" });
  } else if (action === "correct") {
    const label = card.querySelector("[data-label-input]").value.trim();
    if (!label) throw new Error("Enter a label before correcting.");
    const anns = item.annotations && item.annotations.length
      ? item.annotations.map((a) => (a.bbox ? { label, bbox: a.bbox } : { label }))
      : [{ label }];
    await api("PATCH", path, { annotations: anns, review_status: "confirmed" });
  }
  await loadDataset();
}

// ------------------------------------------------------------------ box editor

const BOX_NUDGE = 1;       // image pixels per arrow key press
const BOX_NUDGE_BIG = 10;  // with Shift held
const BOX_MIN = 2;         // a drawn box smaller than this (image pixels) is discarded
const editor = { item: null, boxes: [], selected: -1, drag: null };

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function boxCount(n) {
  return `${n} ${n === 1 ? "box" : "boxes"}`;
}

const NEW_LABEL = "\u0000new";  // the "+ New label…" option; cannot collide with a real label
// Labels added in the box editor, per dataset, so they are offered on every image before any box using them is saved.
const addedBoxLabels = new Map();

function datasetLabels() {
  const labels = new Set(addedBoxLabels.get(state.datasetId) || []);
  for (const it of state.items) for (const a of it.annotations || []) if (a.label) labels.add(a.label);
  for (const label of Object.keys((state.snapshot && state.snapshot.label_counts) || {})) labels.add(label);
  return [...labels].sort();
}

function newLabelOpen() {
  return !$("#box-new-label-row").hidden;
}

function showNewLabel(open) {
  $("#box-new-label-row").hidden = !open;
  if (open) {
    $("#box-label").value = NEW_LABEL;
    $("#box-new-label").value = "";
  }
}

// Rebuild the label select around `current`, which may be "" (no label yet) or a label not seen elsewhere.
function fillLabelOptions(current) {
  const labels = datasetLabels();
  if (current && !labels.includes(current)) labels.push(current);
  const options = labels.map((l) => el("option", { value: l, text: l }));
  if (!current) options.unshift(el("option", { value: "", text: "Choose a label" }));
  options.push(el("option", { value: NEW_LABEL, text: "+ New label…" }));
  $("#box-label").replaceChildren(...options);
  $("#box-label").value = current || "";
  showNewLabel(false);
}

function chosenLabel() {
  const v = $("#box-label").value;
  return v === NEW_LABEL ? "" : v;
}

function onBoxLabelChange() {
  if ($("#box-label").value === NEW_LABEL) {
    showNewLabel(true);
    $("#box-new-label").focus();
    return;
  }
  showNewLabel(false);
  const b = editor.boxes[editor.selected];
  if (b) b.label = chosenLabel();
  renderBoxes();
}

function addNewLabel() {
  const label = $("#box-new-label").value.trim();
  const log = $("#box-label-log");
  if (!label) {
    log.textContent = "Type the new label, then add it.";
    $("#box-new-label").focus();
    return;
  }
  const known = datasetLabels().includes(label);
  if (!addedBoxLabels.has(state.datasetId)) addedBoxLabels.set(state.datasetId, new Set());
  addedBoxLabels.get(state.datasetId).add(label);
  const b = editor.boxes[editor.selected];
  if (b) b.label = label;
  fillLabelOptions(label);
  log.textContent = known
    ? `"${label}" already exists; selected it.`
    : `Added label "${label}". It is offered on every image in this dataset.`;
  renderBoxes();
  $("#box-stage").focus();
}

function onNewLabelKey(e) {
  if (e.key !== "Enter") return;
  e.preventDefault();
  e.stopPropagation();  // Enter elsewhere in the dialog saves the boxes
  addNewLabel();
}

function openBoxEditor(item) {
  if (!item || !item.width || !item.height) throw new Error("This image has no recorded size, so its boxes cannot be edited.");
  editor.item = item;
  editor.boxes = (item.annotations || []).filter((a) => a.bbox)
    .map((a) => ({ label: a.label || "", x: a.bbox[0], y: a.bbox[1], w: a.bbox[2], h: a.bbox[3] }));
  editor.selected = editor.boxes.length ? 0 : -1;
  editor.drag = null;
  const labels = datasetLabels();
  $("#box-label-log").textContent = "";
  fillLabelOptions(editor.boxes.length ? editor.boxes[0].label : labels[0] || "");
  if (!labels.length && !editor.boxes.length) showNewLabel(true);
  $("#box-editor-image").textContent = `${item.image} · ${item.width} × ${item.height} px`;
  const img = $("#box-editor-img");
  img.src = `/v1/datasets/${state.datasetId}/images/${encodeURIComponent(item.image)}/content`;
  img.alt = `Image ${item.image}`;
  const stage = $("#box-stage");
  stage.style.aspectRatio = `${item.width} / ${item.height}`;
  stage.style.width = `min(100%, ${((62 * item.width) / item.height).toFixed(2)}vh)`;
  $("#box-editor-error").hidden = true;
  renderBoxes();
  $("#box-editor").showModal();
  stage.focus();
}

function renderBoxes() {
  const { item } = editor;
  if (!item) return;
  $("#box-layer").replaceChildren(...editor.boxes.map((b, i) => {
    const node = el("div", {
      class: `edit-box${i === editor.selected ? " selected" : ""}`,
      "data-box": i,
      style: `left:${(100 * b.x) / item.width}%;top:${(100 * b.y) / item.height}%;width:${(100 * b.w) / item.width}%;height:${(100 * b.h) / item.height}%`,
    }, el("span", { class: "edit-box-label", text: b.label || "(no label)" }));
    if (i === editor.selected) {
      for (const corner of ["nw", "ne", "sw", "se"]) node.append(el("span", { class: `edit-handle ${corner}`, "data-handle": corner }));
    }
    return node;
  }));
  const sel = editor.boxes[editor.selected];
  $("#box-count").textContent = `${boxCount(editor.boxes.length)}${sel ? ` · box ${editor.selected + 1} selected` : ""}`;
  $("#box-delete").disabled = !sel;
  const labelFocused = [$("#box-label"), $("#box-new-label")].includes(document.activeElement);
  if (sel && !labelFocused && !(newLabelOpen() && !sel.label) && $("#box-label").value !== sel.label) fillLabelOptions(sel.label);
}

function imagePoint(e) {
  const r = $("#box-stage").getBoundingClientRect();
  const { width, height } = editor.item;
  return {
    x: clamp(((e.clientX - r.left) * width) / r.width, 0, width),
    y: clamp(((e.clientY - r.top) * height) / r.height, 0, height),
  };
}

function onStageDown(e) {
  if (!editor.item || e.button !== 0) return;
  e.preventDefault();
  const p = imagePoint(e);
  const handle = e.target.closest("[data-handle]");
  const boxNode = e.target.closest("[data-box]");
  if (handle && editor.boxes[editor.selected]) {
    editor.drag = { mode: "resize", corner: handle.dataset.handle, orig: { ...editor.boxes[editor.selected] } };
  } else if (boxNode) {
    editor.selected = Number(boxNode.dataset.box);
    editor.drag = { mode: "move", start: p, orig: { ...editor.boxes[editor.selected] } };
  } else {
    editor.boxes.push({ label: chosenLabel(), x: p.x, y: p.y, w: 0, h: 0 });
    editor.selected = editor.boxes.length - 1;
    editor.drag = { mode: "draw", start: p };
  }
  const stage = $("#box-stage");
  stage.setPointerCapture(e.pointerId);
  stage.focus();
  renderBoxes();
}

function onStageMove(e) {
  const d = editor.drag;
  const b = editor.boxes[editor.selected];
  if (!d || !b) return;
  const p = imagePoint(e);
  const { width, height } = editor.item;
  if (d.mode === "draw") {
    Object.assign(b, { x: Math.min(d.start.x, p.x), y: Math.min(d.start.y, p.y), w: Math.abs(p.x - d.start.x), h: Math.abs(p.y - d.start.y) });
  } else if (d.mode === "move") {
    b.x = clamp(d.orig.x + p.x - d.start.x, 0, width - d.orig.w);
    b.y = clamp(d.orig.y + p.y - d.start.y, 0, height - d.orig.h);
  } else {
    let [x1, y1, x2, y2] = [d.orig.x, d.orig.y, d.orig.x + d.orig.w, d.orig.y + d.orig.h];
    if (d.corner.includes("w")) x1 = p.x; else x2 = p.x;
    if (d.corner.includes("n")) y1 = p.y; else y2 = p.y;
    Object.assign(b, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1) });
  }
  renderBoxes();
}

function onStageUp() {
  const d = editor.drag;
  if (!d) return;
  editor.drag = null;
  const b = editor.boxes[editor.selected];
  if (b && d.mode !== "move" && (b.w < BOX_MIN || b.h < BOX_MIN)) {
    if (d.mode === "draw") {
      editor.boxes.splice(editor.selected, 1);
      editor.selected = -1;
    } else {
      Object.assign(b, d.orig);
    }
  }
  renderBoxes();
}

function deleteSelectedBox() {
  if (!editor.boxes[editor.selected]) return;
  editor.boxes.splice(editor.selected, 1);
  editor.selected = Math.min(editor.selected, editor.boxes.length - 1);
  renderBoxes();
  $("#box-stage").focus();
}

function onEditorKey(e) {
  if (!editor.item) return;
  if (e.key === "Enter") {
    if (e.target.matches("button, a")) return;
    e.preventDefault();
    saveBoxes();
    return;
  }
  if (e.target.matches("input, select, textarea")) return;
  const b = editor.boxes[editor.selected];
  if (!b) return;
  if (e.key === "Delete" || e.key === "Backspace") {
    e.preventDefault();
    deleteSelectedBox();
    return;
  }
  const step = e.shiftKey ? BOX_NUDGE_BIG : BOX_NUDGE;
  const move = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key];
  if (!move) return;
  e.preventDefault();
  b.x = clamp(b.x + move[0], 0, editor.item.width - b.w);
  b.y = clamp(b.y + move[1], 0, editor.item.height - b.h);
  renderBoxes();
}

function closeBoxEditor() {
  editor.item = null;
  editor.drag = null;
  if ($("#box-editor").open) $("#box-editor").close();
}

// Pixel boxes for the API: one decimal, kept inside the image.
function boxAnnotations() {
  const { width, height } = editor.item;
  const r = (v) => Math.round(v * 10) / 10;
  return editor.boxes.map((b) => {
    const x = clamp(r(b.x), 0, width);
    const y = clamp(r(b.y), 0, height);
    return { label: b.label.trim(), bbox: [x, y, clamp(r(b.w), 0, width - x), clamp(r(b.h), 0, height - y)] };
  });
}

// Errors stay inside the dialog: the page banner sits under the modal.
async function saveBoxes() {
  const errBox = $("#box-editor-error");
  errBox.hidden = true;
  const item = editor.item;
  if (!item) return;
  const button = $("#box-save");
  try {
    const anns = boxAnnotations();
    const unnamed = anns.findIndex((a) => !a.label);
    if (unnamed >= 0) {
      editor.selected = unnamed;
      renderBoxes();
      $("#box-label").focus();
      throw new Error(`Box ${unnamed + 1} has no label. Choose one (or add a new label), then save.`);
    }
    button.disabled = true;
    const path = `/v1/datasets/${state.datasetId}/items/${encodeURIComponent(item.image)}`;
    // An image that was never labelled has no item to PATCH yet: create it, then confirm it.
    const create = async () => {
      await api("POST", `/v1/datasets/${state.datasetId}/annotations`, { items: [{ image: item.image, annotations: anns }] });
      await api("PATCH", path, { review_status: "confirmed" });
    };
    if (item.unlabelled) {
      await create();
    } else {
      try {
        await api("PATCH", path, { annotations: anns, review_status: "confirmed" });
      } catch (err) {
        if (!(err instanceof ApiError && err.status === 404)) throw err;
        await create();
      }
    }
    closeBoxEditor();
    toast(`Saved ${boxCount(anns.length)} for ${item.image}`);
    await guarded(loadDataset)();
  } catch (err) {
    errBox.textContent = err && err.message ? err.message : String(err);
    errBox.hidden = false;
  } finally {
    button.disabled = false;
  }
}

// ------------------------------------------------------------------ detection tools: assist

// api/datasets.py ASSIST_MAX_ITEMS: one assist call labels at most this many images.
const ASSIST_MAX_ITEMS = 500;
// Detection confidence cut-off for pre-drawn boxes. The API accepts it; the UI does not expose it.
function assistConf() {
  const v = Number($("#assist-conf").value);
  if (!Number.isFinite(v) || v < 0.01 || v > 1) throw new Error("Minimum confidence must be between 0.01 and 1.");
  return v;
}
// assist.MEGADETECTOR_ID: the built-in default detector.
const MEGADETECTOR = "megadetector";

function isDetection() {
  return Boolean(state.dataset && state.dataset.task === "detection");
}

function wholeImage() {
  return Boolean(state.dataset && state.dataset.task === "classification" && state.dataset.whole_image);
}

// Detection boxes, or classification crops unless the dataset trains on whole images.
function boxable() {
  return Boolean(state.dataset && (state.dataset.task === "detection" || !state.dataset.whole_image));
}

// FT-35: classification datasets get assist too; the boxes become the crops the classifier trains on.
const ASSIST_HELP = {
  detection: "MegaDetector (the default) finds animals, people and vehicles in camera-trap photos; relabel its boxes to species as you review, or name the species below. You can also pick a detector you trained here. Boxes arrive as pending, so review each one before training. Images that already have boxes or are confirmed are left alone unless you tick \"Replace existing boxes\". Excluded images are never touched.",
  classification: "MegaDetector (the default) boxes each animal so the classifier trains on the animal instead of the whole photo. People and vehicles are left out. Tick \"Name species with SpeciesNet\" to have SpeciesNet name the animal in each box (matched to your labels by common name, Latin name or genus where possible). Otherwise each box takes the name below if you give one, otherwise the image's label when it has exactly one, otherwise \"animal\" for you to correct. You can also pick a detector you trained here. Boxes arrive as pending, so review each image before training. Images that already have boxes or are confirmed are left alone unless you tick \"Replace existing boxes\". Excluded images are never touched.",
};

const WHOLE_IMAGE_ASSIST_HELP = "This dataset trains on whole images, so assist gives each image a label instead of boxes. MegaDetector finds the animals; tick \"Name species with SpeciesNet\" to name them, otherwise the name below is used. Labels arrive as pending, so review each image before training. Images that already have a label or are confirmed are left alone unless you tick \"Replace existing labels\". Excluded images are never touched.";

// FT-44: a whole-image dataset gets image labels from assist, never boxes; say so on every control.
function assistWords() {
  return wholeImage()
    ? { toggle: "Label images with a detector", animal: "Label the images as", overwrite: "Replace existing labels",
      button: "Label images…", title: "Confirm labelling images", confirm: "Yes, label images", busy: "Labelling images",
      things: "labels" }
    : { toggle: "Pre-draw boxes with a detector", animal: "Label animal boxes as", overwrite: "Replace existing boxes",
      button: "Pre-draw boxes…", title: "Confirm pre-drawing boxes", confirm: "Yes, pre-draw boxes", busy: "Pre-drawing boxes",
      things: "boxes" };
}

function renderAssistWording() {
  const w = assistWords();
  $("#assist-enable-text").textContent = w.toggle;
  $("#assist-animal-text").textContent = w.animal;
  $("#assist-overwrite-text").textContent = w.overwrite;
  $("#assist-btn").textContent = w.button;
  $("#assist-confirm-title").textContent = w.title;
  $("#assist-confirm-btn").textContent = w.confirm;
}

async function renderDetectionTools() {
  const task = state.dataset && state.dataset.task;
  const supported = task === "detection" || task === "classification";
  $("#assist-block").hidden = !supported;
  if (!supported) return;
  $("#assist-help").textContent = wholeImage() ? WHOLE_IMAGE_ASSIST_HELP : ASSIST_HELP[task];
  renderAssistWording();
  $("#assist-animal-label").placeholder = task === "classification" ? "the image's label" : "animal";
  const sel = $("#assist-model");
  const [assistInfo, trained] = await Promise.all([
    api("GET", "/v1/assist/models"),
    workspaceList("/v1/models", "models").then((ms) => ms.filter((m) => m.pipeline === "yolo_detector")),
  ]);
  const assistModels = assistInfo.models;
  state.speciesnet = (assistInfo.classifiers || []).find((c) => c.id === "speciesnet") || null;
  const byId = new Map(trained.map((m) => [m.id, m]));
  const options = [];
  for (const m of assistModels) {
    if (m.id === MEGADETECTOR) {
      options.push(el("option", {
        value: m.id,
        text: m.available ? `${m.name} (default)` : `${m.name}: not installed on this server`,
        disabled: !m.available,
      }));
    } else if (byId.has(m.id)) {
      const t = byId.get(m.id);
      options.push(el("option", { value: t.id, text: `${modelTitle(t)} (job ${t.job_id})` }));
    }
  }
  const usable = options.filter((o) => !o.disabled);
  if (!usable.length) options.push(el("option", { value: "", text: "No detector available" }));
  const prev = sel.value;
  sel.replaceChildren(...options);
  if (usable.some((o) => o.value === prev)) sel.value = prev;
  else if (usable.length) sel.value = usable[0].value;
  sel.disabled = !usable.length;
  $("#assist-label-list").replaceChildren(...[...datasetLabels()].sort().map((l) => el("option", { value: l })));
  onAssistModelChange();
  $("#assist-btn").disabled = !usable.length || !state.images.length;
  if ($("#assist-enable").checked) await refreshAssistSummary();
}

function speciesnetOn() {
  return !$("#assist-classifier-row").hidden && $("#assist-speciesnet").checked;
}

function onAssistModelChange() {
  const md = $("#assist-model").value === MEGADETECTOR;
  const sn = state.speciesnet;
  const row = $("#assist-classifier-row");
  row.hidden = !(md && state.dataset && state.dataset.task === "classification" && sn);
  $("#assist-speciesnet").disabled = !(sn && sn.available);
  if (sn && !sn.available) $("#assist-speciesnet").checked = false;
  row.lastChild.textContent = sn && !sn.available ? " Name species with SpeciesNet (not installed on this server)" : " Name species with SpeciesNet";
  $("#assist-animal-row").hidden = !md || speciesnetOn();
  hideAssistConfirm();
}

async function onAssistToggle() {
  const on = $("#assist-enable").checked;
  $("#assist-body").hidden = !on;
  hideAssistConfirm();
  if (on) await refreshAssistSummary();
}

async function refreshAssistSummary() {
  if (!state.datasetId || $("#assist-block").hidden) return;
  state.assistSummary = (await api("GET", `/v1/datasets/${state.datasetId}/assist/summary`)).sources;
  renderAssistWarnings();
}

function assistSources() {
  const all = document.querySelector('input[name="assist-source"][value="all"]');
  return all.checked && !$("#assist-source-all-row").hidden ? ["upload", "studio"] : ["upload"];
}

// Image counts for the chosen sources: what assist would label, replace, and leave alone.
function assistPlan() {
  const sum = state.assistSummary;
  const overwrite = $("#assist-overwrite").checked;
  const plan = { images: 0, empty: 0, annotated: 0, excluded: 0, studio: 0, overwrite };
  if (!sum) return plan;
  for (const src of assistSources()) {
    const c = sum[src];
    plan.images += c.images;
    plan.empty += c.empty;
    plan.annotated += c.with_boxes + c.confirmed;
    plan.excluded += c.excluded;
    if (src === "studio") plan.studio = c.images;
  }
  plan.targets = plan.empty + (overwrite ? plan.annotated : 0);
  return plan;
}

function assistWarnings(plan) {
  const out = [];
  if (plan.annotated && wholeImage()) {
    out.push(plan.overwrite
      ? `${plural(plan.annotated, "image")} already ${plan.annotated === 1 ? "has a label" : "have labels"} or ${plan.annotated === 1 ? "is" : "are"} confirmed. "Replace existing labels" is ticked, so their labels will be replaced by the detector's and set back to pending review. Confirmed reviews are lost.`
      : `${plural(plan.annotated, "image")} already ${plan.annotated === 1 ? "has a label" : "have labels"} or ${plan.annotated === 1 ? "is" : "are"} confirmed. They are left alone. Check the new labels on other images before training.`);
  } else if (plan.annotated) {
    out.push(plan.overwrite
      ? `${plural(plan.annotated, "image")} already ${plan.annotated === 1 ? "has" : "have"} boxes or ${plan.annotated === 1 ? "is" : "are"} confirmed. "Replace existing boxes" is ticked, so their boxes will be replaced by the detector's and set back to pending review. Confirmed reviews are lost.`
      : `${plural(plan.annotated, "image")} already ${plan.annotated === 1 ? "has" : "have"} boxes or ${plan.annotated === 1 ? "is" : "are"} confirmed. They are left alone, but boxes pre-drawn on other images may use different labels or box conventions than your annotations. Check them before training.`);
  }
  if (plan.studio) {
    const things = assistWords().things;
    const fresh = wholeImage() ? "New labels" : "Pre-drawn boxes";
    out.push(`${plural(plan.studio, "Studio-imported image")} ${plan.studio === 1 ? "is" : "are"} included. ${fresh} may disagree with the review done in Sparrow Studio, and the next import from Studio replaces the ${things} on these images.`);
  }
  return out;
}

function renderAssistWarnings() {
  const sum = state.assistSummary;
  if (!sum) return;
  const studio = sum.studio.images;
  $("#assist-source-all-row").hidden = !studio;
  $("#assist-source-upload-text").textContent = `Only images I uploaded (${plural(sum.upload.images, "image")})`;
  $("#assist-source-all-text").textContent = `Uploaded and Studio-imported images (${plural(sum.upload.images + studio, "image")}, ${studio} from Studio)`;
  const plan = assistPlan();
  const nodes = assistWarnings(plan).map((w) => el("p", { class: "assist-warning", text: w }));
  if (studio && !plan.studio) {
    nodes.push(el("p", { class: "assist-note", text: `${plural(studio, "Studio-imported image")} ${studio === 1 ? "is" : "are"} left out. Choose "Uploaded and Studio-imported images" to include ${studio === 1 ? "it" : "them"}.` }));
  }
  $("#assist-warnings").replaceChildren(...nodes);
}

function hideAssistConfirm() {
  $("#assist-confirm").hidden = true;
  if (state.images.length && !$("#assist-model").disabled) $("#assist-btn").disabled = false;
}

async function openAssistConfirm() {
  if (!state.datasetId) throw new Error("Create or select a dataset first.");
  if (!$("#assist-model").value) throw new Error("Choose a detector first.");
  await refreshAssistSummary();
  const plan = assistPlan();
  if (!plan.targets) {
    throw new Error(plan.images
      ? (wholeImage()
        ? "No image to label: every chosen image already has a label, is confirmed or is excluded. Tick \"Replace existing labels\" to redo them."
        : "No image to pre-draw: every chosen image already has boxes, is confirmed or is excluded. Tick \"Replace existing boxes\" to redo them.")
      : "No image in the chosen source. Upload images first.");
  }
  const sel = $("#assist-model");
  const items = [
    `Detector: ${sel.options[sel.selectedIndex].text}${speciesnetOn() && sel.value === MEGADETECTOR ? ", species named by SpeciesNet" : ""}.`,
    `${plural(plan.targets, "image")} will get pending ${assistWords().things}${plan.studio ? `, including Studio-imported images` : ""}.`,
  ];
  const things = assistWords().things;
  if (plan.overwrite && plan.annotated) items.push(`${plural(plan.annotated, "image")} will lose ${plan.annotated === 1 ? "its" : "their"} current ${things}.`);
  else if (plan.annotated) items.push(`${plural(plan.annotated, "image")} with ${things} or confirmed ${plan.annotated === 1 ? "stays" : "stay"} unchanged.`);
  if (plan.excluded) items.push(`${plural(plan.excluded, "excluded image")} ${plan.excluded === 1 ? "is" : "are"} skipped.`);
  items.push(...assistWarnings(plan));
  $("#assist-confirm-list").replaceChildren(...items.map((t) => el("li", { text: t })));
  $("#assist-confirm").hidden = false;
  $("#assist-btn").disabled = true;
  $("#assist-confirm-btn").focus();
}

// FT-32: a detector trained on little data often scores every box below the default 0.25.
function belowConfHint(total, conf) {
  if (total.bestBelow == null || total.items >= total.processed / 2) return "";
  if (total.bestBelow < 0.01) return ` No box scored above ${conf}; the model found almost nothing, so it may need more training data.`;
  const best = total.bestBelow.toFixed(2);
  return ` On the images left without boxes, nothing scored above ${conf} (the best box scored ${best}). Lower "Minimum confidence" to about ${best} to pre-draw them, and review carefully: low-scoring boxes are often wrong.`;
}

async function runAssist() {
  hideAssistConfirm();
  if (!state.datasetId) throw new Error("Create or select a dataset first.");
  const choice = $("#assist-model").value;
  if (!choice) throw new Error("Choose a detector first.");
  const modelId = choice === MEGADETECTOR ? MEGADETECTOR : Number(choice);
  const classifier = modelId === MEGADETECTOR && speciesnetOn() ? "speciesnet" : null;
  const animalLabel = modelId === MEGADETECTOR && !classifier ? $("#assist-animal-label").value.trim() : "";
  const sources = assistSources();
  const names = state.images.filter((i) => sources.includes(i.source || "upload")).map((i) => i.name);
  if (!names.length) throw new Error("No image in the chosen source.");
  const overwrite = $("#assist-overwrite").checked;
  const conf = assistConf();
  const chunks = [];
  for (let i = 0; i < names.length; i += ASSIST_MAX_ITEMS) chunks.push(names.slice(i, i + ASSIST_MAX_ITEMS));
  const total = { processed: 0, items: 0, boxes: 0, skipped: 0, unknown: new Set(), bestBelow: null };
  const status = $("#assist-status");
  const whole = wholeImage();
  const words = assistWords();
  const summary = () => (whole
    ? `Labelled ${plural(total.items, "image")} for review; ${plural(total.skipped, "image")} left unchanged.`
    : `Added ${boxCount(total.boxes)} to ${plural(total.items, "image")} for review; ${plural(total.skipped, "image")} left unchanged.`)
    + (total.unknown.size ? ` The model's labels ${[...total.unknown].join(", ")} are not used in this dataset and were left out.` : "")
    + belowConfHint(total, conf);
  let done = 0;
  try {
    await withBusy(`${words.busy}…`, $("#assist-btn"), async () => {
      for (const images of chunks) {
        const msg = `Batch ${done + 1} of ${chunks.length} (${plural(images.length, "image")})…`;
        status.textContent = msg;
        $("#busy-text").textContent = `${words.busy}: ${msg}`;
        const res = await api("POST", `/v1/datasets/${state.datasetId}/assist`, { model_id: modelId, images, sources, conf, overwrite, ...(animalLabel ? { animal_label: animalLabel } : {}), ...(classifier ? { classifier } : {}) });
        total.processed += res.processed;
        total.items += res.items_written;
        total.boxes += res.boxes;
        total.skipped += (res.skipped || []).length;
        for (const l of res.unknown_labels || []) total.unknown.add(l);
        if (res.best_below_conf != null) total.bestBelow = Math.max(total.bestBelow ?? 0, res.best_below_conf);
        done += 1;
      }
    });
  } catch (err) {
    status.textContent = `Stopped after batch ${done} of ${chunks.length}. ${summary()}`;
    await loadDataset();
    throw err;
  }
  status.textContent = summary();
  toast(whole ? `Labelled ${plural(total.items, "image")}` : `Pre-drew ${boxCount(total.boxes)}`);
  await loadDataset();
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function renderWorkspacePanel() {
  const ws = state.workspaces.find((w) => w.id === state.workspaceId);
  $("#current-workspace").textContent = ws ? ws.name : "No workspace yet";
  $("#current-workspace").classList.toggle("empty", !ws);
  $("#current-workspace-meta").textContent = ws ? plural(state.datasets.length, "dataset") : "Create one to start.";
  $("#workspace-select").closest("label").hidden = state.workspaces.length < 2;
  $("#delete-workspace").hidden = !ws;
  if (!state.workspaces.length) $("#workspace-panel").open = true; // nothing to fold away yet
  $("#show-new-dataset").disabled = !ws;
}

function renderDatasetPanel() {
  const ds = state.dataset;
  $("#current-dataset").textContent = ds ? ds.name : "No dataset yet";
  $("#current-dataset").classList.toggle("empty", !ds);
  const task = ds ? ds.task.charAt(0).toUpperCase() + ds.task.slice(1) : "";
  const mode = ds && ds.task === "classification" ? (ds.whole_image ? " · trains on whole images" : " · trains on box crops") : "";
  $("#current-dataset-meta").textContent = ds
    ? `${task}${mode} · ${plural(state.images.length, "image")}`
    : state.workspaceId ? "Create one in this workspace." : "Create a workspace first.";
  $("#dataset-select").closest("label").hidden = state.datasets.length < 2;
  $("#delete-dataset").hidden = !ds;
  $("#whole-image-row").hidden = !(ds && ds.task === "classification");
  $("#dataset-whole-image").checked = Boolean(ds && ds.whole_image);
}

async function onWholeImageToggle() {
  const box = $("#dataset-whole-image");
  const ds = state.dataset;
  if (!ds) return;
  const on = box.checked;
  const boxed = state.items.filter((i) => (i.annotations || []).some((a) => a.bbox)).length;
  const msg = on
    ? `Train "${ds.name}" on whole images?${boxed ? ` ${plural(boxed, "image")} ${boxed === 1 ? "has" : "have"} boxes: they are removed and each image keeps its box labels as image labels. This cannot be undone.` : ""} Snapshots taken from now on use the whole photo.`
    : `Train "${ds.name}" on box crops? Images keep their labels but need boxes before they count: draw them or run assisted labelling. Uploaded images without a box are left out of new snapshots.`;
  if (!confirm(msg)) { box.checked = !on; return; }
  const res = await api("PATCH", `/v1/datasets/${ds.id}`, { whole_image: on });
  toast(on ? `Whole-image training on${res.boxes_removed ? `; removed ${boxCount(res.boxes_removed)}` : ""}` : "Training on box crops");
  await loadDatasets(ds.id);
}

function renderNewDatasetOptions() {
  $("#new-dataset-whole-image-row").hidden = $("#new-dataset-task").value !== "classification";
}

function toggleCreate(kind, open) {
  const form = $(`#new-${kind}-form`);
  form.hidden = !open;
  $(`#show-new-${kind}`).hidden = open;
  if (open) {
    $(`#new-${kind}`).focus();
  } else {
    $(`#new-${kind}`).value = "";
  }
}

async function createWorkspace() {
  const name = $("#new-workspace").value.trim();
  if (!name) throw new Error("Enter a workspace name.");
  const ws = await api("POST", "/v1/workspaces", { name });
  toggleCreate("workspace", false);
  state.datasetId = null;
  toast(`Workspace "${ws.name}" created`);
  await loadWorkspaces(ws.id);
}

async function createDataset() {
  if (!state.workspaceId) throw new Error("Create or select a workspace first.");
  const name = $("#new-dataset").value.trim();
  if (!name) throw new Error("Enter a dataset name.");
  const ds = await api("POST", `/v1/workspaces/${state.workspaceId}/datasets`, {
    name,
    task: $("#new-dataset-task").value,
    whole_image: $("#new-dataset-task").value === "classification" && $("#new-dataset-whole-image").checked,
  });
  toggleCreate("dataset", false);
  toast(`Dataset "${ds.name}" created`);
  await loadDatasets(ds.id);
}

// Deletion is permanent, so the user types the exact name; anything else cancels.
function usePlace(u) {
  return u.consumer === "studio" ? `installed in Studio project ${u.project}` : `active local model (${u.project})`;
}

// Lists everything a delete removes, so the typed-name confirmation is an informed one.
function deletionSummary(p) {
  const lines = [];
  if (p.kind === "workspace") lines.push(`- ${plural(p.datasets, "dataset")}${p.dataset_names.length ? `: ${p.dataset_names.join(", ")}` : ""}`);
  lines.push(`- ${plural(p.images, "uploaded image")} and ${plural(p.items, "labelled item")}`);
  if (p.jobs) lines.push(`- ${plural(p.jobs, "training job")} and their event logs`);
  if (p.models.length) {
    lines.push(`- ${plural(p.models.length, "trained model")} and their files:`);
    for (const m of p.models) {
      const where = m.in_use.length ? ` (IN USE: ${m.in_use.map(usePlace).join("; ")})` : "";
      lines.push(`    ${m.name || `Model #${m.id}`}${where}`);
    }
    if (p.models.some((m) => m.in_use.some((u) => u.consumer === "studio"))) {
      lines.push("Sparrow Studio keeps the copy it already installed; delete it in Studio if it is no longer needed.");
    }
  }
  return lines.join("\n");
}

async function confirmDeletion(kind, name, path) {
  const preview = await api("GET", `${path}/deletion`);
  if (preview.active_jobs.length) {
    throw new Error(
      `Job(s) ${preview.active_jobs.join(", ")} of this ${kind} are queued or running. Cancel them first, then delete the ${kind}.`
    );
  }
  const typed = window.prompt(
    `Permanently delete ${kind} "${name}"? This removes:\n${deletionSummary(preview)}\n\nThis cannot be undone. Type the ${kind} name to confirm:`
  );
  if (typed === null) return null;
  if (typed.trim() !== name) {
    toast(`Name did not match; ${kind} "${name}" was not deleted.`);
    return null;
  }
  return preview;
}

function deletedText(res) {
  const d = res.deleted;
  const extra = d.models ? `, ${plural(d.models, "model")}, ${plural(d.jobs, "job")}` : "";
  return `${plural(d.datasets, "dataset")}, ${plural(d.images, "image")}${extra}`;
}

async function deleteWorkspace() {
  const ws = state.workspaces.find((w) => w.id === state.workspaceId);
  if (!ws) throw new Error("Select a workspace first.");
  const path = `/v1/workspaces/${ws.id}`;
  if (!(await confirmDeletion("workspace", ws.name, path))) return;
  const res = await api("DELETE", `${path}?include_models=true`);
  state.workspaceId = null;
  state.datasetId = null;
  await loadWorkspaces();
  if (res.deleted.models) await refreshAfterModelDelete();
  toast(`Workspace "${ws.name}" deleted (${deletedText(res)})`);
}

async function deleteDataset() {
  const ds = state.datasets.find((d) => d.id === state.datasetId);
  if (!ds) throw new Error("Select a dataset first.");
  const path = `/v1/datasets/${ds.id}`;
  if (!(await confirmDeletion("dataset", ds.name, path))) return;
  const res = await api("DELETE", `${path}?include_models=true`);
  state.datasetId = null;
  await loadDatasets();
  if (res.deleted.models) await refreshAfterModelDelete();
  toast(`Dataset "${ds.name}" deleted (${deletedText(res)})`);
}

async function refreshAfterModelDelete() {
  await Promise.all([loadJobs(), loadModels()].map((p) => p.catch(() => {})));
}

function pickFile(inputId) {
  if (!state.datasetId) throw new Error("Create or select a dataset first.");
  $(inputId).click();
}

async function uploadImages() {
  if (!state.datasetId) throw new Error("Create or select a dataset first.");
  const files = Array.from($("#upload-files").files || []);
  if (!files.length) return;
  const bar = $("#upload-progress");
  const status = $("#upload-status");
  bar.max = files.length;
  bar.value = 0;
  bar.hidden = false;
  let uploaded = 0;
  let skipped = 0;
  const show = () => {
    status.textContent = `${uploaded} / ${files.length} uploaded${skipped ? `, ${skipped} skipped` : ""}`;
    bar.value = uploaded + skipped;
  };
  show();
  try {
    await uploadEach(files, show, () => { uploaded += 1; }, () => { skipped += 1; });
  } finally {
    $("#upload-files").value = "";
    show();
    await loadDataset();
  }
  toast(`Uploaded ${uploaded} image(s).`);
}

async function uploadEach(files, show, onUploaded, onSkipped) {
  for (const f of files) {
    const path = `/v1/datasets/${state.datasetId}/images/${encodeURIComponent(f.name)}`;
    const opts = { contentType: f.type || "application/octet-stream" };
    try {
      await api("PUT", path, f, opts);
    } catch (err) {
      // 409: the dataset already has a different image under this name.
      if (!(err instanceof ApiError && err.status === 409)) throw err;
      if (!confirm(`The dataset already has a different image named "${f.name}". Replace it with this file?`)) {
        onSkipped();
        show();
        continue;
      }
      await api("PUT", `${path}?replace=true`, f, opts);
    }
    onUploaded();
    show();
  }
}

// ------------------------------------------------------------------ label formats

// Import/export formats. `endpoint` is the /import/<x> and /export/<x> path segment; Sparrow JSON
// imports through POST /annotations and has no export. VOC carries boxes only.
const LABEL_FORMATS = {
  camtrapdp: {
    name: "Camera Trap DP",
    endpoint: "camtrapdp",
    accept: ".zip,application/zip",
    help: "A Camera Trap DP 1.0 package: a zip with datapackage.json, media.csv and observations.csv. Media rows are matched to images by file name; species come from scientificName, and blank observations mark empty images.",
    warning: "Camera Trap DP is an exchange format. Packages may not carry camera locations or capture times: exports from this service leave them out, and imports keep only file names, species and boxes. Inside this service labels are stored in Sparrow's own format.",
  },
  coco: {
    name: "MS COCO",
    endpoint: "coco",
    accept: ".json,.zip,application/json,application/zip",
    help: "A COCO JSON document (images, categories, annotations), or a zip holding one. Images are matched by file_name; an image with no annotations, or only an \"empty\"/\"blank\" category, is an empty image.",
  },
  voc: {
    name: "Pascal VOC",
    endpoint: "voc",
    accept: ".zip,application/zip",
    detectionOnly: true,
    help: "A zip of Pascal VOC XML files, one per image. Boxes only, so it is offered for detection datasets.",
  },
  sparrow: {
    name: "Sparrow JSON",
    endpoint: null,
    accept: ".json,application/json",
    help: 'This service\'s own {"items": [...]} document. Import only.',
  },
};

function labelFormat() {
  const key = $("#annotation-format").value;
  return LABEL_FORMATS[key] ? key : "camtrapdp";
}

function renderLabelFormat() {
  const sel = $("#annotation-format");
  const detection = isDetection();
  for (const opt of sel.options) {
    const off = Boolean(LABEL_FORMATS[opt.value].detectionOnly && !detection);
    opt.hidden = off;
    opt.disabled = off;
  }
  if (sel.selectedOptions[0] && sel.selectedOptions[0].disabled) sel.value = "camtrapdp";
  const key = labelFormat();
  const fmt = LABEL_FORMATS[key];
  $("#annotations-file").accept = fmt.accept;
  $("#annotation-format-help").textContent = fmt.help;
  const warn = $("#annotation-format-warning");
  warn.textContent = fmt.warning || "";
  warn.hidden = !fmt.warning;
  $("#annotations-overwrite-label").hidden = !fmt.endpoint;
  const exp = $("#export-annotations");
  exp.hidden = !(fmt.endpoint && state.datasetId);
  if (!exp.hidden) {
    exp.href = `/v1/datasets/${state.datasetId}/export/${fmt.endpoint}`;
    exp.title = `Download this dataset's labels as ${fmt.name}`;
  }
  renderSnapshotExport();
}

function renderSnapshotExport() {
  const link = $("#snapshot-export");
  const snap = state.snapshot;
  const fmt = LABEL_FORMATS[labelFormat()];
  link.hidden = !(snap && fmt.endpoint);
  if (link.hidden) return;
  link.href = `/v1/snapshots/${snap.id}/export/${fmt.endpoint}`;
  link.textContent = `Download snapshot as ${fmt.name}`;
}

async function importAnnotations() {
  if (!state.datasetId) throw new Error("Create or select a dataset first.");
  const file = $("#annotations-file").files[0];
  if (!file) return;
  const status = $("#annotations-status");
  $("#annotations-file").value = "";
  const key = labelFormat();
  const fmt = LABEL_FORMATS[key];
  const isJson = /\.json$/i.test(file.name);
  if ((key === "camtrapdp" || key === "voc") && isJson) {
    throw new Error(`${file.name} is a JSON file, but ${fmt.name} needs a .zip. Pick MS COCO or Sparrow JSON under Label format.`);
  }
  if (key === "sparrow" && !isJson) {
    throw new Error(`${file.name} is not a .json file. Pick the format that matches it under Label format.`);
  }
  status.textContent = `Importing ${file.name} as ${fmt.name}…`;
  let res;
  let doc;
  try {
    if (key === "sparrow") {
      try { doc = JSON.parse(await file.text()); } catch (e) { throw new Error(`Annotations file is not valid JSON: ${e.message}`); }
      res = await api("POST", `/v1/datasets/${state.datasetId}/annotations`, doc);
    } else {
      const overwrite = $("#annotations-overwrite").checked;
      res = await api("POST", `/v1/datasets/${state.datasetId}/import/${fmt.endpoint}?overwrite=${overwrite}`, file,
        { contentType: file.type || "application/octet-stream" });
    }
  } catch (err) {
    const d = err instanceof ApiError && err.status === 422 ? err.detail : null;
    if (key === "sparrow" && d && d.reason === "labels_without_boxes" && Array.isArray(d.unboxed_images)) {
      status.textContent = `Import of ${file.name} stopped: labels without boxes.`;
      if (await moveUnboxedUploads(d.unboxed_images, file, doc)) return;
    }
    status.textContent = `Import of ${file.name} failed; nothing was changed.`;
    throw err;
  }
  if (key === "sparrow") {
    status.textContent = `Imported labels for ${plural(res.imported, "image")} from ${file.name}.`;
    toast(`Imported ${res.imported} item(s)`);
    await loadDataset();
    return;
  }
  const unmatched = Array.isArray(res.unmatched_files) ? res.unmatched_files : [];
  const skipped = Array.isArray(res.skipped) ? res.skipped : [];
  const what = isDetection() ? boxCount(res.boxes) : "labels";
  const parts = [`Imported ${what} for ${plural(res.matched, "image")} from ${file.name}.`];
  if (skipped.length) {
    parts.push(`${skipped.length} entr${skipped.length === 1 ? "y" : "ies"} skipped: ${skipped.slice(0, 3).map((x) => `${x.file} (${x.reason})`).join("; ")}${skipped.length > 3 ? "; …" : ""}.`);
  }
  if (unmatched.length) parts.push(`${unmatched.length} file(s) matched no image in this dataset.`);
  if (res.new_labels && res.new_labels.length) parts.push(`New labels: ${res.new_labels.join(", ")}.`);
  if (key === "camtrapdp") parts.push("Camera locations and capture times in the package, if any, were not imported.");
  status.textContent = parts.join(" ");
  toast(`Imported labels for ${plural(res.matched, "image")}`);
  await loadDataset();
  const unboxed = Array.isArray(res.unboxed_images) ? res.unboxed_images : [];
  if (unboxed.length) await moveUnboxedUploads(unboxed, file, null, fmt);
}

// Species-only labels were uploaded for images in a detection dataset. Offer to move
// those images to the workspace's classification dataset (created if missing) and
// import their labels there. Returns true when the move and import were done.
async function moveUnboxedUploads(names, file, doc, fmt = null) {
  const source = state.datasets.find((d) => d.id === state.datasetId);
  const ws = state.workspaces.find((w) => w.id === state.workspaceId);
  const target = state.datasets.find((d) => d.task === "classification");
  const targetName = target ? target.name : `${ws ? ws.name : "Workspace"} (classification)`;
  const n = plural(names.length, "image");
  const pick = await chooseDialog(
    "These labels have no boxes",
    `${n} in ${source ? source.name : "this dataset"} only have species labels, without boxes. ` +
      `A detector needs boxes; a species classifier does not.`,
    [{
      value: "move",
      label: `Move the ${n} and their labels to ${targetName}`,
      detail: target
        ? "Train a species classifier with them. Images that already have labels here stay put."
        : "That classification dataset will be created. Images that already have labels here stay put.",
    }],
    "Keep them here",
  );
  if (pick !== "move") return false;
  const status = $("#annotations-status");
  const dest = target || await api("POST", `/v1/workspaces/${state.workspaceId}/datasets`, { name: targetName, task: "classification" });
  const moved = await api("POST", `/v1/datasets/${state.datasetId}/images/move`, { to_dataset_id: dest.id, names });
  await loadDatasets(dest.id);
  let res;
  if (doc) {
    const wanted = new Set(names);
    res = await api("POST", `/v1/datasets/${dest.id}/annotations`, { ...doc, items: doc.items.filter((it) => wanted.has(it.image)) });
  } else {
    const overwrite = $("#annotations-overwrite").checked;
    res = await api("POST", `/v1/datasets/${dest.id}/import/${fmt.endpoint}?overwrite=${overwrite}`, file,
      { contentType: file.type || "application/octet-stream" });
  }
  const labelled = doc ? res.imported : res.matched;
  let note = "";
  const rest = doc ? doc.items.filter((it) => !names.includes(it.image)) : [];
  if (rest.length && source) {
    // The file's entries with boxes were held back by the all-or-nothing import; bring them in now.
    try {
      const r = await api("POST", `/v1/datasets/${source.id}/annotations`, { ...doc, items: rest });
      note = ` Labels with boxes for ${plural(r.imported, "image")} went to ${source.name}.`;
    } catch (err) {
      note = ` The other ${rest.length} entr${rest.length === 1 ? "y" : "ies"} could not be imported into ${source.name}: ${err.message}`;
    }
  }
  status.textContent = `Moved ${plural(moved.moved + moved.already_there, "image")} to ${dest.name} and imported labels for ${plural(labelled, "image")} from ${file.name}.${note}`;
  toast(`Moved ${plural(moved.moved + moved.already_there, "image")} to ${dest.name}`);
  return true;
}

function checkStudioProject(project) {
  // Studio project ids are integers when the service says so; catch typos before the call.
  if (state.studio && state.studio.project_ids === "numeric" && !/^[0-9]+$/.test(project)) {
    throw new Error(`Studio project must be a numeric project id (for example 12), got "${project}".`);
  }
}

const STUDIO_PROJECT_PICKERS = ["#studio-project", "#studio-deploy-project"];

// Sparrow Studio opens this app as /?studio_project=<id>[&load=1]. The project is
// selected once its list arrives; load=1 also imports its data (once: it is dropped
// from the address so a reload does not import again).
const launch = (() => {
  const q = new URLSearchParams(location.search);
  return { project: (q.get("studio_project") || "").trim(), load: q.get("load") === "1" };
})();

function dropLaunchLoad() {
  const url = new URL(location.href);
  if (!url.searchParams.has("load")) return;
  url.searchParams.delete("load");
  history.replaceState(null, "", url);
}

function fillStudioPickers(options) {
  for (const sel of STUDIO_PROJECT_PICKERS.map((id) => $(id))) {
    const prev = sel.value;
    sel.replaceChildren(...options.map(([value, text]) => el("option", { value, text })));
    if (prev && options.some(([value]) => value === prev)) sel.value = prev;
  }
}

async function loadStudioProjects() {
  fillStudioPickers([["", "Loading projects…"]]);
  let res;
  try {
    res = await api("GET", "/v1/connectors/studio/projects");
  } catch (err) {
    state.studioProjects = [];
    fillStudioPickers([["", "Could not load Studio projects"]]);
    renderStudioScope();
    throw err;
  }
  state.studioProjects = res.projects;
  if (!res.projects.length) {
    fillStudioPickers([["", "No projects in Studio"]]);
    renderStudioScope();
    return;
  }
  fillStudioPickers([
    ["", "Choose a Studio project"],
    ...res.projects.map((p) => [p.id, `${p.name} (id ${p.id})`]),
  ]);
  if (launch.project && res.projects.some((p) => p.id === launch.project)) {
    $("#studio-project").value = launch.project;
  }
  renderStudioScope();
  if (state.modelId) renderModel();
}

const STUDIO_SCOPED_CONTROLS = ["#studio-import-btn", "#studio-reviewed-only"];

function renderStudioScope() {
  const project = $("#studio-project").value;
  const name = $("#current-studio-project");
  name.textContent = project ? studioProjectName(project) : "None selected";
  name.classList.toggle("empty", !project);
  for (const id of STUDIO_SCOPED_CONTROLS) $(id).disabled = !project;
  $("#studio-hint").hidden = Boolean(project);
  $("#studio-body").hidden = !project;
  // Activation in Studio usually targets the same project; prefill it if still unset.
  if (project && !$("#studio-deploy-project").value
      && [...$("#studio-deploy-project").options].some((o) => o.value === project)) {
    $("#studio-deploy-project").value = project;
    loadStudioDetectors();
  }
  refreshStudioStatus();
  if (project !== state.studioModelsProject) {
    state.studioModelsProject = project;
    fillStudioModels([]);
    $("#studio-models-status").textContent = "";
    if (project) listStudioModels().catch((err) => showError(err.message));
  }
}

function studioProjectName(id) {
  const p = (state.studioProjects || []).find((x) => x.id === String(id));
  return p ? p.name : id;
}

function selectedStudioProject(selector = "#studio-project") {
  const project = $(selector).value.trim();
  if (!project) throw new Error("Choose a Studio project first.");
  checkStudioProject(project);
  return project;
}

async function studioImport() {
  if (!state.datasetId) throw new Error("Create or select a dataset first.");
  const project = selectedStudioProject();
  const res = await api("POST", "/v1/connectors/studio/import", {
    project, dataset_id: state.datasetId, reviewed_only: $("#studio-reviewed-only").checked,
  });
  toast(`Imported ${res.imported_items} item(s) from ${res.imported_images} Studio image(s)`);
  await loadDataset();
  refreshStudioStatus();
}

// "All data imported" note: shown when the selected dataset already holds everything
// an import of the selected project would bring. Stale answers are dropped.
let studioStatusSeq = 0;
async function refreshStudioStatus() {
  const note = $("#studio-uptodate");
  const seq = ++studioStatusSeq;
  const project = $("#studio-project").value;
  const datasetId = state.datasetId;
  if (!project || !datasetId || !state.studio || !state.studio.configured) {
    note.hidden = true;
    return;
  }
  const reviewed = $("#studio-reviewed-only").checked;
  let res;
  try {
    const q = new URLSearchParams({ project, dataset_id: datasetId, reviewed_only: reviewed });
    res = await api("GET", `/v1/connectors/studio/import-status?${q}`);
  } catch (_) {
    res = null;  // the status note is advisory; the import button reports real errors
  }
  if (seq !== studioStatusSeq) return;
  note.hidden = !(res && res.up_to_date);
  if (res && res.up_to_date) {
    const n = res.studio_images;
    note.textContent = `All ${n} ${reviewed ? "reviewed " : ""}Studio image${n === 1 ? "" : "s"} imported — up to date.`;
  }
}

// Opened from Studio with load=1: load the project into its own workspace and select it.
async function loadLaunchProject() {
  const project = launch.project;
  launch.load = false;
  dropLaunchLoad();
  if (!project || !state.studio || !state.studio.configured) return;
  const res = await api("POST", "/v1/connectors/studio/load-project", { project, reviewed_only: true });
  const det = res.datasets.find((d) => d.task === "detection");
  const cls = res.datasets.find((d) => d.task === "classification");
  // Show the Studio workspace (detection dataset first) before asking, so the page
  // behind the question is the project that was just loaded.
  state.datasetId = (det || res.datasets[0] || {}).id ?? null;
  await loadWorkspaces(res.workspace.id);
  if (det && cls) {
    const from = (d) => `${plural(d.imported_items || 0, "labelled image")} from Studio`;
    const pick = await chooseDialog(
      "What do you want to train?",
      `Studio project ${res.workspace.name} is loaded. Pick the dataset to work on; you can switch later under Dataset.`,
      [
        { value: det.id, label: "A detector that draws boxes around animals", detail: `${det.name} · ${from(det)}. Upload labels with boxes here.` },
        { value: cls.id, label: "A species classifier that names the animal in each image", detail: `${cls.name} · ${from(cls)}. Upload species-only labels here.` },
      ],
      "Decide later",
    );
    if (pick === cls.id) await loadDatasets(cls.id);
  }
  const total = res.datasets.reduce((n, d) => n + (d.imported_items || 0), 0);
  toast(`Loaded Studio project ${res.workspace.name}: ${total} item(s) into workspace ${res.workspace.name}`);
  refreshStudioStatus();
}

// The chosen base model: { row, name, projectName } (projectName is null for a local package) or null.
function renderBaseModel() {
  renderStartWeights();
  const box = $("#base-model-info");
  const chosen = state.baseModel;
  $("#studio-model-clear").disabled = !chosen || !baseModelAllowed();
  $("#local-base-model-clear").disabled = !chosen;
  const local = Boolean(chosen && chosen.row.origin === "local");
  $("#local-base-model-select").value = local ? String(chosen.row.id) : "";
  const parent = state.parentModel;
  if (!chosen && parent && parent.pipeline === $("#pipeline-select").value) {
    box.replaceChildren(
      el("strong", { text: `Retraining ${modelTitle(parent)}` }),
      el("span", { text: `${pipelineName(parent.pipeline)} · model #${parent.id} from job ${parent.job_id} (snapshot ${parent.snapshot_id})` }),
      el("span", { class: "muted", text: "The finished model is not changed; training makes a new version that records it as the parent." }),
    );
    return;
  }
  if (!chosen) {
    const text = $("#pipeline-select").value === "yolo_detector"
      ? "None — training starts from the starting weights above."
      : "None — training starts from ImageNet weights.";
    box.replaceChildren(el("span", { class: "muted", text }));
    return;
  }
  const { row } = chosen;
  const labels = row.labels || [];
  const shown = labels.slice(0, 8).join(", ") + (labels.length > 8 ? `, +${labels.length - 8} more` : "");
  const origin = local
    ? `Local package (sha256 ${String(row.sha256).slice(0, 12)}) · base model #${row.id}`
    : `From Studio project ${chosen.projectName} (Studio model ${row.studio_import_id}) · base model #${row.id}`;
  box.replaceChildren(
    el("strong", { text: chosen.name }),
    el("span", { text: `${row.model_arch} · ${labels.length} class${labels.length === 1 ? "" : "es"}: ${shown}` }),
    el("span", { class: "muted", text: origin }),
  );
}

function clearBaseModel() {
  state.baseModel = null;
  renderBaseModel();
}

// --------------------------------------------- retrain a finished model (parent reuse)

const FINETUNE_YES = "Can be fine-tuned";
const FINETUNE_NO = "Fine-tune not supported";

// A finished conventional or DINOv3 classifier can be the starting point of a new job.
function reusable(m) {
  return Boolean(m && CLASSIFIER_PIPELINES.has(m.pipeline) && m.finetune_tag !== FINETUNE_NO);
}

// The architecture a finished model was trained with, when its record names it.
function parentArch(m) {
  const md = (m && m.metadata) || {};
  return md.model_arch || md.model || null;
}

// Choosing a base model drops a chosen parent, and the other way round: a job starts from one.
function dropParentModel() {
  if (!state.parentModel) return;
  state.parentModel = null;
  state.reusePreview = null;
  renderParentModels();
}

async function loadParentModels() {
  try {
    state.parentCandidates = (await workspaceList("/v1/models", "models")).filter(reusable).sort((a, b) => b.id - a.id);
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) throw err;
    state.parentCandidates = [];
  }
  renderParentModels();
}

function renderParentModels() {
  const pipeline = $("#pipeline-select").value;
  $("#parent-model-block").hidden = !CLASSIFIER_PIPELINES.has(pipeline);
  if (state.parentModel && state.parentModel.pipeline !== pipeline) {
    state.parentModel = null;
    state.reusePreview = null;
  }
  const models = (state.parentCandidates || []).filter((m) => m.pipeline === pipeline);
  const sel = $("#parent-model-select");
  sel.replaceChildren(
    el("option", { value: "", text: models.length ? "No finished model selected" : `No finished ${pipelineName(pipeline)} model yet` }),
    ...models.map((m) => el("option", {
      value: String(m.id),
      text: `${modelTitle(m)} · job ${m.job_id} · snapshot ${m.snapshot_id}`,
    })),
  );
  sel.value = state.parentModel ? String(state.parentModel.id) : "";
  sel.disabled = !models.length;
  renderReusePreview();
}

async function chooseParentModel() {
  const id = Number($("#parent-model-select").value);
  state.parentModel = (state.parentCandidates || []).find((m) => m.id === id) || null;
  state.reusePreview = null;
  if (state.parentModel) state.baseModel = null;
  renderBaseModel();
  await previewReuse();
}

// GET /v1/models/{id}/reuse?snapshot_id=N: what of the finished model carries over to this snapshot.
async function previewReuse() {
  const parent = state.parentModel;
  const snap = state.snapshot;
  if (!parent || !snap) {
    state.reusePreview = null;
    renderReusePreview();
    return;
  }
  const prev = state.reusePreview;
  if (prev && prev.modelId === parent.id && prev.snapshotId === snap.id) {
    renderReusePreview();
    return;
  }
  state.reusePreview = { modelId: parent.id, snapshotId: snap.id, pending: true };
  renderReusePreview();
  let result;
  try {
    result = await api("GET", `/v1/models/${parent.id}/reuse?snapshot_id=${snap.id}`);
  } catch (err) {
    if (!(err instanceof ApiError) || err.status === 401) throw err;
    result = err.status === 404 && (!err.detail || err.detail === "Not Found")
      ? { unavailable: true }
      : { usable: false, reason: detailText(err.detail) || err.message };
  }
  if (state.parentModel !== parent || state.snapshot !== snap) return;  // the choice changed meanwhile
  state.reusePreview = { ...result, modelId: parent.id, snapshotId: snap.id };
  renderReusePreview();
}

function renderReusePreview() {
  const box = $("#reuse-preview");
  const parent = state.parentModel;
  const p = state.reusePreview;
  box.className = "reuse-preview";
  if (!parent) {
    box.hidden = true;
    box.replaceChildren();
    return;
  }
  box.hidden = false;
  const line = (text, cls) => el("p", { class: cls || "", text });
  if (!state.snapshot) {
    box.replaceChildren(line("Create a snapshot first. The preview compares the finished model with that snapshot."));
    return;
  }
  if (!p || p.pending) {
    box.replaceChildren(line("Checking what carries over to this snapshot…", "muted"));
    return;
  }
  if (p.unavailable) {
    box.replaceChildren(line("This server cannot preview the reuse. Training still checks the model and refuses it with a reason if it does not fit."));
    return;
  }
  const t = p.transfers || {};
  box.classList.add(p.usable ? "reuse-ok" : "reuse-bad");
  const nodes = [line(p.usable
    ? `${modelTitle(parent)} can be the starting point for snapshot ${state.snapshot.id}.`
    : `${modelTitle(parent)} cannot be retrained on snapshot ${state.snapshot.id}.`, "reuse-verdict")];
  if (p.usable && p.encoder === "same_frozen") {
    const shared = Number(p.shared_labels || 0);
    const total = Number(p.snapshot_labels || 0);
    nodes.push(el("ul", {},
      el("li", { text: "DINOv3 encoder: the same frozen encoder; it is never trained, so nothing is lost or changed" }),
      el("li", { text: shared
        ? `Linear head: starts from the finished model's weights for ${shared} of ${total} ${total === 1 ? "class" : "classes"}${total > shared ? `; the other ${total - shared} start from zero` : ""}`
        : "Linear head: no classes in common, so it starts from zero" }),
    ));
  } else if (p.usable) {
    nodes.push(el("ul", {},
      el("li", { text: `Feature extractor (backbone): ${t.backbone ? "carried over" : "not carried over"}` }),
      el("li", { text: `Class layer (head): ${t.head ? "carried over, same classes" : "trained from scratch"}` }),
    ));
    if (t.backbone && !t.head) nodes.push(line("The classes in this snapshot differ from the finished model, so only the feature extractor is reused. Expect to need about as much labelled data as for a new class layer."));
  }
  if (p.reason) nodes.push(line(p.reason, p.usable ? "muted" : "reuse-reason"));
  nodes.push(line("The architecture (or DINOv3 encoder) comes from the finished model and cannot be changed for this job. The finished model itself is never modified.", "muted"));
  box.replaceChildren(...nodes);
}

// Models page → "Use as base for new training": open the Data page with this model as the parent.
async function reuseModelFromDetail() {
  const m = state.models.find((x) => x.id === state.modelId);
  if (!reusable(m)) throw new Error("Only a finished image classifier or DINOv3 model can be the starting point of new training.");
  await showPage("data");
  const sel = $("#pipeline-select");
  if (![...sel.options].some((o) => o.value === m.pipeline && !o.disabled)) {
    throw new Error(`Open a classification dataset first, then choose ${modelTitle(m)} again on the Models page.`);
  }
  if (sel.value !== m.pipeline) {
    sel.value = m.pipeline;
    sel.dispatchEvent(new Event("change"));
  }
  if (!(state.parentCandidates || []).some((x) => x.id === m.id)) state.parentCandidates = [m, ...(state.parentCandidates || [])];
  state.parentModel = state.parentCandidates.find((x) => x.id === m.id);
  state.reusePreview = null;
  state.baseModel = null;
  renderParentModels();
  renderBaseModel();
  $("#parent-model-block").scrollIntoView({ block: "center" });
  toast(`${modelTitle(m)} will be the starting point for the next ${pipelineName(m.pipeline)} job`);
  await previewReuse();
}

function localBaseModelName(row) {
  return `Local ${row.model_arch} package #${row.id}`;
}

// Local packages already uploaded to this service, for the picker under the base model.
async function loadLocalBaseModels() {
  const res = await api("GET", "/v1/base-models");
  state.localBaseModels = (res.base_models || []).filter(
    (r) => r.origin === "local" && r.pipeline === LOCAL_BASE_PIPELINE,
  );
  const sel = $("#local-base-model-select");
  sel.replaceChildren(
    el("option", { value: "", text: "No uploaded package selected" }),
    ...state.localBaseModels.map((r) => el("option", {
      value: String(r.id),
      text: `${localBaseModelName(r)} · ${(r.labels || []).length} classes · ${String(r.sha256).slice(0, 12)}`,
    })),
  );
  sel.disabled = !state.localBaseModels.length;
  renderBaseModel();
}

function chooseLocalBaseModel() {
  const id = Number($("#local-base-model-select").value);
  const row = (state.localBaseModels || []).find((r) => r.id === id);
  state.baseModel = row ? { row, name: localBaseModelName(row), projectName: null } : null;
  if (row) dropParentModel();
  renderBaseModel();
}

async function uploadLocalBaseModel() {
  if ($("#pipeline-select").value !== LOCAL_BASE_PIPELINE) {
    throw new Error("A local classifier package is a base model for the image classifier pipeline only.");
  }
  const input = $("#local-base-model-file");
  const file = input.files && input.files[0];
  if (!file) throw new Error("Choose a classifier package (.zip) first.");
  const form = new FormData();
  form.append("file", file, file.name);
  const btn = $("#local-base-model-upload");
  const status = $("#local-base-model-status");
  btn.disabled = true;
  status.textContent = `Checking ${file.name}…`;
  try {
    const row = await api("POST", "/v1/base-models/local", form);
    await loadLocalBaseModels();
    state.baseModel = { row, name: file.name, projectName: null };
    dropParentModel();
    renderBaseModel();
    input.value = "";
    status.textContent = `${file.name} accepted as base model #${row.id}.`;
    toast(`${file.name} will be the base model for the next ${pipelineName(row.pipeline)} job`);
  } catch (err) {
    status.textContent = "";
    throw err;
  } finally {
    btn.disabled = false;
  }
}

function studioModelName(m) {
  return (m.manifest && m.manifest.name) || `Studio model ${m.id}`;
}

function renderStudioModelChoice() {
  const sel = $("#studio-model-select");
  const m = (state.studioModels || []).find((x) => x.id === sel.value);
  $("#studio-model-use").disabled = !usableAsBase(m) || !baseModelAllowed();
  const details = $("#studio-model-details");
  if (!m) {
    details.textContent = "";
    return;
  }
  const f1 = m.metrics && m.metrics.val_macro_f1;
  details.textContent = [
    studioModelName(m),
    `Studio model ${m.id}`,
    m.pipeline,
    `labels: ${(m.labels || []).join(", ") || "none"}`,
    f1 == null ? null : `val macro-F1 ${Number(f1).toFixed(2)}`,
    m.created_at ? `pushed ${String(m.created_at).slice(0, 10)}` : null,
  ].filter(Boolean).join(" · ");
}

function fillStudioModels(models) {
  const sel = $("#studio-model-select");
  // Models usable as a base for the chosen pipeline first, newest first within each group.
  state.studioModels = [...models].sort((a, b) =>
    (usableAsBase(b) - usableAsBase(a)) || String(b.created_at || "").localeCompare(String(a.created_at || "")));
  const prev = sel.value;
  if (!state.studioModels.length) {
    sel.replaceChildren(el("option", { value: "", text: $("#studio-project").value ? "No models in this project" : "No model selected" }));
  } else {
    sel.replaceChildren(...state.studioModels.map((m) => {
      const date = m.created_at ? ` · ${String(m.created_at).slice(0, 10)}` : "";
      const why = !m.importable ? " · cannot be a base model"
        : usableAsBase(m) ? "" : ` · ${pipelineName(m.pipeline)} model`;
      const opt = el("option", {
        value: m.id,
        text: `${studioModelName(m)} (Studio model ${m.id})${date}${why}`,
      });
      opt.disabled = !usableAsBase(m);
      return opt;
    }));
    const keep = state.studioModels.find((m) => m.id === prev && usableAsBase(m));
    const first = state.studioModels.find(usableAsBase);
    sel.value = keep ? keep.id : first ? first.id : "";
  }
  sel.disabled = !baseModelAllowed() || !state.studioModels.some(usableAsBase);
  renderStudioModelChoice();
}

async function listStudioModels() {
  const project = selectedStudioProject();
  const status = $("#studio-models-status");
  status.textContent = "Loading…";
  let res;
  try {
    res = await api("GET", `/v1/connectors/studio/models?project=${encodeURIComponent(project)}`);
  } catch (err) {
    status.textContent = "Could not load models";
    throw err;
  }
  if ($("#studio-project").value !== project) return;  // the user switched projects meanwhile
  const n = res.models.length;
  const usable = res.models.filter(usableAsBase).length;
  status.textContent = `${n} model${n === 1 ? "" : "s"} in ${studioProjectName(project)}`
    + (usable === n ? "" : ` (${usable} usable as a base)`)
    + ` · updated ${new Date().toLocaleTimeString()}`;
  fillStudioModels(res.models);
}

async function importStudioModel() {
  if (!baseModelAllowed()) throw new Error("Only the image classifier and YOLO11 detector pipelines can start from a base model.");
  const project = selectedStudioProject();
  const studioModelId = $("#studio-model-select").value;
  if (!studioModelId) throw new Error("Choose a Studio model first.");
  const btn = $("#studio-model-use");
  btn.disabled = true;
  try {
    const row = await api("POST", "/v1/connectors/studio/models/import", {
      project, studio_model_id: studioModelId,
    });
    const chosen = (state.studioModels || []).find((m) => m.id === studioModelId);
    state.baseModel = {
      row,
      name: chosen ? studioModelName(chosen) : `Studio model ${studioModelId}`,
      projectName: studioProjectName(project),
    };
    dropParentModel();
    renderBaseModel();
    toast(`${state.baseModel.name} will be the base model for the next ${pipelineName(row.pipeline)} job`);
  } finally {
    renderStudioModelChoice();
  }
}

function renderSnapshot() {
  const snap = state.snapshot;
  const box = $("#class-balance");
  if (!snap) {
    $("#snapshot-info").textContent = "No snapshot yet.";
    box.replaceChildren();
  } else {
    $("#snapshot-info").textContent = `Snapshot ${snap.id}: ${snap.item_count} item(s), hash ${String(snap.content_hash).slice(0, 12)}`;
    const counts = Object.entries(snap.label_counts || {}).sort((a, b) => b[1] - a[1]);
    const max = Math.max(1, ...counts.map(([, n]) => n));
    box.replaceChildren(...counts.map(([label, n]) => el("div", {},
      el("span", { text: label }), el("i", { style: `--bar: ${Math.round((100 * n) / max)}%` }), el("b", { text: n }))));
  }
  renderSnapshotExport();
  updateTrainButton();
  if (state.parentModel) previewReuse().catch(() => {});
}

// missingBoxes="exclude" sends excludeImages, the exact list the user saw, so the server refuses
// (409) instead of leaving out an image that started needing a box after the user agreed.
async function createSnapshot(missingBoxes, excludeImages) {
  if (!state.datasetId) throw new Error("Create or select a dataset first.");
  let snap;
  let body;
  if (missingBoxes) {
    body = { missing_boxes: missingBoxes };
    if (missingBoxes === "exclude") {
      body.exclude_images = excludeImages || ((state.cropPrep && state.cropPrep.needs_boxes) || []);
    }
  }
  try {
    snap = await api("POST", `/v1/datasets/${state.datasetId}/snapshots`, body);
  } catch (err) {
    if (err instanceof ApiError && err.status === 409 && missingBoxes === "exclude") {
      await loadCropPrep();
      renderCropPrep();
      throw new Error("More images need a box than the ones you chose to leave out, so no snapshot was created. Review the updated list and choose again.");
    }
    if (!(err instanceof ApiError) || err.status !== 409 || missingBoxes) throw err;
    await loadCropPrep();
    renderCropPrep();
    const d = err.detail && typeof err.detail === "object" ? err.detail : {};
    const needs = (state.cropPrep && state.cropPrep.needs_boxes && state.cropPrep.needs_boxes.length)
      ? state.cropPrep.needs_boxes : (d.needs_boxes || d.images || []);
    if (!needs.length) throw err;
    $("#crop-prep").scrollIntoView({ block: "center" });
    const options = [];
    if (cropPrepDetector()) options.push({ value: "assist", label: "Pre-draw boxes with a detector", detail: "Runs the installed detector on these images only. Reviewed boxes, Studio crops and images a reviewer confirmed are not touched." });
    options.push(
      { value: "draw", label: "Show them so I can draw boxes", detail: "Filters the image list to these images." },
      { value: "exclude", label: "Leave them out of this snapshot", detail: "They stay in the dataset and can be boxed later." },
    );
    const pick = await chooseDialog(
      `${plural(needs.length, "image")} need a box`,
      `This pipeline trains on animal crops, and ${plural(needs.length, "image")} have a label but no box: ${needs.slice(0, 5).join(", ")}${needs.length > 5 ? ", …" : ""}. Nothing is dropped without your choice.`,
      options,
    );
    if (pick === "assist") await cropPrepAssist();
    else if (pick === "draw") await cropPrepDraw();
    else if (pick === "exclude") await createSnapshot("exclude", needs);
    return;
  }
  toast(missingBoxes === "exclude"
    ? `Snapshot ${snap.id} created with ${snap.item_count} item(s); images without a box were left out`
    : `Snapshot ${snap.id} created with ${snap.item_count} item(s)`);
  await loadDataset();
}

// ------------------------------------------------- crop preparation before a snapshot

// GET /v1/datasets/{id}/crop-prep: {mode, boxed, studio_crops, needs_boxes}. Older servers lack it.
async function loadCropPrep() {
  state.cropPrep = null;
  if (!state.datasetId) return;
  try {
    state.cropPrep = await api("GET", `/v1/datasets/${state.datasetId}/crop-prep`);
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) throw err;
  }
}

function cropPrepDetector() {
  const sel = $("#crop-prep-detector");
  const opt = sel.options[sel.selectedIndex];
  return opt && opt.value && !opt.disabled ? opt.value : "";
}

function renderCropPrep() {
  const box = $("#crop-prep");
  const prep = state.cropPrep;
  const filtering = Boolean(state.onlyImages && state.onlyImagesDataset === state.datasetId);
  $("#crop-prep-show-all").hidden = !filtering;
  if (!prep || prep.mode !== "crop") {
    box.hidden = !filtering;
    $("#crop-prep-needs").hidden = true;
    $("#crop-prep-summary").textContent = filtering ? "Showing only the images that needed a box." : "";
    return;
  }
  box.hidden = false;
  const needs = prep.needs_boxes || [];
  const parts = [`${plural(prep.boxed || 0, "image")} with a reviewed box`, `${plural(prep.studio_crops || 0, "Studio crop")}`];
  if (needs.length) parts.push(`${plural(needs.length, "image")} still need a box`);
  $("#crop-prep-summary").textContent = `This pipeline trains on animal crops. ${parts.join(", ")}.`
    + (needs.length ? "" : " All labelled images are ready for a snapshot.");
  $("#crop-prep-needs").hidden = !needs.length;
  if (!needs.length) return;
  $("#crop-prep-needs-text").textContent = `These raw images have a label but no box. Choose how to handle ${needs.length === 1 ? "it" : "them"} before the snapshot; none is dropped silently. Images where the detector finds nothing stay on this list, and so do images a reviewer confirmed: the detector never changes a confirmed image.`;
  const shown = needs.slice(0, 20);
  $("#crop-prep-list").replaceChildren(
    ...shown.map((n) => el("li", { text: n })),
    ...(needs.length > shown.length ? [el("li", { class: "muted", text: `…and ${needs.length - shown.length} more` })] : []),
  );
  const src = $("#assist-model");
  const sel = $("#crop-prep-detector");
  const prev = sel.value;
  sel.replaceChildren(...[...src.options].map((o) => o.cloneNode(true)));
  if ([...sel.options].some((o) => o.value === prev && !o.disabled)) sel.value = prev;
  else {
    const first = [...sel.options].find((o) => o.value && !o.disabled);
    if (first) sel.value = first.value;
  }
  const usable = Boolean(cropPrepDetector());
  sel.disabled = !usable;
  $("#crop-prep-assist").disabled = !usable;
  if (!usable) $("#crop-prep-status").textContent = "No detector is installed on this service, so boxes cannot be pre-drawn. Draw the boxes yourself or leave these images out.";
  else if ($("#crop-prep-status").dataset.kind !== "result") $("#crop-prep-status").textContent = "";
}

// Runs the chosen installed detector on the images that still need a box, and only on them.
async function cropPrepAssist() {
  const modelId = cropPrepDetector();
  if (!modelId) throw new Error("No detector is installed. Draw the boxes yourself or leave these images out.");
  const needs = (state.cropPrep && state.cropPrep.needs_boxes) || [];
  if (!needs.length) return;
  const status = $("#crop-prep-status");
  let written = 0;
  const confirmed = [];
  const empty = [];
  const other = [];
  await withBusy(`Pre-drawing boxes on ${plural(needs.length, "image")}…`, $("#crop-prep-assist"), async () => {
    for (let i = 0; i < needs.length; i += ASSIST_MAX_ITEMS) {
      const images = needs.slice(i, i + ASSIST_MAX_ITEMS);
      const res = await api("POST", `/v1/datasets/${state.datasetId}/assist`, {
        model_id: modelId, images, sources: ["upload", "studio"], conf: assistConf(), overwrite: false,
        needs_boxes_only: true,
      });
      written += Number((res && res.items_written) || 0);
      for (const s of (res && res.skipped) || []) {
        const reason = String((s && s.reason) || "");
        if (/confirmed/i.test(reason)) confirmed.push(s.image);
        else if (/^no detections/i.test(reason)) empty.push(s.image);
        else other.push(s.image);
      }
    }
  });
  await loadDataset();
  const left = ((state.cropPrep && state.cropPrep.needs_boxes) || []).length;
  const them = (n) => (n === 1 ? "it" : "them");
  const parts = [`Boxes pre-drawn on ${plural(written, "image")}.`];
  // Confirmed images are protected: the detector never replaces a reviewer's decision, so they
  // are skipped, not "nothing found".
  if (confirmed.length) parts.push(`${plural(confirmed.length, "image")} confirmed by a reviewer ${confirmed.length === 1 ? "was" : "were"} skipped and not changed (${confirmed.slice(0, 3).join(", ")}${confirmed.length > 3 ? ", …" : ""}); draw a box yourself or leave ${them(confirmed.length)} out.`);
  if (empty.length) parts.push(`The detector found nothing on ${plural(empty.length, "image")}; draw a box or leave ${them(empty.length)} out.`);
  if (other.length) parts.push(`${plural(other.length, "image")} ${other.length === 1 ? "was" : "were"} skipped for another reason (for example changed while the detector ran).`);
  if (left) parts.push(`${plural(left, "image")} still need a box.`);
  parts.push("Review the pre-drawn boxes before the snapshot.");
  status.dataset.kind = "result";
  status.textContent = parts.join(" ");
  toast(`Pre-drawn boxes on ${plural(written, "image")}`);
}

async function cropPrepDraw() {
  const needs = (state.cropPrep && state.cropPrep.needs_boxes) || [];
  if (!needs.length) return;
  state.onlyImages = new Set(needs);
  state.onlyImagesDataset = state.datasetId;
  await loadDataset();
  $("#example-grid").scrollIntoView({ block: "start" });
}

async function cropPrepShowAll() {
  state.onlyImages = null;
  state.onlyImagesDataset = null;
  await loadDataset();
}

async function cropPrepExclude() {
  const needs = (state.cropPrep && state.cropPrep.needs_boxes) || [];
  const ok = await chooseDialog(
    "Leave these images out?",
    `${plural(needs.length, "image")} without a box will not be in this snapshot. They stay in the dataset, and their labels are kept.`,
    [{ value: "exclude", label: "Create the snapshot without them" }],
  );
  if (ok === "exclude") await createSnapshot("exclude", needs);
}

// Shows a blocking "working" panel while fn runs, and disables the button that started it.
async function withBusy(message, button, fn) {
  const overlay = $("#busy-overlay");
  $("#busy-text").textContent = message;
  overlay.hidden = false;
  if (button) {
    button.disabled = true;
    button.setAttribute("aria-busy", "true");
  }
  try {
    return await fn();
  } finally {
    overlay.hidden = true;
    if (button) button.removeAttribute("aria-busy");
  }
}

async function startTraining() {
  const pipeline = $("#pipeline-select").value;
  if (!state.snapshot) throw new Error("Create a snapshot first.");
  const params = {};
  const parent = state.parentModel && state.parentModel.pipeline === pipeline ? state.parentModel : null;
  if (parent) {
    const preview = state.reusePreview;
    if (preview && preview.snapshotId === state.snapshot.id && preview.usable === false) {
      throw new Error(`${modelTitle(parent)} cannot be retrained on snapshot ${state.snapshot.id}: ${preview.reason || "it does not fit this snapshot"}. Choose another model or clear the selection.`);
    }
    params.parent_model_id = parent.id;
  } else if (state.baseModel && state.baseModel.row.pipeline === pipeline) {
    params.base_model_id = state.baseModel.row.id;
  }
  if (pipeline === "yolo_detector") {
    // A base model fixes the network; otherwise the chosen starting weights do.
    params.model = params.base_model_id ? state.baseModel.row.model_arch : $("#start-weights").value;
    params.batch = batchParam(pipeline);
    params.imgsz = Number($("#imgsz").value);
  }
  if (pipeline === "conventional_classifier") {
    params.batch = batchParam(pipeline);
    // A base model or finished parent fixes the network the job trains; the package or model
    // names its own timm architecture, so it is never assumed here.
    if (params.base_model_id) params.model = state.baseModel.row.model_arch;
    else if (parent && parentArch(parent)) params.model = parentArch(parent);
  }
  if (CLASSIFIER_PIPELINES.has(pipeline)) {
    params.threshold_target_accuracy = Number($("#target-accuracy").value);
    if ($("#stock-model").value) params.stock_model = $("#stock-model").value;
  }
  Object.assign(params, trainingSettingsParams(pipeline));
  try {
    await withBusy("Starting training…", $("#start-training"), async () => {
      const job = await api("POST", "/v1/jobs", { pipeline, snapshot_id: state.snapshot.id, params });
      state.jobId = job.id;
      toast(`Job ${job.id} queued`);
      await showPage("jobs");
    });
  } finally {
    updateTrainButton();
  }
}

// ------------------------------------------------------------------ jobs page

function closeStream() {
  if (state.stream) {
    state.stream.close();
    state.stream = null;
  }
}

// Jobs and models belong to the workspace their training dataset is in.
function workspaceScope(kind) {
  const ws = state.workspaces.find((w) => w.id === state.workspaceId);
  return ws ? `Showing ${kind} trained on datasets in workspace "${ws.name}".` : "Select a workspace on Prepare data to see its " + kind + ".";
}

async function workspaceList(path, key) {
  if (!state.workspaceId) return [];
  const sep = path.includes("?") ? "&" : "?";
  return (await api("GET", `${path}${sep}workspace_id=${state.workspaceId}`))[key];
}

async function loadJobs() {
  state.jobs = (await workspaceList("/v1/jobs?limit=200", "jobs")).sort((a, b) => b.id - a.id);
  $("#jobs-scope").textContent = workspaceScope("jobs");
  if (!state.jobs.some((j) => j.id === state.jobId)) state.jobId = null;
  renderHistory();
  renderActiveChip();
  if (!state.jobId && state.jobs.length) {
    const live = state.jobs.find((j) => !TERMINAL.has(j.state));
    state.jobId = (live || state.jobs[0]).id;
  }
  if (state.jobId) {
    await selectJob(state.jobId);
  } else {
    closeStream();
    $("#job-dashboard").hidden = true;
    $("#job-empty").hidden = false;
  }
}

function renderActiveChip() {
  const live = state.jobs.filter((j) => !TERMINAL.has(j.state));
  $("#active-job-chip").hidden = live.length === 0;
  $("#active-job-chip-text").textContent = `${live.length} job(s) in progress`;
  const badge = $("#nav-job-badge");
  badge.hidden = live.length === 0;
  badge.textContent = String(live.length);
}

function renderHistory() {
  $("#job-history").replaceChildren(...state.jobs.map((j) => {
    const row = el("tr", { class: `selectable${j.id === state.jobId ? " selected" : ""}`, "data-job-id": j.id },
      el("td", { text: `#${j.id}` }),
      el("td", { text: pipelineName(j.pipeline) }),
      el("td", { text: j.snapshot_id }),
      el("td", {}, el("span", {
        class: `status-badge ${STATE_BADGE[j.state] || ""}`, text: j.state,
        ...(j.state === "queued" && j.queue ? { title: queueText(j.queue) } : {}),
      })),
      el("td", { text: fmtTime(j.created_at) }));
    return row;
  }));
}

async function selectJob(jobId) {
  closeStream();
  state.jobId = jobId;
  state.jobEvents = [];
  const job = await api("GET", `/v1/jobs/${jobId}`);
  renderJob(job);
  renderHistory();
  $("#event-list").replaceChildren();
  drawChart();
  openStream(jobId);
}

function queueText(q) {
  if (!q) return "waiting for a worker";
  const gpu = (q.requires || []).includes("gpu") ? "GPU worker" : "worker";
  const ahead = q.ahead ? ` · ${q.ahead} ahead` : "";
  if (q.reason === "no_worker") return `No ${gpu} online${ahead}`;
  if (q.reason === "user_limit") {
    return `Your limit of ${q.user_limit} running job${q.user_limit === 1 ? "" : "s"} reached${ahead}`;
  }
  if (q.reason === "busy") return `Waiting for a ${gpu}: all ${q.busy} busy${ahead}`;
  return `Starting on a free ${gpu}${ahead}`;
}

function renderJob(job) {
  $("#job-empty").hidden = true;
  $("#job-dashboard").hidden = false;
  $("#job-id").textContent = `JOB #${job.id}`;
  $("#job-name").textContent = pipelineName(job.pipeline);
  $("#job-recipe").textContent = `Snapshot ${job.snapshot_id} · attempts ${job.attempts}/${job.max_attempts}`;
  const badge = $("#job-status");
  badge.textContent = job.state;
  badge.className = `status-badge ${STATE_BADGE[job.state] || ""}`;
  badge.dataset.state = job.state;
  if (job.state === "queued" && job.queue === undefined) {
    // Stream updates carry no queue status; keep the last one the API returned.
    const known = state.jobs.find((j) => j.id === job.id);
    if (known && known.queue) job = { ...job, queue: known.queue };
  }
  $("#job-time").textContent = job.started_at ? `started ${fmtTime(job.started_at)}` : queueText(job.queue);
  $("#cancel-job").hidden = TERMINAL.has(job.state);
  $("#retry-job").hidden = !(job.state === "failed" || job.state === "cancelled");
  const err = $("#job-error");
  err.hidden = !job.error;
  err.textContent = job.error ? `Error: ${job.error}` : "";
  renderStages(job);
  renderJobHardware(job);
  const idx = state.jobs.findIndex((j) => j.id === job.id);
  if (idx >= 0) state.jobs[idx] = job; else state.jobs.unshift(job);
  renderActiveChip();
}

// The YOLO detector and the image classifier report their device and resolved batch size as a train event message.
const HARDWARE_EVENT = /^training on .+, batch \d+/;

function renderJobHardware(job) {
  const line = $("#job-hardware");
  const ev = [...state.jobEvents].reverse().find((e) => HARDWARE_EVENT.test(e.message || ""));
  const requested = job && job.params ? job.params.batch : undefined;
  if (ev) {
    line.textContent = `Hardware: ${ev.message.replace(/^training on /, "")}`;
  } else if (requested !== undefined && requested !== null) {
    line.textContent = requested === "auto"
      ? "Batch size: automatic (picked when training starts)"
      : `Batch size: ${requested} (set by parameter)`;
  } else {
    line.textContent = "";
  }
  line.hidden = !line.textContent;
}

function renderStages(job) {
  const stages = [];
  for (const ev of state.jobEvents) if (!stages.includes(ev.stage)) stages.push(ev.stage);
  const done = TERMINAL.has(job.state);
  $("#job-stage-title").textContent = done ? job.state : (stages[stages.length - 1] || job.state);
  $("#stage-list").replaceChildren(...stages.map((s, i) => el("li", {
    class: done || i < stages.length - 1 ? "complete" : "active",
  }, s)));
}

function openStream(jobId) {
  if (typeof EventSource === "undefined") return;
  const es = new EventSource(`/v1/jobs/${jobId}/stream`, { withCredentials: true });
  state.stream = es;
  es.addEventListener("job-event", (msg) => {
    if (state.jobId !== jobId) return;
    const ev = JSON.parse(msg.data);
    if (state.jobEvents.some((e) => e.seq === ev.seq)) return;
    state.jobEvents.push(ev);
    const metrics = numericMetrics(ev.metrics);
    const detail = Object.entries(metrics).map(([k, v]) => `${k}=${fmtNum(v)}`).join(" ");
    $("#event-list").prepend(el("li", {},
      el("time", { text: fmtTime(ev.created_at) }),
      `${ev.stage}: ${ev.message || ""}${detail ? ` (${detail})` : ""}`));
    const job = state.jobs.find((j) => j.id === jobId);
    if (job) renderStages(job);
    if (job && HARDWARE_EVENT.test(ev.message || "")) renderJobHardware(job);
    drawChart();
  });
  es.addEventListener("job-state", async (msg) => {
    if (state.jobId !== jobId) return;
    const cached = state.jobs.find((j) => j.id === jobId);
    if (cached && cached.state === JSON.parse(msg.data).state) return;
    // Re-read the whole job so started_at, attempts and errors match the new state.
    // Not guarded(): a background refresh must not clear an unrelated error banner.
    try {
      renderJob(await api("GET", `/v1/jobs/${jobId}`));
      renderHistory();
    } catch (err) {
      showError(err && err.message ? err.message : String(err));
    }
  });
  es.addEventListener("end", guarded(async () => {
    es.close();
    if (state.stream === es) state.stream = null;
    if (state.jobId !== jobId) return;
    renderJob(await api("GET", `/v1/jobs/${jobId}`));
    renderHistory();
  }));
  es.onerror = () => {
    // The browser reconnects with Last-Event-ID; refresh the job so a lost session surfaces.
    if (es.readyState === EventSource.CLOSED) guarded(async () => renderJob(await api("GET", `/v1/jobs/${jobId}`)))();
  };
}

const SERIES_COLOURS = ["#0e4a59", "#b86608", "#16864c", "#70509a"];

function chartSeries() {
  const points = state.jobEvents.filter((e) => e.stage === "train");
  const keys = [];
  for (const p of points) {
    for (const k of Object.keys(numericMetrics(p.metrics))) {
      if (k !== "epoch" && !keys.includes(k)) keys.push(k);
    }
  }
  return keys.slice(0, 4).map((k) => ({
    key: k,
    values: points.map((p, i) => [typeof p.metrics.epoch === "number" ? p.metrics.epoch : i, p.metrics[k]])
      .filter(([, v]) => typeof v === "number" && Number.isFinite(v)),
  }));
}

function drawChart() {
  const canvas = $("#live-curve");
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const series = chartSeries();
  $("#chart-legend").replaceChildren(...series.map((s, i) => el("span", {},
    el("i", { style: `background:${SERIES_COLOURS[i]}` }), el("b", { text: s.key }))));
  const pad = 36;
  ctx.strokeStyle = "#e2e6eb";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, 10, W - pad - 10, H - pad - 10);
  const all = series.flatMap((s) => s.values);
  if (!all.length) {
    ctx.fillStyle = "#6b7785";
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText("Waiting for training measurements", pad + 12, H / 2);
    return;
  }
  const xs = all.map(([x]) => x);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs, x0 + 1);
  // Each series is normalised to its own range so losses and scores share one plot.
  series.forEach((s, i) => {
    const vs = s.values.map(([, v]) => v);
    const lo = Math.min(...vs);
    const hi = Math.max(...vs, lo + 1e-9);
    ctx.strokeStyle = SERIES_COLOURS[i];
    ctx.lineWidth = 2;
    const pts = s.values.map(([x, v]) => [
      pad + ((x - x0) / (x1 - x0)) * (W - pad - 10),
      10 + (1 - (v - lo) / (hi - lo)) * (H - pad - 20),
    ]);
    ctx.beginPath();
    pts.forEach(([px, py], j) => { if (j === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
    ctx.stroke();
    // Markers keep a single measurement visible (a one-point path draws nothing).
    ctx.fillStyle = SERIES_COLOURS[i];
    for (const [px, py] of pts) {
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
  ctx.fillStyle = "#6b7785";
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText(`epoch ${fmtNum(x0)}`, pad, H - 8);
  ctx.fillText(`epoch ${fmtNum(x1)}`, W - 80, H - 8);
}

async function cancelJob() {
  if (!state.jobId) return;
  renderJob(await api("POST", `/v1/jobs/${state.jobId}/cancel`));
  toast(`Cancellation requested for job ${state.jobId}`);
}

async function retryJob() {
  if (!state.jobId) return;
  const button = $("#retry-job");
  if (button.disabled) return;
  button.disabled = true;
  try {
    const job = await api("POST", `/v1/jobs/${state.jobId}/retry`);
    toast(`Job ${job.id} queued as a retry of job ${state.jobId}`);
    state.jobId = job.id;
    await loadJobs();
  } finally {
    button.disabled = false;
  }
}

// ------------------------------------------------------------------ models page

async function loadModels() {
  state.models = (await workspaceList("/v1/models", "models")).sort((a, b) => b.id - a.id);
  $("#models-scope").textContent = workspaceScope("models");
  $("#model-empty").hidden = state.models.length > 0;
  if (!state.models.some((m) => m.id === state.modelId)) state.modelId = state.models.length ? state.models[0].id : null;
  renderModelList();
  renderModel();
}

function modelTitle(m) {
  return m.name || `Model #${m.id} · ${pipelineName(m.pipeline)}`;
}

// The name a toast uses for a model: its custom name, or "Model #<id>".
function modelLabel(id) {
  const m = state.models.find((x) => x.id === id);
  return m && m.name ? `"${m.name}"` : `Model #${id}`;
}

function renderModelList() {
  $("#model-list").replaceChildren(...state.models.map((m) => {
    const active = (m.active_deployments || []).length > 0;
    return el("button", { type: "button", class: `model-list-item${m.id === state.modelId ? " selected" : ""}`, "data-model-id": m.id },
      el("span", { class: "pipeline-icon", text: pipelineName(m.pipeline).charAt(0).toUpperCase() }),
      el("span", {}, el("strong", { text: modelTitle(m) }), el("small", { text: `${m.name ? `#${m.id} · ${pipelineName(m.pipeline)} · ` : ""}job ${m.job_id} · snapshot ${m.snapshot_id}` })),
      el("span", { class: `status-badge ${active ? "status-complete" : "status-ready"}`, text: active ? "Active" : "Ready" }));
  }));
}

function renderModel() {
  const m = state.models.find((x) => x.id === state.modelId);
  $("#model-detail").hidden = !m;
  if (!m) return;
  $("#model-name").textContent = modelTitle(m);
  if (state.renamingModel !== m.id) toggleRename(false);
  $("#model-format").textContent = m.format;
  const which = m.name ? `Model #${m.id} · ${pipelineName(m.pipeline)} · ` : "";
  $("#model-description").textContent = `${which}Created ${fmtTime(m.created_at)} · sha256 ${String(m.sha256).slice(0, 16)}…`;
  const deps = m.active_deployments || [];
  const badge = $("#model-active-badge");
  badge.hidden = deps.length === 0;
  badge.textContent = deps.length
    ? `Active: ${deps.map((d) => `${d.consumer}/${d.consumer === "studio" ? studioProjectName(d.project) : d.project}`).join(", ")}`
    : "";
  $("#activate-studio").hidden = !state.studio.configured;
  $("#studio-deploy-label").hidden = !state.studio.configured;
  loadStudioDetectors();
  $("#model-download").href = `/v1/models/${m.id}/artifact`;
  const ft = $("#model-finetune");
  const tag = m.finetune_tag === FINETUNE_YES || m.finetune_tag === FINETUNE_NO ? m.finetune_tag : null;
  ft.hidden = !tag;
  ft.className = `model-finetune ${tag === FINETUNE_YES ? "finetune-yes" : "finetune-no"}`;
  ft.replaceChildren(...(tag ? [
    el("span", { class: `finetune-tag ${tag === FINETUNE_YES ? "finetune-yes" : "finetune-no"}`, text: tag }),
    ...(m.finetune_reason ? [el("span", { class: "muted", text: ` ${m.finetune_reason}` })] : []),
  ] : []));
  $("#model-reuse").hidden = !reusable(m);
  renderStudioInstalls(m);

  renderStockComparison(m.stock_comparison);
  renderReferenceComparison(m.reference_comparison);

  const metrics = overallMetrics(m.metrics);
  const keys = Object.keys(metrics).slice(0, 8);
  $("#metrics-grid").replaceChildren(...keys.map((k) => el("div", {}, el("span", { text: k }), el("strong", { text: fmtNum(metrics[k]) }))));
  renderThreshold(m);
  renderPerClass(m);

  const cmp = $("#compare-select");
  const others = state.models.filter((x) => x.id !== m.id);
  const prev = Number(cmp.value);
  cmp.replaceChildren(...others.map((x) => el("option", { value: x.id, text: modelTitle(x) })));
  if (!others.length) cmp.append(el("option", { value: "", text: "(no other version)" }));
  if (others.some((x) => x.id === prev)) cmp.value = String(prev);
  renderCompare();

  const record = [
    ["Name", m.name || "(default)"], ["Job", `#${m.job_id}`], ["Snapshot", m.snapshot_id], ["Pipeline", m.pipeline],
    ["Format", m.format], ["Artifact sha256", m.sha256],
  ];
  const hw = hardwareText(m.metadata && m.metadata.hardware);
  if (hw) record.push(["Trained on", hw]);
  const parentId = m.metadata && m.metadata.parent_model_id;
  if (parentId) record.push(["Started from", `Model #${parentId}`]);
  $("#model-record").replaceChildren(...record.map(([k, v]) => el("div", {}, el("dt", { text: k }), el("dd", { text: v }))));
}

// ------------------------------------------------------------- model catalogue

// GET /v1/catalog: {mode: "standalone"|"connected", models, catalog_error}. Every row carries a
// fine-tune tag before anything is downloaded; downloads install one chosen id at a time.
async function loadCatalog() {
  const err = $("#catalog-error");
  const status = $("#catalog-status");
  err.hidden = true;
  status.textContent = "Loading the catalogue…";
  let body;
  try {
    body = await api("GET", "/v1/catalog");
  } catch (e) {
    if (e instanceof ApiError && e.status === 401) throw e;
    $("#catalog-list").replaceChildren();
    status.textContent = "";
    err.hidden = false;
    err.textContent = e instanceof ApiError && e.status === 404
      ? "This server does not list a model catalogue."
      : `The catalogue could not be loaded: ${e.message}`;
    return;
  }
  state.catalog = body;
  const connected = body.mode === "connected";
  $("#catalog-title").textContent = connected ? "Models in Sparrow Studio" : "Model Zoo (Sparrow Engine)";
  $("#catalog-mode-note").textContent = connected
    ? "This service is connected to Sparrow Studio. The list shows the Studio project's models; install and manage them in Studio. Nothing is downloaded here."
    : "Standalone mode: the list comes from the local Sparrow Engine model zoo. Nothing is downloaded until you choose one model and confirm.";
  if (body.catalog_error) {
    err.hidden = false;
    err.textContent = `The online catalogue is unavailable (${body.catalog_error}). Installed models are still listed and usable.`;
  }
  const models = body.models || [];
  $("#catalog-list").replaceChildren(...models.map((row) => catalogRow(row, connected)));
  const yes = models.filter((r) => r.finetune_tag === FINETUNE_YES).length;
  status.textContent = models.length
    ? `${plural(models.length, "model")} listed; ${yes} can be fine-tuned.`
    : "The catalogue lists no models.";
}

function catalogRow(row, connected) {
  const tag = row.finetune_tag === FINETUNE_YES ? FINETUNE_YES : FINETUNE_NO;
  const yes = tag === FINETUNE_YES;
  const facts = [row.task, row.domain, row.format, row.license ? `licence ${row.license}` : "no licence listed"].filter(Boolean).join(" · ");
  let action;
  if (row.installed) action = el("span", { class: "status-badge status-complete", text: "Installed" });
  else if (connected) action = el("span", { class: "muted", text: "Install through Sparrow Studio" });
  else if (row.downloadable) action = el("button", { type: "button", class: "btn btn-lite compact", "data-catalog-download": String(row.id), text: "Download" });
  else if (row.source_url) action = el("a", { href: row.source_url, target: "_blank", rel: "noopener", text: "Get it from the source" });
  else action = el("span", { class: "muted", text: "Not available to download" });
  return el("div", { class: "catalog-row", "data-catalog-id": String(row.id) },
    el("div", { class: "catalog-main" },
      el("strong", { text: row.name || String(row.id) }),
      el("small", { class: "muted", text: facts }),
    ),
    el("div", { class: "catalog-tag" },
      el("span", { class: `finetune-tag ${yes ? "finetune-yes" : "finetune-no"}`, text: tag }),
      ...(row.finetune_reason ? [el("small", { class: "muted", text: row.finetune_reason })] : []),
    ),
    el("div", { class: "catalog-action" }, action),
  );
}

async function downloadCatalogModel(button) {
  const id = button.dataset.catalogDownload;
  const row = ((state.catalog && state.catalog.models) || []).find((r) => String(r.id) === id) || {};
  const ok = await chooseDialog(
    `Download ${row.name || id}?`,
    `Installs only this model into the shared Sparrow Engine model folder on this server, so every user of this service sees it.${row.license ? ` Licence: ${row.license}.` : ""} ${row.finetune_tag === FINETUNE_YES ? "It can be fine-tuned." : "It can run, but it cannot be fine-tuned here."}`,
    [{ value: "download", label: "Download this model" }],
  );
  if (ok !== "download") return;
  await withBusy(`Downloading ${row.name || id}…`, button, () => api("POST", `/v1/catalog/models/${encodeURIComponent(id)}/download`));
  toast(`${row.name || id} is installed`);
  await loadCatalog();
}

// Detector metrics carry one flat key per class ("precision.zebra"); the per-class table shows them.
const PER_CLASS_KEY = /^(precision|recall|map50_95|map50|instances|ap50_95)\./;
const PER_CLASS_COLUMNS = ["precision", "recall", "map50", "map50_95", "instances"];
const CLASSIFIER_PIPELINES = new Set(["conventional_classifier", "dinov3_linear"]);

function overallMetrics(metrics) {
  const out = numericMetrics(metrics);
  for (const k of Object.keys(out)) if (PER_CLASS_KEY.test(k)) delete out[k];
  return out;
}

function perClassRows(m) {
  const meta = (m.metadata && m.metadata.per_class) || null;
  if (Array.isArray(meta) && meta.length) return meta;
  const byLabel = new Map();
  for (const [k, v] of Object.entries(numericMetrics(m.metrics))) {
    const hit = PER_CLASS_KEY.exec(k);
    if (!hit) continue;
    const label = k.slice(hit[0].length);
    if (!byLabel.has(label)) byLabel.set(label, { label, precision: null, recall: null, map50: null, map50_95: null, instances: null });
    const row = byLabel.get(label);
    if (hit[1] === "ap50_95") { if (row.map50_95 === null) row.map50_95 = v; } else row[hit[1]] = v;
  }
  return [...byLabel.values()].sort((a, b) => (a.label < b.label ? -1 : 1));
}

function renderPerClass(m) {
  const rows = perClassRows(m);
  $("#per-class-section").hidden = rows.length === 0;
  const cell = (v, pct) => el("td", { text: typeof v === "number" && Number.isFinite(v) ? (pct ? fmtNum(v) : String(Math.round(v))) : "—" });
  $("#per-class-body").replaceChildren(...rows.map((r) => el("tr", { "data-label": r.label },
    el("td", { text: r.label }),
    ...PER_CLASS_COLUMNS.map((c) => cell(r[c], c !== "instances")))));
}

function pct(v) {
  return `${(100 * v).toFixed(1)}%`;
}

// FT-30: with little data the calibrated threshold can leave most images "unknown". Say so, and
// quote the coverage a lower target would give (calibration.COVERAGE_TARGETS) when it is recorded.
function lowCoverageHint(coverage, cal) {
  if (typeof coverage !== "number" || coverage >= 0.5) return "";
  let text = `Only ${pct(coverage)} of validation images get a class; the rest are labelled "${cal.abstain_label || "unknown"}". This is common with few images per class.`;
  const target = typeof cal.target_accuracy === "number" ? cal.target_accuracy : 1;
  const better = (cal.coverage_by_target || []).filter((r) => r.target < target - 1e-9 && r.coverage > coverage);
  if (better.length) {
    text += ` On the same validation images, a lower target accuracy would give: ${better.map((r) => `${pct(r.target)} target → ${pct(r.coverage)} get a class`).join("; ")}.`;
  }
  return `${text} To change it, add more images per class or train again with a lower "Target accuracy for the confidence threshold".`;
}

function renderThreshold(m) {
  const box = $("#threshold-info");
  const meta = m.metadata || {};
  const cal = meta.calibration || {};
  const metrics = m.metrics || {};
  const threshold = typeof metrics.calibrated_threshold === "number" ? metrics.calibrated_threshold
    : typeof meta.calibrated_threshold === "number" ? meta.calibrated_threshold
      : typeof cal.calibrated_threshold === "number" ? cal.calibrated_threshold : null;
  const isClassifier = CLASSIFIER_PIPELINES.has(m.pipeline) || threshold !== null;
  box.hidden = !isClassifier;
  if (!isClassifier) return;
  const abstain = meta.abstain_label || cal.abstain_label || "unknown";
  const coverage = typeof metrics.threshold_coverage === "number" ? metrics.threshold_coverage : cal.threshold_coverage;
  const accuracy = typeof metrics.threshold_accuracy === "number" ? metrics.threshold_accuracy : cal.threshold_accuracy;
  const parts = [el("h3", { text: "Confidence threshold" })];
  if (threshold === null) {
    parts.push(el("p", { text: `No confidence threshold was set, so every prediction gets a class. ${cal.reason ? `Reason: ${cal.reason}` : ""}`.trim() }));
  } else {
    const facts = [el("div", {}, el("span", { text: "Threshold" }), el("strong", { text: fmtNum(threshold) }))];
    if (typeof coverage === "number") facts.push(el("div", {}, el("span", { text: "Images that get a class" }), el("strong", { text: pct(coverage) })));
    if (typeof accuracy === "number") facts.push(el("div", {}, el("span", { text: "Accuracy on those images" }), el("strong", { text: pct(accuracy) })));
    facts.push(el("div", {}, el("span", { text: "Label below the threshold" }), el("strong", { text: abstain })));
    parts.push(el("div", { class: "threshold-facts" }, ...facts));
    let how = `Predictions with confidence below ${fmtNum(threshold)} are labelled "${abstain}" instead of a class.`;
    if (typeof cal.target_accuracy === "number") how += ` The threshold was picked on validation images to reach ${pct(cal.target_accuracy)} accuracy on the predictions that keep a class.`;
    parts.push(el("p", { class: "help-text", text: how }));
    const low = lowCoverageHint(coverage, cal);
    if (low) parts.push(el("p", { class: "threshold-warning", role: "note", text: low }));
  }
  box.replaceChildren(...parts);
}

function hardwareText(hw) {
  if (!hw || typeof hw !== "object") return "";
  let where = hw.device === "cpu" || !hw.device ? "CPU" : `GPU ${hw.device}`;
  if (hw.gpu_name) where += ` (${hw.gpu_name}${hw.gpu_total_mb ? `, ${hw.gpu_total_mb} MB` : ""})`;
  if (hw.batch === undefined || hw.batch === null) return where;
  return `${where}, batch ${hw.batch} (${hw.batch_source === "auto" ? "automatic" : "set by parameter"})`;
}

function toggleRename(open) {
  const m = state.models.find((x) => x.id === state.modelId);
  state.renamingModel = open && m ? m.id : null;
  $("#rename-model-form").hidden = !open;
  $("#show-rename-model").hidden = open;
  if (open) {
    $("#rename-model").value = (m && m.name) || "";
    $("#rename-model").focus();
  }
}

async function renameModel() {
  const id = state.modelId;
  const name = $("#rename-model").value.trim();
  const button = $("#save-rename-model");
  try {
    await withBusy("Saving the name…", button, () => api("PATCH", `/v1/models/${id}`, { name: name || null }));
  } finally {
    button.disabled = false;
  }
  toggleRename(false);
  await loadModels();
  toast(name ? `Model #${id} is now named "${name}"` : `Model #${id} uses its default name again`);
}

// Studio installs a pushed model in the background; keep showing where it is.
const INSTALL_POLL_MS = 2000;
let installPoll = null;

function installText(d, st) {
  const where = `Studio project "${studioProjectName(d.project)}"`;
  if (!st) return `Sent to ${where} — checking Studio's install status…`;
  if (st.error) return `Sent to ${where} — could not read the install status: ${st.error}`;
  if (st.install_status === "installed") return `Sent to ${where} — installed; Studio is using it.`;
  if (st.install_status === "failed") return `Sent to ${where} — install failed: ${st.install_error || "no reason given"}`;
  if (st.install_status === "pending" || st.install_status === "installing") {
    return `Sent to ${where} — ${st.install_status}, waiting for Studio to install it…`;
  }
  return `Sent to ${where} — install status: ${st.install_status}`;
}

async function renderStudioInstalls(m) {
  clearTimeout(installPoll);
  const box = $("#studio-install-status");
  const deps = (m.active_deployments || []).filter((d) => d.consumer === "studio" && d.external_id);
  box.hidden = deps.length === 0;
  if (!deps.length) { box.replaceChildren(); return; }
  const draw = (states) => {
    box.replaceChildren(...deps.map((d, i) => {
      const st = states[i];
      const kind = !st ? "pending" : st.error || st.install_status === "failed" ? "failed"
        : st.install_status === "installed" ? "installed" : "pending";
      return el("span", { class: `install-line install-${kind}`, text: installText(d, st) });
    }));
  };
  if (!box.childElementCount) draw([]);
  const states = await Promise.all(deps.map((d) => {
    const q = new URLSearchParams({ project: d.project, studio_model_id: d.external_id });
    return api("GET", `/v1/connectors/studio/install?${q}`).catch((err) => ({ error: err.message, finished: true }));
  }));
  if (state.modelId !== m.id) return;
  draw(states);
  if (states.some((st) => !st.finished)) {
    installPoll = setTimeout(() => {
      const current = state.models.find((x) => x.id === m.id);
      if (current && state.modelId === m.id && !$("#model-detail").hidden) renderStudioInstalls(current);
    }, INSTALL_POLL_MS);
  }
}

function stockText(c) {
  const f1 = (v) => (v == null ? "unknown" : Number(v).toFixed(2));
  const crops = c.crops == null ? "" : ` on ${c.crops} validation crops`;
  return `Stock model ${c.model || "(unnamed)"}: macro-F1 ${f1(c.stock_macro_f1)} · this fine-tuned model: ${f1(c.finetuned_macro_f1)}${crops}.`;
}

function renderStockComparison(c) {
  const box = $("#stock-comparison");
  box.hidden = !c;
  box.classList.toggle("stock-worse", Boolean(c && c.worse));
  if (!c) { box.replaceChildren(); return; }
  box.replaceChildren(
    el("strong", { text: c.worse ? "The stock model scored higher than this fine-tuned model." : "This fine-tuned model scored at least as well as the stock model." }),
    el("span", { text: ` ${stockText(c)}` }),
    ...(c.worse && c.enforced === false
      ? [el("span", { text: ` Fewer than ${c.min_val_crops} validation crops were compared, so activation is not blocked; check the model before activating it.` })]
      : []));
}

// FT-19: the head trailed its logistic-regression reference on a validation set too small to fail on.
function renderReferenceComparison(c) {
  const box = $("#reference-comparison");
  const show = Boolean(c && c.worse);
  box.hidden = !show;
  if (!show) { box.replaceChildren(); return; }
  const f1 = (v) => (v == null ? "unknown" : Number(v).toFixed(3));
  const crops = c.crops == null ? "the" : `only ${c.crops}`;
  box.replaceChildren(
    el("strong", { text: "Warning: the logistic-regression reference scored higher than this model." }),
    el("span", { text: ` Reference macro-F1 ${f1(c.reference_macro_f1)} · this model: ${f1(c.finetuned_macro_f1)}, on ${crops} validation crops. The set is too small for the gate to fail the job; check the model before activating it.` }));
}

function renderCompare() {
  const a = state.models.find((x) => x.id === state.modelId);
  const b = state.models.find((x) => x.id === Number($("#compare-select").value));
  const body = $("#compare-body");
  if (!a || !b) {
    body.replaceChildren(el("tr", {}, el("td", { colSpan: 4, class: "muted", text: "Pick another version to compare." })));
    return;
  }
  $("#compare-head-a").textContent = `#${a.id}`;
  $("#compare-head-b").textContent = `#${b.id}`;
  const ma = overallMetrics(a.metrics);
  const mb = overallMetrics(b.metrics);
  const keys = [...new Set([...Object.keys(ma), ...Object.keys(mb)])];
  body.replaceChildren(...keys.map((k) => {
    const d = k in ma && k in mb ? ma[k] - mb[k] : null;
    return el("tr", {},
      el("td", { text: k }),
      el("td", { text: k in ma ? fmtNum(ma[k]) : "-" }),
      el("td", { text: k in mb ? fmtNum(mb[k]) : "-" }),
      el("td", { class: d === null || d === 0 ? "" : d > 0 ? "delta-up" : "delta-down", text: d === null ? "-" : `${d > 0 ? "+" : ""}${fmtNum(d)}` }));
  }));
}

// FT-43: a fine-tuned classifier runs on a detector's boxes in Studio. List the project's
// detectors so the user sees, and can change, which one the classifier is paired with.
let detectorRequest = 0;
let detectorKey = null;
async function loadStudioDetectors(force = false) {
  const label = $("#studio-detector-label");
  const sel = $("#studio-detector");
  const project = $("#studio-deploy-project").value.trim();
  // Only a classifier runs on a detector's boxes; follow the selected model, not the current dataset.
  const model = state.models.find((x) => x.id === state.modelId);
  const pipeline = model && state.pipelines.find((x) => x.id === model.pipeline);
  const show = Boolean(state.studio.configured && pipeline && pipeline.task === "classification" && project);
  const key = show ? `${project}|${model.id}` : null;
  if (!force && key === detectorKey) return;
  detectorKey = key;
  label.hidden = !show;
  state.studioDetectors = [];
  sel.replaceChildren(new Option(show ? "Loading detectors…" : "Current detector", ""));
  renderDetectorNote();
  if (!show) return;
  const ticket = ++detectorRequest;
  let detectors;
  try {
    detectors = (await api("GET", `/v1/connectors/studio/detectors?project=${encodeURIComponent(project)}`)).detectors;
  } catch (err) {
    if (ticket !== detectorRequest) return;
    sel.replaceChildren(new Option("Current detector (list unavailable)", ""));
    return;
  }
  if (ticket !== detectorRequest) return;
  state.studioDetectors = detectors;
  sel.replaceChildren(...detectors.map((d) => {
    const tags = [d.fine_tuned ? "fine-tuned" : null, d.current ? "current" : null].filter(Boolean);
    const text = `${d.name}${tags.length ? ` (${tags.join(", ")})` : ""}${d.usable ? "" : ` — unavailable: ${d.reason || "cannot run here"}`}`;
    const o = new Option(text, String(d.pipeline_id));
    o.disabled = !d.usable;
    return o;
  }));
  const current = detectors.find((d) => d.current && d.usable) || detectors.find((d) => d.usable);
  if (!current) sel.prepend(new Option("Current detector", ""));
  sel.value = current ? String(current.pipeline_id) : "";
  renderDetectorNote();
}

function chosenDetector() {
  const id = $("#studio-detector").value;
  return (state.studioDetectors || []).find((d) => String(d.pipeline_id) === id) || null;
}

function renderDetectorNote() {
  const note = $("#studio-detector-note");
  const d = $("#studio-detector-label").hidden ? null : chosenDetector();
  note.hidden = !(d && d.fine_tuned);
  note.textContent = d && d.fine_tuned
    ? `${d.name} is a fine-tuned detector. The classifier only sees the boxes it finds, so animals it was not trained on may be missed.`
    : "";
}

async function activate(consumer) {
  let project;
  if (consumer === "studio") {
    project = selectedStudioProject("#studio-deploy-project");
  } else {
    project = $("#deploy-project").value.trim();
    if (!project) throw new Error("Enter the project the model should serve.");
  }
  const m = state.models.find((x) => x.id === state.modelId);
  const c = m && m.stock_comparison;
  let confirmed = false;
  if (c && c.worse && c.enforced !== false) {
    if (!confirm(`${stockText(c)} Activate this fine-tuned model anyway?`)) return;
    confirmed = true;
  }
  const url = `/v1/models/${state.modelId}/activate`;
  const modelId = state.modelId;
  const button = $(consumer === "studio" ? "#activate-studio" : "#activate-local");
  const message = consumer === "studio" ? "Sending the model to Studio…" : "Activating the model…";
  const body = { consumer, project };
  const detector = consumer === "studio" && !$("#studio-detector-label").hidden ? chosenDetector() : null;
  if (detector) body.detector_pipeline_id = Number(detector.pipeline_id);
  const send = (ok) => withBusy(message, button, () => api("POST", url, { ...body, confirm: ok }));
  let result;
  try {
    result = await send(confirmed);
  } catch (err) {
    // The evaluation may have changed since the page loaded: the service asks to confirm.
    if (!(err instanceof ApiError && err.status === 409 && err.detail && err.detail.code === "stock_model_better")) throw err;
    if (!confirm(err.detail.message)) { await loadModels(); return; }
    result = await send(true);
  } finally {
    button.disabled = false;
  }
  await loadModels();
  if (consumer !== "studio") {
    toast(`${modelLabel(modelId)} is now active for ${consumer}/${project}`);
    return;
  }
  const shown = studioProjectName(project);
  const install = (result && result.studio_install) || "";
  // The pushed model is now in that project's list; refresh it if it is on screen.
  if ($("#studio-project").value === project) listStudioModels().catch((err) => showError(err.message));
  if (install.startsWith("unavailable:")) {
    showError(`${modelLabel(modelId)} was sent to Studio project "${shown}", but Studio cannot run it: ${install.slice("unavailable:".length).trim()}`);
  } else if (install === "pending" || install === "installing") {
    toast(`${modelLabel(modelId)} sent to Studio project "${shown}"; Studio is installing it.`);
  } else {
    toast(`${modelLabel(modelId)} sent to Studio project "${shown}"; Studio install status: ${install || "unknown"}.`);
  }
}

// FT-16: stock classifier packages a classifier job can be compared with; the first is preselected.
async function loadStockModels() {
  try {
    state.stockModels = (await api("GET", "/v1/stock-models")).models;
  } catch (err) {
    state.stockModels = [];
  }
  const sel = $("#stock-model");
  sel.replaceChildren(new Option("No comparison", ""));
  for (const m of state.stockModels) {
    sel.append(new Option(`${m.name} (${m.classes} classes)`, m.id));
  }
  sel.value = state.stockModels.length ? state.stockModels[0].id : "";
}

// ------------------------------------------------------------------ boot

async function boot() {
  closeStream();
  if (!(await loadMe())) return;
  const [pipelines, studio] = await Promise.all([
    api("GET", "/v1/pipelines"),
    api("GET", "/v1/connectors/studio"),
  ]);
  state.pipelines = pipelines.pipelines;
  state.studio = studio;
  $("#browse-catalog").textContent = studio.configured ? "Browse Studio models" : "Browse Model Zoo";
  await loadStockModels();
  $("#studio-panel").hidden = !studio.configured;
  $("#studio-scope-panel").hidden = !studio.configured;
  $(".scope-picker").classList.toggle("with-studio", Boolean(studio.configured));
  $("#studio-url").textContent = studio.configured ? `Connected to ${studio.url}` : "";
  if (studio.configured) {
    renderStudioScope();
    loadStudioProjects().catch((err) => showError(err.message));
  }
  renderBaseModel();
  loadLocalBaseModels().catch((err) => showError(err.message));
  // Opened from Studio for a project already loaded here: go to its workspace.
  const linked = launch.project && !launch.load
    ? (await api("GET", "/v1/workspaces")).workspaces.find((w) => w.studio_project === launch.project)
    : null;
  await loadWorkspaces(linked ? linked.id : undefined);
  if (!launch.project && location.hash === "#catalog-title") {
    await showPage("models");
    $("#catalog-title").focus();
    $("#catalog-title").scrollIntoView();
  } else {
    showPage(state.page === "login" ? "data" : state.page);
  }
  if (launch.load) {
    $("#studio-uptodate").hidden = true;
    toast("Loading the Studio project's data…");
    await loadLaunchProject();
  } else {
    refreshStudioStatus();
  }
}

function bind() {
  $("#error-close").addEventListener("click", clearError);
  $("#login-form").addEventListener("submit", guarded(login));
  $("#logout-btn").addEventListener("click", guarded(logout));
  $$("[data-page]").forEach((b) => b.addEventListener("click", () => {
    setNavOpen(false);
    showPage(b.dataset.page);
  }));
  $("#sidebar-toggle").addEventListener("click", () => {
    setNavOpen(!$(".sidebar").classList.contains("open"));
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && $(".sidebar").classList.contains("open")) {
      setNavOpen(false);
      $("#sidebar-toggle").focus();
    }
  });
  matchMedia("(max-width: 820px)").addEventListener("change", () => setNavOpen(false));
  const wsFold = $("#workspace-panel");
  wsFold.open = localStorage.getItem(WORKSPACE_FOLD_KEY) !== "1";
  // Saved on the summary click, before the default action flips `open`: the `toggle` event is
  // queued asynchronously, so a reload right after a click could lose the fold state.
  wsFold.querySelector("summary").addEventListener("click", () =>
    localStorage.setItem(WORKSPACE_FOLD_KEY, wsFold.open ? "1" : "0"));
  $("#workspace-select").addEventListener("change", guarded(async (e) => {
    state.workspaceId = Number(e.target.value) || null;
    remember(WORKSPACE_KEY, state.workspaceId);
    state.datasetId = null;
    await loadDatasets();
    refreshStudioStatus();
  }));
  $("#dataset-select").addEventListener("change", guarded(async (e) => {
    state.datasetId = Number(e.target.value) || null;
    remember(DATASET_KEY, state.datasetId);
    await loadDataset();
    refreshStudioStatus();
  }));
  for (const [kind, create] of [["workspace", createWorkspace], ["dataset", createDataset]]) {
    $(`#show-new-${kind}`).addEventListener("click", () => toggleCreate(kind, true));
    $(`#cancel-new-${kind}`).addEventListener("click", () => toggleCreate(kind, false));
    const submit = guarded(create);
    $(`#new-${kind}-form`).addEventListener("submit", (e) => { e.preventDefault(); submit(); });
    $(`#new-${kind}-form`).addEventListener("keydown", (e) => {
      if (e.key === "Escape") toggleCreate(kind, false);
    });
  }
  $("#delete-workspace").addEventListener("click", guarded(deleteWorkspace));
  $("#delete-dataset").addEventListener("click", guarded(deleteDataset));
  $("#dataset-whole-image").addEventListener("change", guarded(onWholeImageToggle));
  $("#new-dataset-task").addEventListener("change", renderNewDatasetOptions);
  $("#assist-speciesnet").addEventListener("change", onAssistModelChange);
  $("#pipeline-select").addEventListener("change", renderPipelineSummary);
  $("#upload-btn").addEventListener("click", guarded(() => pickFile("#upload-files")));
  $("#upload-files").addEventListener("change", guarded(uploadImages));
  $("#import-annotations-btn").addEventListener("click", guarded(() => pickFile("#annotations-file")));
  $("#annotations-file").addEventListener("change", guarded(importAnnotations));
  $("#annotation-format").addEventListener("change", renderLabelFormat);
  $("#assist-btn").addEventListener("click", guarded(openAssistConfirm));
  $("#assist-confirm-btn").addEventListener("click", guarded(runAssist));
  $("#assist-cancel-btn").addEventListener("click", hideAssistConfirm);
  $("#assist-model").addEventListener("change", onAssistModelChange);
  $("#assist-enable").addEventListener("change", guarded(onAssistToggle));
  for (const node of [$("#assist-overwrite"), ...document.querySelectorAll('input[name="assist-source"]')]) {
    node.addEventListener("change", () => { hideAssistConfirm(); renderAssistWarnings(); });
  }
  const stage = $("#box-stage");
  stage.addEventListener("pointerdown", onStageDown);
  stage.addEventListener("pointermove", onStageMove);
  stage.addEventListener("pointerup", onStageUp);
  stage.addEventListener("pointercancel", onStageUp);
  $("#box-editor").addEventListener("keydown", onEditorKey);
  $("#box-editor").addEventListener("close", () => { editor.item = null; editor.drag = null; });
  $("#box-label").addEventListener("change", onBoxLabelChange);
  $("#box-new-label").addEventListener("keydown", onNewLabelKey);
  $("#box-new-label-add").addEventListener("click", addNewLabel);
  $("#box-delete").addEventListener("click", deleteSelectedBox);
  $("#box-cancel").addEventListener("click", closeBoxEditor);
  $("#box-save").addEventListener("click", saveBoxes);
  $("#batch-auto").addEventListener("change", () => { $("#batch-size").disabled = $("#batch-auto").checked; });
  $("#early-stop").addEventListener("change", () => { $("#patience").disabled = !$("#early-stop").checked; });
  $("#start-weights").addEventListener("change", () => { syncImgszToWeights(); renderStartWeights(); });
  $("#studio-import-btn").addEventListener("click", guarded(studioImport));
  $("#studio-reviewed-only").addEventListener("change", refreshStudioStatus);
  $("#studio-model-clear").addEventListener("click", clearBaseModel);
  $("#local-base-model-clear").addEventListener("click", clearBaseModel);
  $("#local-base-model-select").addEventListener("change", chooseLocalBaseModel);
  $("#local-base-model-upload").addEventListener("click", guarded(uploadLocalBaseModel));
  $("#studio-projects-refresh").addEventListener("click", guarded(async () => {
    await loadStudioProjects();
    // Reload also picks up models pushed to the chosen project since it was listed.
    if ($("#studio-project").value) await listStudioModels();
  }));
  $("#studio-project").addEventListener("change", renderStudioScope);
  $("#studio-model-select").addEventListener("change", renderStudioModelChoice);
  $("#studio-model-use").addEventListener("click", guarded(importStudioModel));
  $("#example-filter").addEventListener("change", guarded(loadDataset));
  $("#example-grid").addEventListener("click", guarded(onExampleAction));
  $("#example-grid").addEventListener("click", onPhotoOpen);
  $("#example-grid").addEventListener("keydown", onPhotoOpen);
  const viewerDlg = $("#image-viewer");
  $("#image-viewer-img").addEventListener("load", fitViewerImage);
  window.addEventListener("resize", () => { if (viewerDlg.open) fitViewerImage(); });
  $("#image-viewer-prev").addEventListener("click", () => showViewerItem(viewer.index - 1));
  $("#image-viewer-next").addEventListener("click", () => showViewerItem(viewer.index + 1));
  $("#image-viewer-close").addEventListener("click", () => viewerDlg.close());
  viewerDlg.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { e.preventDefault(); showViewerItem(viewer.index - 1); }
    if (e.key === "ArrowRight") { e.preventDefault(); showViewerItem(viewer.index + 1); }
  });
  viewerDlg.addEventListener("click", (e) => { if (e.target === viewerDlg) viewerDlg.close(); });
  $("#create-snapshot").addEventListener("click", guarded(() => createSnapshot()));
  $("#crop-prep-assist").addEventListener("click", guarded(cropPrepAssist));
  $("#crop-prep-draw").addEventListener("click", guarded(cropPrepDraw));
  $("#crop-prep-exclude").addEventListener("click", guarded(cropPrepExclude));
  $("#crop-prep-show-all").addEventListener("click", guarded(cropPrepShowAll));
  $("#parent-model-select").addEventListener("change", guarded(chooseParentModel));
  $("#model-reuse").addEventListener("click", guarded(reuseModelFromDetail));
  $("#catalog-refresh").addEventListener("click", guarded(loadCatalog));
  $("#catalog-list").addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-catalog-download]");
    if (btn) guarded(() => downloadCatalogModel(btn))();
  });
  $("#start-training").addEventListener("click", guarded(startTraining));
  $("#job-history").addEventListener("click", guarded(async (e) => {
    const row = e.target.closest("[data-job-id]");
    if (row) await selectJob(Number(row.dataset.jobId));
  }));
  $("#cancel-job").addEventListener("click", guarded(cancelJob));
  $("#retry-job").addEventListener("click", guarded(retryJob));
  $("#model-list").addEventListener("click", (e) => {
    const b = e.target.closest("[data-model-id]");
    if (!b) return;
    state.modelId = Number(b.dataset.modelId);
    renderModelList();
    renderModel();
  });
  $("#compare-select").addEventListener("change", renderCompare);
  $("#show-rename-model").addEventListener("click", () => toggleRename(true));
  $("#cancel-rename-model").addEventListener("click", () => toggleRename(false));
  const saveName = guarded(renameModel);
  $("#rename-model-form").addEventListener("submit", (e) => { e.preventDefault(); saveName(); });
  $("#rename-model-form").addEventListener("keydown", (e) => { if (e.key === "Escape") toggleRename(false); });
  $("#activate-local").addEventListener("click", guarded(() => activate("local")));
  $("#activate-studio").addEventListener("click", guarded(() => activate("studio")));
  $("#studio-deploy-project").addEventListener("change", () => loadStudioDetectors(true));
  $("#studio-detector").addEventListener("change", renderDetectorNote);
}

document.addEventListener("DOMContentLoaded", () => {
  bind();
  guarded(boot)();
});
