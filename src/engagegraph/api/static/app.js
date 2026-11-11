"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const state = {
  meta: null,
  mode: "recommend",
  domain: {
    user_label: "user",
    user_label_plural: "users",
    item_label: "item",
    item_label_plural: "items",
    interaction_label: "interaction",
    catalog_label: "catalogue",
    task_name: "next-period new-item adoption ranking",
    train_period_label: "training period",
    test_period_label: "test period",
    tagline: "A reproducible pipeline for leakage-free temporal splits and auditable recommendations.",
  },
};

function titleCase(value) {
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : value;
}

function applyDomain(meta) {
  const domain = { ...state.domain, ...(meta.domain || {}) };
  state.domain = domain;
  const itemPlural = domain.item_label_plural;
  const userPlural = domain.user_label_plural;
  setText("hero-eyebrow", `${domain.task_name} · ${domain.train_period_label} → ${domain.test_period_label}`);
  setText("band-period", `${domain.train_period_label.toUpperCase()} → ${domain.test_period_label.toUpperCase()}`);
  setText("hero-item", domain.item_label);
  setText("hero-copy", domain.tagline);
  setText("kpi-users-label", titleCase(userPlural));
  setText("kpi-items-label", titleCase(domain.catalog_label));
  setText("kpi-interactions-label", titleCase(domain.interaction_label) + "s");
  setText("kpi-users-note", `${domain.train_period_label} cohort`);
  const minSupport = meta.dataset?.stats?.catalog_item_support_min ?? 10;
  setText("kpi-items-note", `${itemPlural} with ≥${minSupport} ${userPlural}`);
  setText("kpi-interactions-note", `${domain.user_label}–${domain.item_label} pairs`);
  setText("tab-similar-label", `Similar ${itemPlural}`);
  setText("user-field-label", `${titleCase(domain.user_label)} ID`);
  setText("user-hint", `Pick a sample ${domain.user_label} or paste any ID from ${domain.train_period_label}.`);
  setText("item-field-label", `${titleCase(domain.item_label)} name`);
  setText("similar-hint", `Similarity is cosine over ${domain.train_period_label} co-usage or the active model's item geometry.`);
  setText("empty-copy", `Choose a ${domain.user_label} to see ranked ${itemPlural}, scores, and the ${domain.train_period_label} co-usage that supports each recommendation.`);
  setText("meta-task", domain.task_name);
  setText("meta-train", domain.train_period_label);
  setText("meta-test", domain.test_period_label);
  const input = document.getElementById("item-input");
  if (input) input.placeholder = `Start typing ${domain.item_label === "item" ? "an" : "a"} ${domain.item_label}…`;
  const outputTitle = document.getElementById("output-title");
  if (outputTitle && outputTitle.dataset.empty !== "false") outputTitle.textContent = `Waiting for a ${domain.user_label}`;
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return new Intl.NumberFormat("en-US").format(Number(value));
}

function formatMetric(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function escapeSelector(value) {
  return window.CSS && CSS.escape ? CSS.escape(value) : String(value).replace(/[^a-zA-Z0-9_-]/g, "\\$&");
}

async function api(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload.detail) detail = payload.detail;
    } catch (_) { /* keep status text */ }
    throw new Error(detail);
  }
  return response.json();
}

function setStatus(ok, label) {
  const pill = $("#status-pill");
  if (!pill) return;
  pill.textContent = label;
  pill.classList.toggle("is-error", !ok);
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node && value !== undefined && value !== null) node.textContent = value;
}

function setGauge(ndcg) {
  const value = Number(ndcg);
  if (!Number.isFinite(value)) return;
  const max = value > 0.4 ? Math.ceil(value * 10) / 10 : 0.4;
  const progress = Math.max(0.02, Math.min(1, value / max));
  const ring = document.getElementById("gauge-ring");
  if (ring) ring.style.setProperty("--arc", `${(progress * 270).toFixed(2)}deg`);
  setText("gauge-value", value.toFixed(4));
  setText("gauge-scale", `SCALE 0–${max.toFixed(2)}`);
  document.querySelectorAll(".gauge-label").forEach((node, index) => {
    const tick = (max * index) / 4;
    node.textContent = index === 0 ? "0" : tick.toFixed(2).replace(/^0/, "");
  });
}

function setMiniGauge(id, value) {
  const node = document.getElementById(id);
  if (!node) return;
  const numeric = Math.max(0, Math.min(1, Number(value) || 0));
  node.style.setProperty("--v", `${(numeric * 360).toFixed(1)}deg`);
}

function toast(message) {
  const node = $("#toast");
  if (!node) return;
  node.textContent = message;
  node.classList.add("is-visible");
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => node.classList.remove("is-visible"), 1800);
}

function currentMinPercentile() {
  return Number($("#percentile-range")?.value || 0);
}

function updatePercentileLabel() {
  setText("percentile-value", `${Math.round(currentMinPercentile() * 100)}%`);
}

function syncURL() {
  const params = new URLSearchParams();
  if (state.mode === "similar") {
    params.set("mode", "similar");
    const item = $("#item-input").value.trim();
    if (item) params.set("item", item);
  } else {
    const user = $("#user-input").value.trim();
    if (user) params.set("user", user);
    params.set("k", $("#k-select").value);
    params.set("explain", String($("#explain-toggle").checked));
    params.set("diversify", String($("#diversify-toggle").checked));
    const threshold = currentMinPercentile();
    if (threshold > 0) params.set("min_percentile", threshold.toFixed(2));
  }
  history.replaceState(null, "", `${location.pathname}?${params.toString()}`);
}

function setMode(mode) {
  state.mode = mode;
  $$(".tab").forEach((tab) => {
    const active = tab.dataset.mode === mode;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  $("#mode-recommend").hidden = mode !== "recommend";
  $("#mode-similar").hidden = mode !== "similar";
  if (mode === "similar") {
    $("#output-eyebrow").textContent = "Similarity search";
    $("#output-title").textContent = `${titleCase(state.domain.item_label_plural)} with shared audiences`;
    $("#output-title").dataset.empty = "false";
    $("#latency").textContent = "";
    showEmpty(`Choose a ${state.domain.item_label} on the left to inspect its nearest neighbours.`);
  }
}

function goSimilar(itemId) {
  if (!itemId) return;
  $("#item-input").value = itemId;
  setMode("similar");
  similar(itemId);
  syncURL();
}

function showEmpty(message) {
  const output = $("#output");
  output.replaceChildren();
  const wrapper = document.createElement("div");
  wrapper.className = "empty-state";
  const text = document.createElement("p");
  text.textContent = message;
  wrapper.append(text);
  output.append(wrapper);
}

function renderProfile(profile) {
  const container = $("#profile");
  container.replaceChildren();
  if (!profile) { container.hidden = true; return; }
  const title = document.createElement("h3");
  const domain = state.domain;
  const count = profile.n_items ?? profile.n_apps ?? 0;
  const topItems = profile.top_items ?? profile.top_apps ?? [];
  title.textContent = `${count} ${domain.item_label_plural} in ${domain.train_period_label}`;
  const note = document.createElement("p");
  note.textContent = `Most-engaged ${domain.item_label_plural} in the training window:`;
  const list = document.createElement("div");
  list.className = "profile-apps";
  topItems.slice(0, 6).forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "profile-app";
    chip.textContent = item.item_id;
    list.append(chip);
  });
  container.append(title, note, list);
  container.hidden = false;
}

function renderRecommendations(payload) {
  const output = $("#output");
  output.replaceChildren();
  const domain = state.domain;
  const personalized = payload.strategy === "personalized";
  const filtered = payload.strategy === "threshold_filtered";
  $("#output-eyebrow").textContent = personalized
    ? "Personalized ranking"
    : filtered ? "Threshold filter" : "Popularity fallback";
  $("#output-title").textContent = personalized
    ? `Predicted next ${domain.item_label_plural}`
    : filtered ? "No items above the confidence threshold" : "Cold-start recommendations";
  $("#output-title").dataset.empty = "false";
  $("#latency").textContent = `${Number(payload.latency_ms).toFixed(1)} ms · ${payload.model}`;
  if (!payload.items.length) {
    showEmpty(`No unseen ${state.domain.catalog_label} ${state.domain.item_label_plural} remain for this ${state.domain.user_label}.`);
    return;
  }
  const scores = payload.items.map((item) => Number(item.score));
  const high = Math.max(...scores);
  const low = Math.min(...scores);
  const span = Math.max(high - low, 1e-9);
  const template = $("#recommendation-template");
  payload.items.forEach((item) => {
    const node = template.content.firstElementChild.cloneNode(true);
    $(".rank", node).textContent = item.rank;
    $(".app-name", node).textContent = item.item_id;
    $(".score", node).textContent = Number(item.score).toFixed(3);
    const percentile = $(".percentile", node);
    if (percentile) {
      percentile.textContent = item.percentile == null
        ? ""
        : `TOP ${Math.max(0.1, (1 - Number(item.percentile)) * 100).toFixed(1)}%`;
    }
    const width = 18 + 82 * ((Number(item.score) - low) / span);
    $(".score-fill", node).style.width = `${Math.max(4, Math.min(100, width))}%`;
    const reasons = $(".reasons", node);
    const items = item.explanations && item.explanations.length
      ? item.explanations.map((explanation) => explanation.reason)
      : [`popular with the ${state.domain.train_period_label} cohort`];
    items.slice(0, 2).forEach((reason) => {
      const li = document.createElement("li");
      li.textContent = reason;
      reasons.append(li);
    });
    const action = $(".row-action", node);
    if (action) action.addEventListener("click", () => goSimilar(item.item_id));
    output.append(node);
  });
}

async function recommend(userId) {
  if (!userId) return;
  const button = $("#recommend-button");
  button.disabled = true;
  $("#output-eyebrow").textContent = "Scoring";
  $("#output-title").textContent = `Ranking the ${state.domain.catalog_label}…`;
  $("#output-title").dataset.empty = "false";
  try {
    const k = $("#k-select").value;
    const explain = $("#explain-toggle").checked;
    const diversify = $("#diversify-toggle").checked;
    const query = new URLSearchParams({
      k,
      explain: String(explain),
      diversify: String(diversify),
      min_percentile: currentMinPercentile().toFixed(2),
    });
    const [payload, profile] = await Promise.all([
      api(`/v1/users/${encodeURIComponent(userId)}/recommendations?${query}`),
      api(`/v1/users/${encodeURIComponent(userId)}`).catch(() => null),
    ]);
    renderRecommendations(payload);
    renderProfile(profile);
    syncURL();
  } catch (error) {
    $("#output-title").textContent = `Could not score this ${state.domain.user_label}`;
    showEmpty(error.message);
  } finally {
    button.disabled = false;
  }
}

async function similar(itemId) {
  if (!itemId) return;
  const output = $("#similar-result");
  output.replaceChildren();
  const loading = document.createElement("p");
  loading.className = "muted";
  loading.textContent = "Finding neighbours…";
  output.append(loading);
  try {
    const payload = await api(`/v1/items/similar?${new URLSearchParams({ item_id: itemId, k: "10" })}`);
    output.replaceChildren();
    if (!payload.items.length) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = `No similar ${state.domain.item_label_plural} found in the ${state.domain.train_period_label} catalogue.`;
      output.append(empty);
      return;
    }
    const template = $("#recommendation-template");
    payload.items.forEach((item) => {
      const node = template.content.firstElementChild.cloneNode(true);
      $(".rank", node).textContent = item.rank;
      $(".app-name", node).textContent = item.item_id;
      $(".score", node).textContent = Number(item.score).toFixed(3);
      $(".score-fill", node).style.width = `${Math.max(4, Math.min(100, Number(item.score) * 100))}%`;
      $(".reasons", node).replaceChildren();
      const note = document.createElement("li");
      note.textContent = `co-usage similarity in ${state.domain.train_period_label}`;
      $(".reasons", node).append(note);
      const action = $(".row-action", node);
      if (action) action.hidden = true;
      output.append(node);
    });
    syncURL();
  } catch (error) {
    output.replaceChildren();
    const message = document.createElement("p");
    message.className = "muted";
    message.textContent = error.message;
    output.append(message);
  }
}

function applyMeta(meta) {
  state.meta = meta;
  applyDomain(meta);
  const stats = meta.dataset.stats || {};
  const metrics = meta.metrics || {};
  const ndcg = metrics["ndcg@10"];
  setText("band-users", formatNumber(meta.dataset.n_users || stats.n_users));
  setText("band-model", String(meta.model.model_type || "model").toUpperCase());
  setText("kpi-users", formatNumber(meta.dataset.n_users || stats.n_users));
  setText("kpi-items", formatNumber(meta.dataset.n_items || stats.n_items));
  setText("kpi-interactions", formatNumber(stats.n_train_interactions));
  setText("status-cold", formatNumber(stats.cold_start_test_users ?? stats.cold_start_february_users));
  setText("meta-fingerprint", meta.dataset.fingerprint || "—");
  setText("footer-model", `${meta.model.model_type} · ${meta.dataset.fingerprint || ""}`);
  setGauge(ndcg);
  setMiniGauge("mini-hit", metrics["hit_rate@10"]);
  setMiniGauge("mini-recall", metrics["recall@10"]);
  setText("mini-hit-value", ndcg === undefined ? "—" : `${(Number(metrics["hit_rate@10"] || 0) * 100).toFixed(1)}%`);
  setText("mini-recall-value", `${(Number(metrics["recall@10"] || 0) * 100).toFixed(1)}%`);
  setText("coverage-value", `${(Number(metrics["coverage@10"] || 0) * 100).toFixed(1)}%`);
  setText("gauge-status", "LIVE");
  setText("gauge-model", `NDCG@${stats.primary_k || 10}`);
  const samples = $("#sample-users");
  samples.replaceChildren();
  (meta.sample_users || []).forEach((user) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = user;
    chip.addEventListener("click", () => {
      $("#user-input").value = user;
      recommend(user);
    });
    samples.append(chip);
  });
  if (meta.sample_users && meta.sample_users.length) {
    $("#user-input").value = meta.sample_users[0];
  }
}

async function loadPopularApps() {
  try {
    const payload = await api("/v1/popular?k=50");
    const list = $("#popular-apps");
    list.replaceChildren();
    payload.items.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.item_id;
      list.append(option);
    });
    if (payload.items.length && !$("#item-input").value) {
      $("#item-input").value = payload.items[0].item_id;
    }
  } catch (_) { /* non-critical */ }
}

function wireTabs() {
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      setMode(tab.dataset.mode);
      syncURL();
    });
  });
}

async function init() {
  wireTabs();
  $("#recommend-button").addEventListener("click", () => recommend($("#user-input").value.trim()));
  $("#user-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") recommend($("#user-input").value.trim());
  });
  $("#similar-button").addEventListener("click", () => similar($("#item-input").value.trim()));
  $("#item-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") similar($("#item-input").value.trim());
  });
  $("#k-select").addEventListener("change", () => {
    const userId = $("#user-input").value.trim();
    if (userId) recommend(userId);
  });
  $("#percentile-range").addEventListener("input", updatePercentileLabel);
  $("#percentile-range").addEventListener("change", () => {
    const userId = $("#user-input").value.trim();
    if (userId) recommend(userId);
  });
  $("#copy-link").addEventListener("click", async () => {
    syncURL();
    try {
      await navigator.clipboard.writeText(location.href);
      toast("LINK COPIED");
    } catch (_) {
      toast("COPY NOT AVAILABLE");
    }
  });
  try {
    const [meta, version] = await Promise.all([api("/v1/meta"), api("/version")]);
    applyMeta(meta);
    setText("footer-version", `v${version.version}`);
    setStatus(true, "READY");
    loadPopularApps();
    const params = new URLSearchParams(location.search);
    const k = params.get("k");
    if (k) $("#k-select").value = k;
    if (params.get("explain") === "false") $("#explain-toggle").checked = false;
    if (params.get("diversify") === "true") $("#diversify-toggle").checked = true;
    const threshold = Number(params.get("min_percentile") || 0);
    if (Number.isFinite(threshold) && threshold > 0 && threshold < 1) {
      $("#percentile-range").value = threshold.toFixed(2);
    }
    updatePercentileLabel();
    const requestedUser = params.get("user");
    if (requestedUser) $("#user-input").value = requestedUser;
    if (params.get("mode") === "similar" && params.get("item")) {
      $("#item-input").value = params.get("item");
      setMode("similar");
      similar(params.get("item"));
    } else {
      recommend($("#user-input").value.trim());
    }
  } catch (error) {
    setStatus(false, "NO MODEL");
    showEmpty(`The API is running but no artefact is loaded. ${error.message}`);
  }
}

document.addEventListener("DOMContentLoaded", init);
