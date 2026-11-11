"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const state = {
  meta: null,
  mode: "recommend",
};

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
  pill.textContent = label;
  pill.className = `pill ${ok ? "pill-ok" : "pill-warn"}`;
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
  title.textContent = `${profile.n_apps} apps in January`;
  const note = document.createElement("p");
  note.textContent = "Most-engaged apps in the training window:";
  const list = document.createElement("div");
  list.className = "profile-apps";
  profile.top_apps.slice(0, 6).forEach((item) => {
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
  $("#output-eyebrow").textContent = payload.strategy === "personalized" ? "Personalized ranking" : "Popularity fallback";
  $("#output-title").textContent = payload.strategy === "personalized"
    ? "Predicted next apps"
    : "Cold-start recommendations";
  $("#latency").textContent = `${Number(payload.latency_ms).toFixed(1)} ms · ${payload.model}`;
  if (!payload.items.length) {
    showEmpty("No unseen catalogue apps remain for this panelist.");
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
    const width = 18 + 82 * ((Number(item.score) - low) / span);
    $(".score-fill", node).style.width = `${Math.max(4, Math.min(100, width))}%`;
    const reasons = $(".reasons", node);
    const items = item.explanations && item.explanations.length
      ? item.explanations.map((explanation) => explanation.reason)
      : ["popular with the January cohort"];
    items.slice(0, 2).forEach((reason) => {
      const li = document.createElement("li");
      li.textContent = reason;
      reasons.append(li);
    });
    output.append(node);
  });
}

async function recommend(userId) {
  if (!userId) return;
  const button = $("#recommend-button");
  button.disabled = true;
  $("#output-eyebrow").textContent = "Scoring";
  $("#output-title").textContent = "Ranking the catalogue…";
  try {
    const k = $("#k-select").value;
    const explain = $("#explain-toggle").checked;
    const diversify = $("#diversify-toggle").checked;
    const query = new URLSearchParams({ k, explain: String(explain), diversify: String(diversify) });
    const [payload, profile] = await Promise.all([
      api(`/v1/users/${encodeURIComponent(userId)}/recommendations?${query}`),
      api(`/v1/users/${encodeURIComponent(userId)}`).catch(() => null),
    ]);
    renderRecommendations(payload);
    renderProfile(profile);
  } catch (error) {
    $("#output-title").textContent = "Could not score this panelist";
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
    const payload = await api(`/v1/apps/similar?${new URLSearchParams({ item_id: itemId, k: "10" })}`);
    output.replaceChildren();
    if (!payload.items.length) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "No similar apps found in the January catalogue.";
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
      note.textContent = "co-usage similarity in January";
      $(".reasons", node).append(note);
      output.append(node);
    });
  } catch (error) {
    output.replaceChildren();
    const message = document.createElement("p");
    message.className = "muted";
    message.textContent = error.message;
    output.append(message);
  }
}

function renderMetricsTable(meta) {
  const body = $("#metrics-table tbody");
  body.replaceChildren();
  const rows = meta.benchmark && Array.isArray(meta.benchmark) && meta.benchmark.length
    ? meta.benchmark
    : [{
        model: meta.model.model_type,
        "ndcg@10": meta.metrics["ndcg@10"],
        "recall@10": meta.metrics["recall@10"],
        "coverage@10": meta.metrics["coverage@10"],
      }];
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const cells = [
      row.model || row.name || "active",
      formatMetric(row["ndcg@10"]),
      formatMetric(row["recall@10"]),
      formatMetric(row["coverage@10"], 3),
    ];
    cells.forEach((value, index) => {
      const td = document.createElement("td");
      td.textContent = value;
      if (index > 0) td.className = "mono";
      tr.append(td);
    });
    body.append(tr);
  });
}

function applyMeta(meta) {
  state.meta = meta;
  const stats = meta.dataset.stats || {};
  $("#kpi-users").textContent = formatNumber(meta.dataset.n_users || stats.n_users);
  $("#kpi-items").textContent = formatNumber(meta.dataset.n_items || stats.n_items);
  $("#kpi-interactions").textContent = formatNumber(stats.n_train_interactions);
  $("#kpi-ndcg").textContent = formatMetric(meta.metrics["ndcg@10"]);
  $("#kpi-model").textContent = meta.model.model_type;
  $("#meta-model").textContent = `${meta.model.model_type} · ${meta.model.params ? JSON.stringify(meta.model.params) : ""}`;
  $("#meta-fingerprint").textContent = meta.dataset.fingerprint || "—";
  $("#meta-artifact").textContent = meta.artifact.fingerprint || "—";
  renderMetricsTable(meta);
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
    if (payload.items.length) $("#item-input").value = payload.items[0].item_id;
  } catch (_) { /* non-critical */ }
}

function wireTabs() {
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const mode = tab.dataset.mode;
      state.mode = mode;
      $$(".tab").forEach((other) => {
        other.classList.toggle("is-active", other === tab);
        other.setAttribute("aria-selected", String(other === tab));
      });
      $("#mode-recommend").hidden = mode !== "recommend";
      $("#mode-similar").hidden = mode !== "similar";
      if (mode === "similar") {
        $("#output-eyebrow").textContent = "Similarity search";
        $("#output-title").textContent = "Apps with shared audiences";
        $("#latency").textContent = "";
        showEmpty("Choose an app on the left to inspect its nearest neighbours.");
      }
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
  try {
    const [meta, version] = await Promise.all([api("/v1/meta"), api("/version")]);
    applyMeta(meta);
    $("#footer-version").textContent = `v${version.version}`;
    setStatus(true, `${meta.model.model_type} loaded`);
    loadPopularApps();
    recommend($("#user-input").value.trim());
  } catch (error) {
    setStatus(false, "model unavailable");
    showEmpty(`The API is running but no artefact is loaded. ${error.message}`);
  }
}

document.addEventListener("DOMContentLoaded", init);
