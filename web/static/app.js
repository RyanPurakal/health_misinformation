const input = document.getElementById("claim-input");
const submitBtn = document.getElementById("submit-btn");
const clearBtn = document.getElementById("clear-btn");
const resultPanel = document.getElementById("result-panel");
const errorPanel = document.getElementById("error-panel");
const labelBadge = document.getElementById("label-badge");
const confidenceMeter = document.getElementById("confidence-meter");
const confidenceValue = document.getElementById("confidence-value");
const explanationBody = document.getElementById("explanation-body");
const metaDl = document.getElementById("meta-dl");
const errorMessage = document.getElementById("error-message");
const btnLabel = submitBtn.querySelector(".btn-label");
const btnSpinner = submitBtn.querySelector(".btn-spinner");

function hidePanels() {
  resultPanel.hidden = true;
  errorPanel.hidden = true;
}

function showError(msg) {
  hidePanels();
  errorMessage.textContent = msg;
  errorPanel.hidden = false;
}

function labelClass(label) {
  if (label === "RELIABLE") return "reliable";
  if (label === "MISINFORMATION") return "misinformation";
  return "uncertain";
}

function renderMeta(meta) {
  metaDl.innerHTML = "";
  const rows = [
    ["Dataset explanation", meta.used_dataset_explanation ? "Yes" : "No (generic)"],
    ["Semantic similarity", meta.semantic_similarity.toFixed(3)],
    ["Margin vs. 2nd", meta.semantic_margin.toFixed(3)],
    ["Min similarity required", meta.min_similarity.toFixed(2)],
    ["Min margin required", meta.min_margin.toFixed(3)],
    ["URL input", meta.is_url ? "Yes" : "No"],
    ["Low-confidence flag", meta.uncertain ? "Yes" : "No"],
  ];
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    metaDl.append(dt, dd);
  }
}

async function analyze() {
  const text = input.value.trim();
  if (!text) {
    showError("Enter a claim or URL first.");
    return;
  }

  submitBtn.disabled = true;
  btnSpinner.hidden = false;
  hidePanels();

  try {
    const res = await fetch("/api/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      showError(data.detail?.[0]?.msg || data.message || `Request failed (${res.status})`);
      return;
    }

    if (data.error === "fetch_failed") {
      showError(data.message);
      return;
    }
    if (data.error === "empty") {
      showError(data.message);
      return;
    }

    labelBadge.textContent = data.label;
    labelBadge.className = "label-badge " + labelClass(data.label);
    confidenceMeter.value = data.confidence;
    confidenceValue.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    explanationBody.textContent = data.explanation;
    renderMeta(data.meta);
    resultPanel.hidden = false;
  } catch (e) {
    showError(e.message || "Network error. Is the server running?");
  } finally {
    submitBtn.disabled = false;
    btnSpinner.hidden = true;
  }
}

submitBtn.addEventListener("click", analyze);
clearBtn.addEventListener("click", () => {
  input.value = "";
  hidePanels();
  input.focus();
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    analyze();
  }
});
