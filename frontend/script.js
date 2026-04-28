// Frontend logic for the Health Claim Checker UI.
// Uses FastAPI endpoints for both model metadata and prediction.

const claimInput = document.getElementById("claim");
const articleInput = document.getElementById("articleText");
const predictBtn = document.getElementById("predictBtn");
const resultEl = document.getElementById("result");
const errorEl = document.getElementById("error");
const modelSubtextEl = document.getElementById("modelSubtext");

const labelEl = document.getElementById("label");
const confidenceEl = document.getElementById("confidence");
const misinfoProbEl = document.getElementById("misinfoProb");
const explanationEl = document.getElementById("explanation");

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
}

function clearError() {
  errorEl.classList.add("hidden");
  errorEl.textContent = "";
}

async function loadModelInfo() {
  try {
    const response = await fetch("/api/model-info");
    const data = await response.json();
    if (!response.ok || !data.model_name) {
      throw new Error("Model info unavailable");
    }
    modelSubtextEl.textContent = `Powered by ${data.model_name} (${data.model_path}) \u00b7 ${data.note}`;
  } catch (_err) {
    modelSubtextEl.textContent = "Connected model unknown. Make sure backend is running.";
  }
}

loadModelInfo();

predictBtn.addEventListener("click", async () => {
  clearError();
  const claim = claimInput.value.trim();
  const articleText = articleInput.value.trim();

  if (!claim) {
    showError("Please enter a health claim.");
    return;
  }
  if (!articleText) {
    showError("Article text is required for accurate results. Paste the article body above.");
    return;
  }

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claim, article_text: articleText }),
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      showError(data.error || "Prediction failed.");
      return;
    }

    labelEl.textContent = data.label;
    confidenceEl.textContent = data.confidence;
    misinfoProbEl.textContent = data.misinformation_probability;
    explanationEl.textContent = data.explanation;
    resultEl.classList.remove("hidden");
  } catch (_err) {
    showError("Could not reach backend. Make sure the API server is running.");
  }
});
