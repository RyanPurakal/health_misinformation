const claimInput = document.getElementById("claim");
const predictBtn = document.getElementById("predictBtn");
const resultEl = document.getElementById("result");
const errorEl = document.getElementById("error");

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

predictBtn.addEventListener("click", async () => {
  clearError();
  const claim = claimInput.value.trim();
  if (!claim) {
    showError("Please enter a claim.");
    return;
  }

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claim }),
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
const input = document.getElementById("claimInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultCard = document.getElementById("resultCard");
const labelBadge = document.getElementById("labelBadge");
const confidenceText = document.getElementById("confidenceText");
const explanationText = document.getElementById("explanationText");

function classifyClaim(claim) {
  const text = claim.toLowerCase();
  const riskyMarkers = [
    "miracle cure",
    "cures cancer",
    "cure cancer",
    "completely cures",
    "cures all",
    "detox",
    "flush all toxins",
    "secret remedy",
    "doctors don't want you to know",
    "cause autism",
    "prevent every",
  ];

  if (riskyMarkers.some((marker) => text.includes(marker))) {
    return {
      label: "MISINFORMATION",
      confidence: 0.79,
      explanation: "The claim includes common misinformation-style phrasing.",
    };
  }

  return {
    label: "RELIABLE",
    confidence: 0.61,
    explanation: "No strong misinformation markers were detected in this text.",
  };
}

analyzeBtn.addEventListener("click", () => {
  const claim = input.value.trim();
  if (!claim) return;

  const result = classifyClaim(claim);
  const isReliable = result.label === "RELIABLE";

  labelBadge.textContent = result.label;
  labelBadge.className = `badge ${isReliable ? "reliable" : "misinformation"}`;
  confidenceText.textContent = `Confidence: ${result.confidence.toFixed(2)}`;
  explanationText.textContent = result.explanation;
  resultCard.classList.remove("hidden");
});
