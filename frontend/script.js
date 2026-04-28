// Frontend logic for the Health Claim Checker UI.
// Uses FastAPI endpoints for both model metadata and prediction.
// EXAMPLES is the source of truth for the 10 clickable example chips.

const EXAMPLES = [
  {
    label: "MISINFORMATION",
    claim: "Vaccines cause autism.",
    article: "A now-retracted 1998 study by Andrew Wakefield falsely claimed a link between the MMR vaccine and autism. The paper was found to be fraudulent and Wakefield lost his medical license. Since then, dozens of large-scale studies involving millions of children across multiple countries have found no connection between any vaccine and autism. The original study involved only 12 children and its data was manipulated. Major health organizations worldwide, including the CDC, WHO, and the American Academy of Pediatrics, confirm that vaccines do not cause autism. The ingredients in vaccines have been extensively studied and none have been shown to contribute to autism spectrum disorder."
  },
  {
    label: "MISINFORMATION",
    claim: "Drinking bleach or disinfectants can kill the coronavirus inside your body.",
    article: "Health authorities have issued urgent warnings after social media posts claimed that ingesting bleach or household disinfectants could cure COVID-19. The claim is dangerous and false. Bleach and disinfectants are corrosive chemicals designed for use on surfaces only. Ingesting them causes severe chemical burns to the mouth, throat, esophagus, and stomach, and can be fatal. Poison control centers reported a spike in calls after such claims circulated online. The CDC and FDA explicitly state that no disinfectant product is safe or effective when consumed or injected. There is no mechanism by which bleach could target a virus inside the human body without causing catastrophic organ damage."
  },
  {
    label: "MISINFORMATION",
    claim: "5G towers are spreading the coronavirus.",
    article: "The claim that 5G mobile networks are responsible for spreading COVID-19 has been widely circulated on social media and has led to arson attacks on cell towers in several countries. The claim is scientifically impossible. Viruses are biological organisms and cannot travel on radio waves or be transmitted by electromagnetic signals of any kind. COVID-19 spreads through respiratory droplets and aerosols produced when an infected person breathes, talks, coughs, or sneezes. Furthermore, countries without any 5G infrastructure have experienced COVID-19 outbreaks, which disproves any causal link. Telecommunications experts and virologists have unanimously rejected the claim. Spreading this misinformation has real-world consequences including violence against engineers and disruption of critical communications infrastructure."
  },
  {
    label: "MISINFORMATION",
    claim: "Ivermectin is a proven cure for COVID-19.",
    article: "Early in the COVID-19 pandemic, some studies suggested ivermectin, an antiparasitic drug, might have antiviral properties. However, multiple large, well-designed randomized controlled trials have since found that ivermectin does not reduce hospitalizations, death, or duration of illness in COVID-19 patients. The Together Trial, one of the largest studies, enrolled over 3,000 patients and found no benefit. The FDA and WHO do not recommend ivermectin for COVID-19 treatment outside of clinical trials. Many of the early positive studies were found to contain data errors, duplication, or fraud. Despite this, misinformation about ivermectin as a miracle cure spread widely, leading to dangerous self-medication with veterinary formulations and a surge in poison control calls."
  },
  {
    label: "MISINFORMATION",
    claim: "Sunscreen causes cancer.",
    article: "A claim circulating online suggests that the chemicals in sunscreen are carcinogenic and that sunscreen actually causes the cancer it claims to prevent. Dermatologists and cancer researchers strongly reject this claim. The evidence linking ultraviolet radiation from the sun to skin cancer — including melanoma, basal cell carcinoma, and squamous cell carcinoma — is overwhelming and spans decades of research. Sunscreen works by absorbing or reflecting UV rays before they can damage DNA in skin cells. The FDA reviews sunscreen ingredients for safety. While some studies have detected trace amounts of certain UV filters in the bloodstream, no study has shown these quantities cause harm. Avoiding sunscreen due to this misinformation significantly raises skin cancer risk."
  },
  {
    label: "RELIABLE",
    claim: "Smoking causes lung cancer.",
    article: "The causal relationship between cigarette smoking and lung cancer is one of the most thoroughly established findings in medical history. Tobacco smoke contains over 70 known carcinogens, including benzene, formaldehyde, and polycyclic aromatic hydrocarbons, which directly damage the DNA of lung cells. Smokers are 15 to 30 times more likely to develop lung cancer than non-smokers. Lung cancer accounts for more cancer deaths than any other type in the United States, and approximately 85 percent of all lung cancer cases are attributable to smoking. The risk increases with duration and quantity of smoking and decreases progressively after quitting. These findings are supported by epidemiological studies dating back to the 1950s and have been confirmed by the U.S. Surgeon General and health bodies worldwide."
  },
  {
    label: "RELIABLE",
    claim: "Regular exercise reduces the risk of heart disease.",
    article: "Decades of cardiovascular research confirm that regular physical activity is one of the most effective ways to reduce the risk of heart disease. Exercise strengthens the heart muscle, improves circulation, lowers resting blood pressure, raises HDL cholesterol, reduces LDL cholesterol, and helps maintain a healthy body weight — all of which reduce cardiovascular risk. The American Heart Association recommends at least 150 minutes of moderate aerobic activity per week. Studies show that physically active individuals have a 35 percent lower risk of coronary heart disease compared to sedentary individuals. Even moderate activity such as brisk walking has been shown to significantly reduce mortality from cardiovascular causes. The protective effects are observed across all age groups and are independent of other risk factors."
  },
  {
    label: "RELIABLE",
    claim: "High blood pressure increases the risk of stroke.",
    article: "Hypertension, or high blood pressure, is the single most important modifiable risk factor for stroke. When blood pressure is chronically elevated, it damages the walls of arteries over time, making them more susceptible to rupture or blockage. Ischemic stroke, caused by a clot blocking blood flow to the brain, and hemorrhagic stroke, caused by a burst blood vessel, are both strongly associated with high blood pressure. People with hypertension are four to six times more likely to have a stroke than those with normal blood pressure. The risk rises proportionally with blood pressure level — even readings in the high-normal range carry elevated risk. Effective treatment of hypertension with lifestyle changes and medication has been shown to reduce stroke incidence by 35 to 40 percent in clinical trials."
  },
  {
    label: "RELIABLE",
    claim: "Handwashing with soap reduces the spread of infectious diseases.",
    article: "Handwashing with soap is one of the most cost-effective public health interventions known. The mechanical action of scrubbing with soap removes pathogens from the skin surface, including bacteria, viruses, and other microorganisms. Studies show that proper handwashing can reduce diarrheal illness by up to 40 percent and respiratory infections by up to 20 percent. During the COVID-19 pandemic, health authorities emphasized handwashing alongside mask use as a primary prevention tool because SARS-CoV-2 can survive on surfaces and be transferred to the mouth, nose, and eyes by contaminated hands. The WHO and CDC recommend washing hands with soap and water for at least 20 seconds, particularly before eating, after using the restroom, and after being in public spaces."
  },
  {
    label: "RELIABLE",
    claim: "Sleep deprivation impairs cognitive function and memory.",
    article: "Sleep plays a critical role in brain function, and research consistently shows that insufficient sleep impairs cognition in multiple ways. During sleep, the brain consolidates memories, clears metabolic waste through the glymphatic system, and restores neural connections. Even a single night of poor sleep measurably reduces attention, reaction time, decision-making, and working memory. Chronic sleep deprivation is associated with increased risk of Alzheimer's disease, as the clearance of amyloid-beta — a protein associated with the disease — is impaired without adequate sleep. The National Sleep Foundation recommends seven to nine hours of sleep per night for adults. Studies using cognitive assessments have shown that sleep-deprived individuals perform as poorly as those with a blood alcohol level of 0.08 percent, yet they consistently underestimate their own impairment."
  }
];

const claimInput = document.getElementById("claim");
const articleInput = document.getElementById("articleText");
const chipsContainer = document.getElementById("exampleChips");

EXAMPLES.forEach((ex, i) => {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = `chip chip--${ex.label.toLowerCase()}`;
  btn.textContent = ex.claim;
  btn.addEventListener("click", () => {
    claimInput.value = ex.claim;
    articleInput.value = ex.article;
    claimInput.scrollIntoView({ behavior: "smooth", block: "center" });
    clearError();
    resultEl.classList.add("hidden");
  });
  chipsContainer.appendChild(btn);
});

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
