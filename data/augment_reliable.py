"""
Augments the training data with:
  1. Well-established RELIABLE health facts (label=1 in original schema → label_bin=0)
  2. Well-known MISINFORMATION claims (label=0 in original schema → label_bin=1)

These cover cases that a fact-checking dataset never includes because nobody
submitted them for review, causing the classifier to default incorrectly.

Run once:
    python3 data/augment_reliable.py

Writes:
    data/train/Augmented Reliable 0000.parquet
    data/train/Augmented Misinfo 0000.parquet
"""

from pathlib import Path
import pandas as pd

TRAIN = Path(__file__).resolve().parent / "train"

# original label=1 → RELIABLE (label_bin=0)
RELIABLE_CLAIMS = [
    # Smoking / tobacco
    ("Smoking causes lung cancer.",
     "Tobacco smoking is the leading cause of lung cancer, responsible for ~85% of cases — one of medicine's most established causal links."),
    ("Cigarette smoking raises the risk of heart attack and stroke.",
     "Smoking damages blood vessels, reduces oxygen delivery, and dramatically increases coronary artery disease and stroke risk."),
    ("Quitting smoking improves health outcomes.",
     "Within years of quitting, ex-smokers' risk of lung cancer and heart disease falls substantially compared to those who continue."),
    ("Secondhand smoke harms non-smokers.",
     "Exposure to secondhand smoke causes lung cancer, heart disease, and respiratory problems in bystanders, including children."),

    # Vaccines
    ("Approved vaccines are safe and effective at preventing infectious diseases.",
     "Decades of clinical trials and surveillance confirm that licensed vaccines are safe and prevent measles, polio, influenza, and other diseases."),
    ("Childhood vaccination reduces child mortality worldwide.",
     "Mass vaccination programs have dramatically reduced deaths from measles, diphtheria, whooping cough, and other diseases."),

    # Exercise
    ("Regular physical exercise reduces the risk of heart disease.",
     "Aerobic exercise strengthens the heart, lowers blood pressure, improves cholesterol, and reduces cardiovascular disease risk."),
    ("Exercise lowers the risk of developing type 2 diabetes.",
     "Physical activity improves insulin sensitivity and glucose metabolism, significantly reducing type 2 diabetes risk."),
    ("Physical activity improves mental health.",
     "Exercise raises serotonin and endorphin levels; clinical studies show it reduces depression and anxiety symptoms."),

    # Handwashing and hygiene
    ("Washing hands with soap prevents the spread of germs.",
     "Hand hygiene is one of the most effective ways to stop transmission of pathogens causing diarrhea and respiratory illness."),
    ("Handwashing reduces respiratory infection transmission.",
     "Regular handwashing cuts transmission of viruses and bacteria responsible for colds, flu, and other respiratory illness."),

    # Diet and nutrition
    ("Eating fruits and vegetables is associated with better health outcomes.",
     "Diets rich in produce are linked to lower rates of heart disease, stroke, cancer, and overall mortality."),
    ("Excessive alcohol consumption damages the liver and raises cancer risk.",
     "Heavy drinking causes liver cirrhosis, cardiovascular problems, and raises the risk of several cancers."),
    ("Obesity increases the risk of type 2 diabetes.",
     "Excess body fat, especially abdominal fat, causes insulin resistance — a primary driver of type 2 diabetes."),

    # Sleep
    ("Chronic sleep deprivation is harmful to health.",
     "Insufficient sleep is linked to obesity, diabetes, cardiovascular disease, impaired immunity, and cognitive decline."),

    # Sun / skin
    ("Excessive sun exposure increases the risk of skin cancer.",
     "UV radiation damages skin-cell DNA and is the primary cause of melanoma and other skin cancers."),
    ("Sunscreen reduces UV skin damage and skin cancer risk.",
     "Regular use of broad-spectrum SPF 30+ sunscreen significantly reduces UV-related damage and cancer risk."),

    # Blood pressure
    ("High blood pressure raises the risk of stroke and heart attack.",
     "Hypertension damages artery walls over time, increasing atherosclerosis, stroke, heart attack, and kidney failure risk."),

    # Cancer
    ("Early cancer detection improves treatment success.",
     "Screening for colorectal, breast, and cervical cancers catches disease at earlier, more treatable stages."),

    # Diabetes management
    ("Controlling blood sugar reduces diabetic complications.",
     "Keeping glucose in target range reduces the risk of nerve damage, kidney disease, and vision loss in diabetics."),
]

# original label=0 → MISINFORMATION (label_bin=1)
MISINFO_CLAIMS = [
    ("Vaccines cause autism.",
     "Dozens of large studies involving millions of children have found no link between any vaccine and autism."),
    ("Drinking bleach or household disinfectants cures COVID-19.",
     "Ingesting bleach or disinfectants is extremely dangerous and can be fatal; they have no antiviral effect in the body."),
    ("5G towers spread COVID-19.",
     "COVID-19 is caused by the SARS-CoV-2 virus, not radio waves. 5G is electromagnetic radiation and cannot carry or transmit viruses."),
    ("Antibiotics cure the common cold and flu.",
     "Colds and flu are caused by viruses; antibiotics only kill bacteria and have no effect on viral infections."),
    ("Vitamin C megadoses prevent or cure the common cold.",
     "Clinical evidence does not support high-dose vitamin C as a prevention or cure for colds; benefits are minimal at best."),
    ("The Earth is flat.",
     "The Earth is an oblate spheroid, confirmed by satellite imagery, physics, GPS, and centuries of scientific observation."),
    ("Eating carrots significantly improves eyesight.",
     "While carrots contain vitamin A needed for normal vision, eating extra carrots does not improve eyesight beyond normal levels."),
    ("Homeopathic remedies are effective medical treatments.",
     "Rigorous clinical trials consistently show homeopathic preparations perform no better than placebo."),
    ("You only use 10% of your brain.",
     "Brain imaging shows virtually all regions are active; there is no large dormant portion waiting to be unlocked."),
    ("Cell phones cause brain cancer.",
     "Large epidemiological studies have not found evidence that mobile phone use causes brain tumors."),
    ("Detox diets and cleanses remove toxins from the body.",
     "The liver and kidneys continuously filter waste; no scientific evidence supports that commercial 'detox' products improve this process."),
    ("Sugar causes hyperactivity in children.",
     "Multiple double-blind studies have found no link between sugar consumption and increased hyperactivity in children."),
]


def make_rows(claims, label_orig, prefix):
    rows = []
    for i, (claim, explanation) in enumerate(claims):
        rows.append({
            "claim_id": f"{prefix}_{i:04d}",
            "claim": claim,
            "date_published": "",
            "explanation": explanation,
            "fact_checkers": "augmented",
            "main_text": "",
            "sources": "",
            "label": label_orig,
            "subjects": "health",
        })
    return rows


TRAIN.mkdir(parents=True, exist_ok=True)

reliable_df = pd.DataFrame(make_rows(RELIABLE_CLAIMS, label_orig=1, prefix="AUG_R"))
reliable_out = TRAIN / "Augmented Reliable 0000.parquet"
reliable_df.to_parquet(reliable_out, index=False)
print(f"Wrote {len(reliable_df)} RELIABLE examples → {reliable_out}")

misinfo_df = pd.DataFrame(make_rows(MISINFO_CLAIMS, label_orig=0, prefix="AUG_M"))
misinfo_out = TRAIN / "Augmented Misinfo 0000.parquet"
misinfo_df.to_parquet(misinfo_out, index=False)
print(f"Wrote {len(misinfo_df)} MISINFORMATION examples → {misinfo_out}")
