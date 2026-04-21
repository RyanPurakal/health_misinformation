from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_training_data():
    examples = [
        ("Vaccines cause autism.", 1),
        ("Lemon water cures cancer naturally.", 1),
        ("Detox tea flushes all toxins overnight.", 1),
        ("Doctors do not want you to know this secret remedy.", 1),
        ("A single supplement can prevent every disease.", 1),
        ("Drinking bleach can kill viruses inside your body.", 1),
        ("Garlic alone can replace prescribed blood pressure medication.", 1),
        ("You can reverse type 1 diabetes with detox diets.", 1),
        ("Handwashing reduces spread of infections.", 0),
        ("Vaccines reduce risk of severe disease.", 0),
        ("Exercise supports cardiovascular health.", 0),
        ("Adequate sleep helps immune function.", 0),
        ("Smoking increases risk of lung disease.", 0),
        ("Sunscreen helps reduce skin cancer risk.", 0),
        ("High blood pressure can increase stroke risk.", 0),
        ("Breast milk has immune-protective benefits for infants.", 0),
    ]
    texts = [x[0] for x in examples]
    labels = [x[1] for x in examples]
    return texts, labels


def main():
    texts, labels = build_training_data()
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(texts, labels)

    output_path = Path(__file__).resolve().parent / "model" / "claim_model.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Saved trained model to: {output_path}")


if __name__ == "__main__":
    main()
