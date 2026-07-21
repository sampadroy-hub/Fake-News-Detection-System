from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-fake-news-detection"
)

def predict_bert(text):
    result = classifier(text)[0]

    label = result['label']
    confidence = result['score'] * 100

    if label == "FAKE":
        return "Fake ❌", round(confidence, 2)
    else:
        return "Real ✅", round(confidence, 2)