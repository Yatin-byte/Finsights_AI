import pandas as pd
from textblob import TextBlob
from nrclex import NRCLex

# Load data
data = pd.read_csv("news_data.csv")

# Sentiment detection
def analyze_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Emotion detection
def detect_emotion(text):
    emotion = NRCLex(text)
    if emotion.top_emotions:
        return emotion.top_emotions[0][0]
    return "Neutral"

data["Sentiment"] = data["Headline"].apply(analyze_sentiment)
data["Emotion"] = data["Headline"].apply(detect_emotion)

# Predict market trend based on sentiment
data["Predicted_Trend"] = data["Sentiment"].apply(
    lambda s: "Up" if s == "Positive" else ("Down" if s == "Negative" else "Neutral")
)

data.to_csv("analyzed_data.csv", index=False)
print("✅ Analysis complete! Results saved to analyzed_data.csv")
print(data)
