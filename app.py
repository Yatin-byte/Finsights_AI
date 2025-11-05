import streamlit as st
import pandas as pd
from textblob import TextBlob
from nrclex import NRCLex
from GoogleNews import GoogleNews
import plotly.express as px

# ---------------------------
# 🎨 APP CONFIGURATION
# ---------------------------
st.set_page_config(page_title="Finsights AI", layout="wide")

# ---------------------------
# 🌈 GRADIENT HEADER
# ---------------------------
st.markdown("""
    <style>
    .gradient-text {
        font-size: 38px;
        font-weight: 700;
        background: linear-gradient(90deg, #007bff, #00ffcc, #ff66ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        animation: move 6s infinite alternate;
    }
    @keyframes move {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    </style>
    <h1 class="gradient-text">💹 Finsights AI – Market Mood Analyzer</h1>
""", unsafe_allow_html=True)

st.write("### An AI-powered system that reads the market’s emotions through financial news.")

# ---------------------------
# 🌓 THEME TOGGLE
# ---------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

toggle = st.toggle("🌙 Dark Mode", value=False)
st.session_state.theme = "dark" if toggle else "light"
theme_template = "plotly_dark" if st.session_state.theme == "dark" else "plotly_white"

# ---------------------------
# 📰 FETCH LIVE NEWS
# ---------------------------
def fetch_live_news(keyword="stock market", pages=1):
    googlenews = GoogleNews(lang='en')
    googlenews.search(keyword)
    all_news = []
    for i in range(pages):
        googlenews.getpage(i)
        all_news.extend(googlenews.result())
    df = pd.DataFrame(all_news)[["date", "title", "media", "link"]]
    df.rename(columns={"title": "Headline", "media": "Source", "date": "Date", "link": "URL"}, inplace=True)
    df["Company"] = keyword
    return df

st.sidebar.header("📡 Live News Settings")
company = st.sidebar.text_input("Enter Company or Keyword", "Reliance Industries")
pages = st.sidebar.slider("Number of News Pages", 1, 3, 1)

if st.sidebar.button("🔍 Fetch Latest News"):
    with st.spinner("Fetching latest news..."):
        news_df = fetch_live_news(company, pages)

        # --- Financial News Filter ---
        FINANCIAL_SOURCES = [
            "moneycontrol.com", "economictimes.indiatimes.com", "business-standard.com",
            "bloomberg.com", "reuters.com", "cnbctv18.com", "livemint.com",
            "ndtvprofit.com", "financialexpress.com", "investing.com",
            "marketwatch.com", "forbes.com", "finance.yahoo.com"
        ]

        # Keep only financial sources
        filtered_df = news_df[news_df["URL"].apply(
            lambda x: any(domain in str(x) for domain in FINANCIAL_SOURCES)
        )]

        if filtered_df.empty:
            st.warning("⚠️ No financial news found for this topic. Try another company or keyword.")
        else:
            filtered_df.to_csv("news_data.csv", index=False)
            st.success(f"✅ Fetched {len(filtered_df)} financial news articles for '{company}'")
            st.dataframe(filtered_df.head())

# ---------------------------
# 🧠 SENTIMENT & EMOTION ANALYSIS
# ---------------------------
def analyze_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

def detect_emotion(text):
    emotion = NRCLex(text)
    if emotion.top_emotions:
        return emotion.top_emotions[0][0]
    return "Neutral"

if st.sidebar.button("🧩 Run Market Mood Analysis"):
    try:
        data = pd.read_csv("news_data.csv")

        if data.empty:
            st.warning("⚠️ No financial news available to analyze.")
        else:
            data["Sentiment"] = data["Headline"].apply(analyze_sentiment)
            data["Emotion"] = data["Headline"].apply(detect_emotion)
            data["Predicted_Trend"] = data["Sentiment"].apply(
                lambda s: "Up" if s == "Positive" else ("Down" if s == "Negative" else "Neutral")
            )

            data.to_csv("analyzed_data.csv", index=False)
            st.success("✅ Analysis complete! Results saved to analyzed_data.csv")
            st.dataframe(data.head())

            # ---------------------------
            # 📊 VISUALIZATIONS
            # ---------------------------
            st.subheader("📊 Sentiment Distribution")
            sent_count = data["Sentiment"].value_counts().reset_index()
            sent_count.columns = ["Sentiment", "Count"]
            fig1 = px.bar(sent_count, x="Sentiment", y="Count", color="Sentiment",
                          title="Overall Sentiment Count", template=theme_template)
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("💬 Emotion Distribution")
            emo_count = data["Emotion"].value_counts().reset_index()
            emo_count.columns = ["Emotion", "Count"]
            fig2 = px.pie(emo_count, values="Count", names="Emotion",
                          title="Emotional Tone in Market News", template=theme_template)
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("📈 Predicted Market Trends")
            trend_count = data["Predicted_Trend"].value_counts().reset_index()
            trend_count.columns = ["Trend", "Count"]
            fig3 = px.bar(trend_count, x="Trend", y="Count", color="Trend",
                          text="Count", title="Market Trend Forecast", template=theme_template)
            fig3.update_traces(textposition="outside")
            st.plotly_chart(fig3, use_container_width=True)

            # ---------------------------
            # 💹 INVESTMENT DECISION SECTION
            # ---------------------------
            st.subheader("💹 Investment Suggestion")

            positive = len(data[data["Sentiment"] == "Positive"])
            negative = len(data[data["Sentiment"] == "Negative"])
            total = len(data)

            if total > 0:
                investment_score = (positive - negative) / total

                if investment_score > 0.3:
                    st.success(f"✅ Market Mood: **Bullish** ({investment_score:.2f}) → Suggested Action: **BUY** 🟢")
                elif investment_score < -0.3:
                    st.error(f"🚨 Market Mood: **Bearish** ({investment_score:.2f}) → Suggested Action: **SELL** 🔴")
                else:
                    st.warning(f"⚖️ Market Mood: **Neutral** ({investment_score:.2f}) → Suggested Action: **HOLD** 🟡")
            else:
                st.info("No sufficient news data for investment analysis.")

    except FileNotFoundError:
        st.error("❌ Please fetch live news first before running analysis.")

# ---------------------------
# 📎 FOOTER
# ---------------------------
st.markdown("---")
st.markdown("🧠 Built by **Yatin Kumar** | KIET Group of Institutions | Powered by AI 💡")
