# 📊 Pro Market Sentiment Analysis Bot v1.8

A high-performance sentiment analysis engine that decodes **market psychology** by analyzing news headlines in real-time. Built with a focus on institutional-grade metrics, multi-engine AI support, and visual data story-telling.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Dashboard+Preview+placeholder)

## 🎯 Key Features

- **Multi-Engine AI Core**: Pivot between ultra-fast **VADER**, professional **FinBERT** (HuggingFace transformers), and advanced **Gemini Pro** (Large Language Model).
- **Topic Modeling**: Answers *"Why is the market moving?"* by automatically categorizing news into [Regulation], [Macro], [Security], [Tech], and [Market] drivers.
- **Uncertainty Index**: Measures market consensus vs. conflict based on score variance (Standard Deviation).
- **Historical Tracking**: Integrated SQLite database to monitor sentiment trends over time for specific assets.
- **Modern Dashboard**: A premium Streamlit interface with glassmorphism aesthetics, dynamic Gauge charts, and interactive insights.
- **Validation Engine**: Built-in accuracy tester to verify AI performance against manually labeled datasets.

---

## 🛠️ Technical Stack

- **Frontend**: Streamlit (Python-based Web UI)
- **Data Visuals**: Plotly Express & GraphObjects
- **NLP Engines**:
  - `google-genai` (Gemini Pro 1.5)
  - `transformers` (ProsusAI/finbert)
  - `vaderSentiment` (Lexicon-based)
- **Scraping**: BeautifulSoup4 & FeedParser (RSS Feed Aggregation)
- **Database**: SQLite3 (Local historical data)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- A **Gemini API Key** (Optional, required for AI Summaries and Gemini Engine)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/market-sentiment-bot.git
   cd market-sentiment-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (Optional):
   Create a `.env` file or set your key in the environment:
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```

### 🖥️ Running the Application

For the full visual experience (Web Dashboard):
```bash
streamlit run app.py
```

For the Command Line Interface (CLI):
```bash
python sentiment_bot.py --asset crypto --symbol bitcoin --count 50
```

---

## 🔬 Methodology

### Sentiment Score Interpretation
| Class | Compound Score | Market Label |
|-------|----------------|--------------|
| 🟢 Positive | ≥ +0.15 | Bullish |
| 🔴 Negative | ≤ -0.15 | Bearish |
| ⚪ Neutral | -0.15 to +0.15 | Neutral |

### Market Score Distribution
The **Market Score (-100 to +100)** is calculated as the mean compound score across all headlines, normalized to a percentage scale.
- **60 to 100**: Extreme Greed
- **20 to 60**: Bullish
- **-20 to 20**: Neutral
- **-60 to -20**: Bearish
- **-100 to -60**: Extreme Fear

### Topic Categories
The system uses a robust keyword-based categorization to identify the "Catalyst" for price action:
- **Regulation**: SEC filings, legal cases, ETF approvals, policy changes.
- **Macro**: FED decisions, inflation (CPI), interest rates, global economy.
- **Security**: Hacks, scams, security vulnerabilities, exploits.
- **Tech**: Protocol upgrades, fork updates, node developments.
- **Market**: Direct price action, whale moves, liquidations.

---

## 📊 Sample Visuals

````carousel
![Sentiment Gauge](https://via.placeholder.com/400x200?text=Gauge+Chart)
<!-- slide -->
![Topic Distribution](https://via.placeholder.com/400x200?text=Topic+Analysis)
<!-- slide -->
![Keyword Trends](https://via.placeholder.com/400x200?text=Keyword+Cloud)
````

---

## 📋 Folder Structure

```
├── app.py              # Main Streamlit Dashboard UI
├── analyzer.py         # Multi-engine Sentiment Logic & SDKs
├── scraper.py          # News aggregation (Google News, CryptoNews, CoinDesk)
├── database.py         # SQLite trend tracking & persistence
├── sentiment_bot.py    # CLI Entry point
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

---

## 📄 License

MIT License - Full freedom to use for personal or commercial projects.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact
Developed by **Talay** - 2026

*Disclaimer: This tool is for educational and informational purposes only. Trading financial assets involves significant risk. Always perform your own due diligence.*
