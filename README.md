# Customer Segmentation — RFM Analysis with AI-Powered Personas
 
An end-to-end customer segmentation pipeline that groups online retail customers by purchasing behavior using RFM analysis and K-Means clustering, then uses Google Gemini to generate actionable marketing personas for each segment — all wrapped in an interactive Streamlit dashboard.
 
Built with Python, scikit-learn, Streamlit, and Google Gemini.
 
---
 
## What It Does
 
```
Raw Transactions  →  Clean  →  RFM Features  →  Scale  →  K-Means  →  Clusters
                                                                           ↓
                                                            Gemini  →  Personas
```
 
Each customer is reduced to three numbers — how recently they bought, how often, and how much — then K-Means groups similar customers together. Gemini analyzes each cluster's RFM profile and generates a named persona with a behavioral description and targeted marketing recommendations.
 
---
 
## Features
 
- **RFM Feature Engineering** — Builds Recency, Frequency, and Monetary features from raw transaction data per customer
- **Optimal K Selection** — Evaluates K=2 through K=8 using elbow method (inertia) and silhouette score; auto-selects best K or accepts manual override
- **K-Means Clustering** — Fits and labels customer segments with reproducible results (random_state=42)
- **PCA Visualization** — Projects 3D RFM clusters into 2D scatter plot with variance explained on each axis
- **AI Persona Generation** — Gemini generates a persona name, behavioral profile, and 3 marketing recommendations per cluster, informed by median RFM benchmarks
- **Streamlit Dashboard** — Three-tab interface: Data Overview, Cluster Results, and AI Personas
- **Cluster Comparison** — Side-by-side comparison of any two clusters directly in the dashboard
- **Try Different K** — Re-run segmentation with a new K value from the sidebar without reloading data
- **CSV Export** — Download the full RFM table with cluster labels assigned
---
 
## Tech Stack
 
| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Machine Learning | scikit-learn (KMeans, StandardScaler, PCA, silhouette_score) |
| AI Personas | Google Gemini (gemini-2.5-flash) |
| Data Processing | pandas, NumPy |
| Visualization | matplotlib |
| Environment | python-dotenv |
 
---
 
## The Dataset
 
A sample of online retail transactions (~8,000 rows) modeled after the [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail):
 
| Column | Description |
|--------|-------------|
| `InvoiceNo` | Unique invoice identifier |
| `StockCode` | Product code |
| `Description` | Product name |
| `Quantity` | Number of items (negative = return) |
| `InvoiceDate` | Transaction date |
| `UnitPrice` | Price per item |
| `CustomerID` | Customer identifier (~5% missing) |
| `Country` | Customer country |
 
After cleaning: ~800 customers, returns and missing IDs removed, TotalPrice calculated.
 
---
 
## Project Structure
 
```
customer-segmentation/
├── segmentation_engine.py   ← Full ML pipeline: clean, RFM, scale, cluster, visualize, personas
├── segmentation_app.py      ← Streamlit dashboard
├── requirements.txt         ← Project dependencies
├── .env.example             ← Environment variable template
├── .env                     ← API keys (not committed)
├── .gitignore
│
├── data/
│   └── online_retail_sample.csv   ← ~8,000 transactions, ~800 customers
│
└── .streamlit/
    └── config.toml          ← Custom dark theme configuration
```
 
---
 
## Getting Started
 
### 1. Clone the Repository
 
```bash
git clone https://github.com/Drizztovski/customer-segmentation.git
cd customer-segmentation
```
 
### 2. Install Dependencies
 
```bash
pip install -r requirements.txt
```
 
### 3. Configure API Key (optional — only needed for persona generation)
 
```bash
cp .env.example .env
```
 
Open `.env` and add your key:
 
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
 
Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). The app runs fully without it — persona generation is the only feature that requires a key.
 
### 4. Run the App
 
```bash
streamlit run segmentation_app.py
```
 
Opens at `http://localhost:8501`
 
---
 
## How It Works
 
### The Segmentation Engine (`segmentation_engine.py`)
 
1. **Data Cleaning** — Loads the CSV with latin-1 encoding, drops rows with missing CustomerIDs, filters out returns and zero-price entries, and calculates TotalPrice per line item.
2. **RFM Feature Engineering** — Groups by CustomerID and computes three features: Recency (days since last purchase, anchored to max date + 1 day for reproducibility), Frequency (distinct invoice count via nunique), and Monetary (sum of TotalPrice).
3. **Feature Scaling** — Applies StandardScaler to all three RFM features. K-Means is distance-based — without scaling, Monetary values in the hundreds or thousands would dominate Recency and Frequency, producing distorted clusters.
4. **Optimal K Selection** — Loops K=2 through K=8, recording inertia and silhouette score at each value. The elbow curve shows where inertia stops dropping sharply; silhouette score shows where cluster separation peaks.
5. **K-Means Clustering** — Fits the final model with the selected K, random_state=42, n_init=10 for reproducibility.
6. **PCA Projection** — Reduces the 3D scaled RFM space to 2 principal components for visualization. With only 3 input features, PCA typically captures 80-90%+ of variance in 2 components.
7. **AI Persona Generation** — For each cluster, builds a Gemini prompt with the cluster's mean RFM stats, dataset-wide median benchmarks, segment size as a percentage of total customers, and an inferred RFM pattern (e.g. "Champions", "At-Risk High-Value", "Bargain Hunters"). Gemini returns a persona name, behavioral profile, and 3 targeted marketing recommendations.
---
 
## Screenshots
 
*Screenshots coming soon.*
 
---
 
## Key Technical Decisions
 
**Anchor reference date to data, not today** — Using `df['InvoiceDate'].max() + 1 day` as the recency reference point instead of `pd.Timestamp.today()` keeps the analysis reproducible. Running the same dataset a year from now produces identical RFM values.
 
**Scale before clustering, always** — K-Means minimizes Euclidean distance. A $500 vs $1,500 monetary difference looks enormous compared to a 2-day recency difference unless features are normalized. StandardScaler brings all three to mean=0, std=1.
 
**Median benchmarks in persona prompts** — Rather than giving Gemini raw numbers with no context, the prompt includes dataset-wide medians so the model can assess whether a cluster's Recency of 45 days is "recent" or "dormant" relative to the actual customer base.
 
**nunique for frequency, not count** — Each order has multiple line items. Counting rows would inflate frequency counts for customers who bought many products in one order. `nunique()` on InvoiceNo counts distinct orders correctly.
 
---
 
## Author
 
**AJ Amatrudo** — IT professional transitioning to data science and business intelligence.
 
- GitHub: [github.com/Drizztovski](https://github.com/Drizztovski)
- Certifications: Python 3, SQL, Git & GitHub (Codecademy)
- Training: Data Scientist: Analytics Specialist (Codecademy) + Data Science with AI Bootcamp