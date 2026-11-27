# Indian Startup Funding Analysis & Investor Segmentation

An interactive Exploratory Data Analysis (EDA) and Machine Learning web dashboard built with **Python** and **Streamlit** to analyze funding trends, startup growth patterns, and investor behavior across the Indian startup ecosystem.

## Key Features

### 1. Overall Market Analysis (Macro View)
- **KPI Metric Cards:** Displays Total Funding (Cr), Maximum Single Round, Average Funding per Startup, and Total Funded Startups.
- **Sector Analysis:** Interactive pie charts comparing the top 10 verticals by **Total Funding (Sum)** and **Deal Count**.
- **Type of Funding:** Statistical breakdown of funding stages (Seed, Series A–D, Private Equity, Debt) showing deal counts alongside **mean** and **median** ticket sizes to highlight outlier effects.
- **City-Wise Funding:** Side-by-side leaderboards ranking top startup hubs by total capital deployed and deal volume.
- **Top Startups:** Scatter plot comparing *Number of Rounds* vs. *Total Capital Raised* paired with a top-10 breakdown table.
- **Top Investors:** Splits and explodes multi-investor strings to rank the most active investors by deal count.
- **Month-on-Month (MoM) & Seasonal Heatmap:** Time-series line chart and Seaborn Year-vs-Month heatmap tracking capital surges and funding winters.

### 2. Startup Perspective (Company POV)
- Displays industry, sub-industry, headquarters city, and total capital raised for any selected startup.
- Lists complete historical funding rounds in reverse chronological order.
- Recommends up to **5 similar competitors** matched by sub-industry (with a fallback to broader industry).

### 3. Investor Perspective & ML Clustering (Investor POV)
- **Investor Details:** Shows the 5 most recent investments, biggest startup bets, sector preferences, preferred funding stages, city distribution, and Year-over-Year (YoY) investment trajectory.
- **Similar Investors (K-Means Clustering):** Engineers five behavioral features per investor (`total_investment`, `startups_invested`, `sectors`, `cities`, `rounds`), standardizes them using `StandardScaler`, and applies **K-Means (`n_clusters=5`)** to group investors with similar investment footprints.

## Tech Stack
- **Language:** Python
- **Web Framework:** Streamlit
- **Data Manipulation:** Pandas
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (`StandardScaler`, `KMeans`)

## Project Structure
```text
├── app.py                     # Main Streamlit application
├── startup_clean_final.csv    # Cleaned & standardized startup funding dataset
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation