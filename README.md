# Facebook vs AdWords Ad Campaign Analysis

## Business Problem
A marketing agency wants to maximize the **return on investment (ROI)** for clients’ advertising campaigns.  
We ran **two ad campaigns** (Facebook and AdWords) in 2019 and need to determine:

- Which platform generates **more clicks**  
- Which drives **higher conversions**  
- Which is **more cost-effective**  

The findings will help optimize budget allocation and advertising strategies.

---

## Research Question
**Which ad platform is more effective in terms of conversions, clicks, and overall cost-effectiveness?**

---

## Dataset
- **Period:** Jan 1 – Dec 31, 2019 (365 days)  
- **Platforms:** Facebook & AdWords  
- **Features:**  
  - Ad Views  
  - Ad Clicks  
  - Ad Conversions  
  - Cost per Ad  
  - Click-Through Rate (CTR)  
  - Conversion Rate  
  - Cost per Click (CPC)  

---

## Tools & Libraries
```python
pandas, matplotlib, seaborn, numpy, scipy.stats
sklearn (Linear Regression, metrics)
statsmodels (time-series decomposition, cointegration)

```
## Analysis Performed

### Exploratory Data Analysis (EDA)
- Histograms for clicks & conversions  
- Conversion categories comparison  
- Weekly & monthly conversion trends  

### Correlation Analysis
- **Facebook Clicks vs Conversions** → Strong correlation (**0.87**)  
- **AdWords Clicks vs Conversions** → Moderate correlation (**0.45**)  

### Hypothesis Testing
- **H0:** µ_Facebook ≤ µ_AdWords  
- **H1:** µ_Facebook > µ_AdWords  
- **Result:** ✅ Reject H0 (Facebook conversions significantly higher, *p ≈ 0*)  

### Regression Analysis
- Linear Regression model for **Facebook clicks → conversions**  
- Predictive power: **R² ≈ 76%**  
- Example:  
  - 50 clicks → ~13 conversions  
  - 80 clicks → ~19 conversions  

### Cost & ROI Analysis
- Monthly **Cost per Conversion (CPC)** trends  
- Facebook more cost-efficient in **May & November**  
- **Cointegration test** shows long-term equilibrium between ad spend & conversions  

---

## Key Insights
- **Facebook outperformed AdWords** in both conversions and cost-effectiveness.  
- **Average Conversions/day:**  
  - Facebook ≈ **11.74**  
  - AdWords ≈ **5.98**  
- **Higher ROI:** Stronger relationship between clicks and conversions on Facebook.  
- **Best Days:** Mondays & Tuesdays show the highest conversions.  
- **Budget Strategy:** Allocate more budget to Facebook ads, especially in **May & November**.  

---

## Visualizations
This analysis produces the following plots:  
- Conversion distributions (histograms)  
- Frequency of daily conversions by category (bar chart)  
- Clicks vs Conversions scatter plots  
- Weekly & monthly conversion trends  
- Monthly CPC trend line  

---

## How to Run

```bash
# Clone repo
git clone https://github.com/yourusername/ad-campaign-analysis.git
cd ad-campaign-analysis

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook campaign_analysis.ipynb

```
## Future Improvements

- Add ROI dashboard (Tableau / Power BI / Streamlit)

- Include A/B test simulation for campaign optimization

- Extend analysis with multi-channel attribution modeling




