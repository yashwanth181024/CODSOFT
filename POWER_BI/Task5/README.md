# Task-5 📊 Financial Health Dashboard 💰

> **CodSoft Power BI Internship | ID: BY26RY230812**
> **Author : Yashwanth G S**
> **GitHub : https://github.com/Yashwanth181024**

---

## 📌 Objective
Develop a comprehensive financial dashboard that provides real-time insights into an organization's financial performance. The dashboard helps management monitor income, expenses, profitability, and cash flow while supporting strategic financial planning.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| File | `Task5_Financial_Analytics.pbix` |
| Data Table | `FinancialData` |
| Fields | COGS, Financing Cash Flow, Gross Profit, Investing Cash Flow, Month, Month Name, Date Hierarchy, Net Cash Flow, Net Profit (After Tax), Net Profit (Pre-Tax), Operating Cash Flow, Operating Expenses, Revenue, Tax, Year |
| Missing Values | Cleaned during Power Query import |

---

## ⚙️ Key Requirements Implemented
- Imported financial datasets including income, expenses, and cash flow data
- Created KPI cards for Total Revenue, Gross Profit, and Net Cash Flow
- Analyzed monthly and yearly financial performance using interactive charts
- Built a Date Hierarchy (Year → Quarter → Month → Day) to support trend analysis
- Added slicers and filters (Year range, Month Name) for detailed financial analysis

---

## 🔧 Dashboard Breakdown

### KPI Cards
| Metric | Value |
|---|---|
| Sum of Revenue | 48.14M |
| Sum of Gross Profit | 25.52M |
| Sum of Net Cash Flow | 11.00M |

### Visuals
- **Clustered Bar Chart** — Sum of Gross Profit and Sum of Revenue by Month Name
- **Donut Chart** — Sum of Net Profit (After Tax) by Month Name

### Slicers / Filters
- Year (range slider, 2000–2025)
- Month Name (April, August, December, February, January, etc.)

---

## 📊 Output Files

| File | Description |
|---|---|
| `Task5_Financial_Analytics.pbix` | Power BI financial health dashboard file |
| Screenshot | PNG export of the dashboard page |

---

## 🚀 How to Run
1. Install **Power BI Desktop** (free from Microsoft).
2. Open `Task5_Financial_Analytics.pbix`.
3. Use **Refresh** (Home tab) to reload data if connected to a live source.
4. Interact with the Year and Month Name slicers to explore revenue, profit, and cash flow trends.

---

## 💡 Features
- KPI-first layout with Revenue, Gross Profit, and Net Cash Flow cards
- Month-over-month comparison of Gross Profit and Revenue
- Net Profit (After Tax) distribution by month via donut chart
- Year-range slicer for flexible period-over-period analysis
- Date hierarchy support for drill-down from Year to Day

---

## 📈 Key Results
- Gross Profit tracks closely with Revenue across most months, indicating a stable cost structure
- Net Profit (After Tax) is fairly evenly distributed across months, each contributing roughly 7–10% of the annual total
- Net Cash Flow of 11.00M against 48.14M in Revenue highlights the portion of revenue converting to actual cash
- Dashboard supports quick identification of the strongest and weakest performing months for financial planning

---

## 🛠️ Tools
```
Power BI Desktop | Power Query | DAX | Data Modeling
```

---

## 👤 Author
**Yashwanth G S**
GitHub : https://github.com/Yashwanth181024
Internship : CodSoft Power BI Internship | ID: BY26RY230812 (15 Aug 2026 – 15 Sept 2026)
