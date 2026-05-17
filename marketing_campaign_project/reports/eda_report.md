# Exploratory Data Analysis Report

## Business Insights
1. **Revenue by Brand**: Visualizes which brand brings the most revenue.
2. **Channel Effectiveness**: Helps identify the best-performing marketing channels based on total revenue.
3. **Campaign Performance**: Highlights the top 10 revenue-generating campaigns and the bottom 10 lowest ROI campaigns.

## Key Findings
- **Data Quality**: The dataset originally contained nulls which have been successfully imputed. Outliers were capped using the IQR method.
- **Correlations**: The heatmap shows strong correlations between target variables like Revenue and performance metrics like Conversions.
- **Distributions**: Revenue and Acquisition Cost display normal/skewed distributions which were handled via capping.

## Recommendations
- **Allocate Budget Efficiently**: Reinvest heavily in the top marketing channels identified in the 'Channel Effectiveness' chart.
- **Optimize Low ROI Campaigns**: Investigate the campaigns listed in the 'Bottom 10 Campaigns' to identify weak points in targeting or ad creatives.
- **Focus on High-Value Segments**: Leverage insights from the Customer Segment analysis to target the most profitable audience.
