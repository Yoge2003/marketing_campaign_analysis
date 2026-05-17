-- Business Queries for Marketing Campaign Analysis

USE marketing_campaign_db;

-- 1. Top 10 campaigns by revenue
SELECT Campaign_ID, Brand, Revenue 
FROM marketing_campaign 
ORDER BY Revenue DESC 
LIMIT 10;

-- 2. Lowest ROI campaigns (Bottom 10)
SELECT Campaign_ID, Brand, ROI 
FROM marketing_campaign 
ORDER BY ROI ASC 
LIMIT 10;

-- 3. Best marketing channels (Average Revenue per Channel)
-- Note: Since Channel_Used can contain multiple values, in a fully normalized DB we'd use a junction table. 
-- Assuming simple grouping here for demonstration, or focusing on primary channels:
SELECT Channel_Used, AVG(Revenue) as Avg_Revenue, SUM(Revenue) as Total_Revenue
FROM marketing_campaign
GROUP BY Channel_Used
ORDER BY Total_Revenue DESC;

-- 4. Brand-wise performance
SELECT Brand, 
       SUM(Revenue) as Total_Revenue, 
       AVG(ROI) as Avg_ROI, 
       SUM(Conversions) as Total_Conversions
FROM marketing_campaign
GROUP BY Brand
ORDER BY Total_Revenue DESC;

-- 5. Monthly revenue trends
SELECT DATE_FORMAT(Date, '%Y-%m') AS Month, 
       SUM(Revenue) as Total_Revenue
FROM marketing_campaign
GROUP BY Month
ORDER BY Month ASC;

-- 6. Customer segment performance
SELECT Customer_Segment, 
       SUM(Revenue) as Total_Revenue, 
       AVG(Engagement_Score) as Avg_Engagement
FROM marketing_campaign
GROUP BY Customer_Segment
ORDER BY Total_Revenue DESC;

-- 7. Conversion analysis (Conversion Rate)
SELECT Campaign_Type, 
       SUM(Conversions) as Total_Conversions, 
       SUM(Clicks) as Total_Clicks,
       (SUM(Conversions) / NULLIF(SUM(Clicks), 0)) * 100 AS Conversion_Rate
FROM marketing_campaign
GROUP BY Campaign_Type
ORDER BY Conversion_Rate DESC;
