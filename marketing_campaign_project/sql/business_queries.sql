-- Professional Business Intelligence Queries for Marketing Analysis

-- [BRAND PERFORMANCE]
-- 1. Total Revenue and ROI by Brand
SELECT b.BrandName, SUM(cp.Revenue) as TotalRevenue, AVG(cp.ROI) as AvgROI
FROM CampaignPerformance cp
JOIN Brands b ON cp.BrandID = b.BrandID
GROUP BY b.BrandName;

-- 2. Top Brand by Conversion Rate
SELECT b.BrandName, (SUM(cp.Conversions) / SUM(cp.Clicks)) * 100 as ConvRate
FROM CampaignPerformance cp
JOIN Brands b ON cp.BrandID = b.BrandID
GROUP BY b.BrandName ORDER BY ConvRate DESC LIMIT 1;

-- 3. Monthly Revenue Growth per Brand (Window Function)
WITH MonthlyRev AS (
    SELECT b.BrandName, MONTH(cp.CampaignDate) as ResMonth, SUM(cp.Revenue) as Rev
    FROM CampaignPerformance cp
    JOIN Brands b ON cp.BrandID = b.BrandID
    GROUP BY b.BrandName, ResMonth
)
SELECT BrandName, ResMonth, Rev, 
       LAG(Rev) OVER(PARTITION BY BrandName ORDER BY ResMonth) as PrevMonthRev
FROM MonthlyRev;

-- [CHANNEL ANALYSIS]
-- 4. Most Cost-Effective Campaign Type (Lowest Acquisition Cost per Conversion)
SELECT ct.TypeName, SUM(cp.AcquisitionCost) / SUM(cp.Conversions) as CostPerConv
FROM CampaignPerformance cp
JOIN CampaignTypes ct ON cp.TypeID = ct.TypeID
GROUP BY ct.TypeName ORDER BY CostPerConv ASC;

-- 5. Channels contributing to Top 10% Revenue campaigns (CTE)
WITH HighRev AS (
    SELECT CampaignID, Revenue, NTILE(10) OVER(ORDER BY Revenue DESC) as decile
    FROM CampaignPerformance
)
SELECT * FROM HighRev WHERE decile = 1;

-- [CUSTOMER BEHAVIOR]
-- 6. Revenue contribution by Customer Segment
SELECT cs.SegmentName, SUM(cp.Revenue) as SegmentRevenue,
       (SUM(cp.Revenue) / (SELECT SUM(Revenue) FROM CampaignPerformance)) * 100 as RevTag
FROM CampaignPerformance cp
JOIN CustomerSegments cs ON cp.SegmentID = cs.SegmentID
GROUP BY cs.SegmentName;

-- [ROI & PROFITABILITY]
-- 7. List campaigns where ROI is above average
SELECT CampaignID, ROI 
FROM CampaignPerformance 
WHERE ROI > (SELECT AVG(ROI) FROM CampaignPerformance);

-- 8. Brands with consistent ROI > 20%
SELECT b.BrandName, COUNT(*) as HighROICount
FROM CampaignPerformance cp
JOIN Brands b ON cp.BrandID = b.BrandID
WHERE cp.ROI > 20
GROUP BY b.BrandName;

-- [TIME SERIES]
-- 9. Daily Revenue Moving Average (7-day window)
SELECT CampaignDate, Revenue,
       AVG(Revenue) OVER(ORDER BY CampaignDate ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as MovingAvg
FROM CampaignPerformance;

-- 10. Quarterly Performance Summary
SELECT QUARTER(CampaignDate) as Qtr, SUM(Revenue) as TotalRev, SUM(Conversions) as TotalConv
FROM CampaignPerformance
GROUP BY Qtr;

-- [ADVANCED METRICS]
-- 11. Marketing Efficiency Score per Campaign
SELECT CampaignID, (ROI * (Conversions/Clicks)) as EfficiencyScore
FROM CampaignPerformance;

-- 12. Ranking Campaigns by Revenue within each Brand
SELECT b.BrandName, cp.CampaignID, cp.Revenue,
       RANK() OVER(PARTITION BY b.BrandID ORDER BY cp.Revenue DESC) as RankInBrand
FROM CampaignPerformance cp
JOIN Brands b ON cp.BrandID = b.BrandID;

-- 13. Campaign Duration vs ROI Correlation Helper
SELECT Duration, AVG(ROI) as AvgROI
FROM CampaignPerformance
GROUP BY Duration ORDER BY Duration;

-- 14. Language-wise Engagement Score Analysis
SELECT Language, AVG(EngagementScore) as AvgEngagement
FROM CampaignPerformance
GROUP BY Language;

-- 15. Top 5 Campaigns with highest Clicks-to-Impressions (CTR)
SELECT CampaignID, (Clicks/Impressions) as CTR
FROM CampaignPerformance
ORDER BY CTR DESC LIMIT 5;

-- 16. Total Acquisition Cost vs Budget (Assuming 1M Budget)
SELECT BrandID, SUM(AcquisitionCost) as TotalSpend, (1000000 - SUM(AcquisitionCost)) as RemainingBudget
FROM CampaignPerformance GROUP BY BrandID;

-- 17. Identifying Underperforming Campaigns (Low ROI and Low Engagement)
SELECT CampaignID, ROI, EngagementScore 
FROM CampaignPerformance 
WHERE ROI < 0 AND EngagementScore < (SELECT AVG(EngagementScore) FROM CampaignPerformance);

-- 18. Cumulative Revenue over time
SELECT CampaignDate, SUM(Revenue) OVER(ORDER BY CampaignDate) as CumulativeRevenue
FROM CampaignPerformance;

-- 19. Campaign Type Popularity by Brand
SELECT b.BrandName, ct.TypeName, COUNT(*) as TypeCount
FROM CampaignPerformance cp
JOIN Brands b ON cp.BrandID = b.BrandID
JOIN CampaignTypes ct ON cp.TypeID = ct.TypeID
GROUP BY b.BrandName, ct.TypeName;

-- 20. Segment wise Average ROI
SELECT cs.SegmentName, AVG(cp.ROI) as AvgROI
FROM CampaignPerformance cp
JOIN CustomerSegments cs ON cp.SegmentID = cs.SegmentID
GROUP BY cs.SegmentName ORDER BY AvgROI DESC;

-- [MISC BUSINESS CHECKS]
-- 21. Count of Campaigns running for more than 30 days
SELECT COUNT(*) FROM CampaignPerformance WHERE Duration > 30;

-- 22. Average Impressions per Language
SELECT Language, AVG(Impressions) FROM CampaignPerformance GROUP BY Language;

-- 23. Revenue per Click (RPC) by Brand
SELECT b.BrandName, SUM(cp.Revenue) / SUM(cp.Clicks) as RPC
FROM CampaignPerformance cp
JOIN Brands b ON cp.BrandID = b.BrandID
GROUP BY b.BrandName;

-- 24. Day of the week with highest revenue (Requires date functions)
SELECT DAYNAME(CampaignDate) as Day, SUM(Revenue) as TotalRev
FROM CampaignPerformance
GROUP BY Day ORDER BY TotalRev DESC;

-- 25. Median Revenue Calculation (Approximate via ranking)
WITH RankedRev AS (
    SELECT Revenue, ROW_NUMBER() OVER(ORDER BY Revenue) as row_num, COUNT(*) OVER() as total_count
    FROM CampaignPerformance
)
SELECT AVG(Revenue) as MedianRevenue FROM RankedRev WHERE row_num BETWEEN total_count/2 AND total_count/2 + 1;

-- 26. Campaigns with zero conversions but high clicks
SELECT CampaignID, Clicks FROM CampaignPerformance WHERE Conversions = 0 AND Clicks > 100;

-- 27. Total Leads by Brand and Campaign Type
SELECT b.BrandName, ct.TypeName, SUM(cp.Leads) as TotalLeads
FROM CampaignPerformance cp
JOIN Brands b ON cp.BrandID = b.BrandID
JOIN CampaignTypes ct ON cp.TypeID = ct.TypeID
GROUP BY b.BrandName, ct.TypeName;

-- 28. Revenue per Lead (RPL) by Segment
SELECT cs.SegmentName, SUM(cp.Revenue) / SUM(cp.Leads) as RPL
FROM CampaignPerformance cp
JOIN CustomerSegments cs ON cp.SegmentID = cs.SegmentID
GROUP BY cs.SegmentName;

-- 29. Percentage of Profitable Campaigns per Brand
SELECT b.BrandName, 
       (COUNT(CASE WHEN cp.ROI > 0 THEN 1 END) / COUNT(*)) * 100 as ProfitablePct
FROM CampaignPerformance cp
JOIN Brands b ON cp.BrandID = b.BrandID
GROUP BY b.BrandName;

-- 30. Top 3 Segments by Average Duration
SELECT cs.SegmentName, AVG(cp.Duration) as AvgDur
FROM CampaignPerformance cp
JOIN CustomerSegments cs ON cp.SegmentID = cs.SegmentID
GROUP BY cs.SegmentName ORDER BY AvgDur DESC LIMIT 3;
