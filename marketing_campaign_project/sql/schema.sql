-- Normalized Schema for Marketing Campaign Analysis

-- 1. Brands Table
CREATE TABLE IF NOT EXISTS Brands (
    BrandID INT AUTO_INCREMENT PRIMARY KEY,
    BrandName VARCHAR(50) UNIQUE NOT NULL
);

-- 2. Customer Segments Table
CREATE TABLE IF NOT EXISTS CustomerSegments (
    SegmentID INT AUTO_INCREMENT PRIMARY KEY,
    SegmentName VARCHAR(100) UNIQUE NOT NULL
);

-- 3. Campaign Types Table
CREATE TABLE IF NOT EXISTS CampaignTypes (
    TypeID INT AUTO_INCREMENT PRIMARY KEY,
    TypeName VARCHAR(100) UNIQUE NOT NULL
);

-- 4. Channels Table
CREATE TABLE IF NOT EXISTS Channels (
    ChannelID INT AUTO_INCREMENT PRIMARY KEY,
    ChannelName VARCHAR(100) UNIQUE NOT NULL
);

-- 5. Campaign Performance Table (Fact Table)
CREATE TABLE IF NOT EXISTS CampaignPerformance (
    CampaignID VARCHAR(50) PRIMARY KEY,
    BrandID INT,
    TypeID INT,
    SegmentID INT,
    Duration INT,
    Impressions BIGINT,
    Clicks BIGINT,
    Leads INT,
    Conversions INT,
    Revenue DECIMAL(15, 2),
    AcquisitionCost DECIMAL(15, 2),
    ROI DECIMAL(10, 2),
    EngagementScore DECIMAL(5, 2),
    CampaignDate DATE,
    Language VARCHAR(50),
    
    FOREIGN KEY (BrandID) REFERENCES Brands(BrandID),
    FOREIGN KEY (TypeID) REFERENCES CampaignTypes(TypeID),
    FOREIGN KEY (SegmentID) REFERENCES CustomerSegments(SegmentID)
);

-- 6. Indexing for Performance
CREATE INDEX idx_campaign_date ON CampaignPerformance(CampaignDate);
CREATE INDEX idx_brand_id ON CampaignPerformance(BrandID);
CREATE INDEX idx_revenue ON CampaignPerformance(Revenue);
