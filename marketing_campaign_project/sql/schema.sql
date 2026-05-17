-- SQL Schema for Marketing Campaign Performance Prediction System

CREATE DATABASE IF NOT EXISTS marketing_campaign_db;
USE marketing_campaign_db;

CREATE TABLE IF NOT EXISTS marketing_campaign (
    Campaign_ID VARCHAR(50) PRIMARY KEY,
    Brand VARCHAR(50),
    Campaign_Type VARCHAR(50),
    Target_Audience VARCHAR(100),
    Duration FLOAT,
    Channel_Used VARCHAR(255),
    Impressions FLOAT,
    Clicks FLOAT,
    Leads FLOAT,
    Conversions FLOAT,
    Revenue FLOAT,
    Acquisition_Cost FLOAT,
    ROI FLOAT,
    Language VARCHAR(50),
    Engagement_Score FLOAT,
    Customer_Segment VARCHAR(100),
    Date DATE
);
