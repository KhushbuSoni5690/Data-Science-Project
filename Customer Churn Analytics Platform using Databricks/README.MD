
Customer Churn Analytics Platform using Databricks
Business Problem
Telecom companies lose customers every month, which directly impacts revenue.
This project identifies customers likely to churn, calculates revenue at risk, and provides business recommendations to improve customer retention.

Project Objective
The objective of this project is to build an end-to-end churn analytics platform using Databricks, PySpark, SQL, Machine Learning, dashboards, and AI-generated business insights.

Architecture Diagram
Architecture

Dataset
The project uses the IBM Telco Customer Churn dataset.

Source: https://github.com/IBM/telco-customer-churn-on-icp4d

This dataset contains customer demographics, account information, subscribed services, billing details, and churn status used to analyze customer retention patterns and predict customer churn.

Key fields include:

Customer ID
Gender
Tenure
Contract Type
Internet Service
Monthly Charges
Total Charges
Churn
Data Architecture
This project follows the Medallion Architecture:

Bronze Layer
Raw customer churn data uploaded into Databricks with no transformation.

Silver Layer
Cleaned and transformed customer data.

Activities included:

Removed duplicates
Handled missing values
Converted data types
Standardised categorical columns
Gold Layer
Business-ready KPI tables used for reporting and dashboards.

Gold tables include:

Churn rate by customer segment
Revenue at risk
High-risk customers
Contract-based churn analysis
Exploratory Data Analysis
EDA was performed to understand:

Overall churn rate
Churn by contract type
Churn by internet service
Churn by payment method
Tenure impact on churn
Monthly charges impact on churn
Machine Learning Model
A churn prediction model was developed to predict whether a customer is likely to churn.

Model output includes:

Predicted churn
Churn probability
High-risk customer flag
Executive Dashboard
The dashboard provides:

Overall churn KPIs
Revenue at risk
High-risk customer segments
Churn trends
AI-generated business insights
Dashboard

AI Insight Generator
The AI Insight Generator converts KPI results into business-friendly insights and recommendations for executives.

Example:

Customers on month-to-month contracts show the highest churn risk. Retention campaigns should focus on this segment through loyalty offers or contract upgrade incentives.

Business Recommendations
Key recommendations:

Target month-to-month customers with retention offers.
Provide incentives for customers to move to yearly contracts.
Focus on high monthly charge customers with high churn probability.
Monitor high-risk customers regularly using churn probability scores.
Use dashboard insights for proactive retention campaigns.
Future Enhancements
Add real-time data ingestion
Deploy the ML model as an API
Add automated email alerts for high-risk customers
Integrate Power BI dashboard
Add customer lifetime value prediction
Build a full AI Data Analyst Agent
Tools Used
Databricks Free Edition
PySpark
Spark SQL
Delta Tables
Machine Learning
Databricks Dashboard
AI Insight Generator
GitHub


     