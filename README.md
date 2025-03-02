# LSTM-HDL
Sample code for DL-based hierarchical reconciliation
# Summary
Deep learning (DL) models have become integral to hydrological forecasting, yet ensuring consistency across multiple timescales remains a significant challenge. For the first time, we integrate Long Short-Term Memory (LSTM) networks with Ordinary Least Squares (OLS), Hierarchical Least Squares (HLS), and Weighted Least Squares (WLS) reconciliation layers to enforce coherence in multi-timescale (MTS) streamflow predictions. This novel hierarchical deep learning (HDL) framework enhances forecast reliability by addressing discrepancies in daily and weekly streamflow predictions. HDL improves weekly Nash Sutcliffe efficiency (NSE) by up to 49.7% while outperforming bottom-up aggregation approaches. This advancement establishes a benchmark for consistent and accurate MTS hydrological forecasting.

The results suggest that temporal reconciliation improves the performance compared to bottom-up aggregation (BU) and joint forecasting with reconciliation.
Daily results example:
![HDL_Daily_Comparison_v1](https://github.com/user-attachments/assets/3530c023-5c7b-4fc6-bdb1-f8f9b448901a)
Weekly results example:
![HDL_Weekly_Comparison_v1](https://github.com/user-attachments/assets/9bb658c1-eb22-4441-8c04-8965b8c0e100)

# Code
Sample code for inline HLS reconciliation is provided
