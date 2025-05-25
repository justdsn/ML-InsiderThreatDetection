# Machine Learning Approach for Insider Threat Detection

Machine learning pipeline for detecting insider threats using email and psychometric data. This project leverages the **Isolation Forest** algorithm to identify anomalous user behaviors, that could indicate risks like data theft or privilege abuse, analyzing email activity, personality traits, and user-PC interactions. Inspired by research on insider threat detection, such as CMU’s CERT datasets, this project combines behavioral analytics with data science to enhance cybersecurity.

## Project Motivation
Insider threats, where authorized individuals misuse their access, pose significant risks to organizations, leading to data breaches or financial losses. This project aims to detect such threats by analyzing email patterns (e.g., frequency, size, attachments), psychometric traits (e.g., Big Five personality scores), and network interactions (user-PC connections). By integrating multiple data sources, this project provide a robust approach to identifying anomalous behaviors, contributing to safer organizational environments.

## Features
- **Core Analysis**: Processes email data to detect anomalies based on email frequency, size, and attachments.
- **Extended Analysis**: Incorporates psychometric data (Big Five traits) and bipartite graph features for user-PC interactions.
- **Visualizations**: Generates scatter plots, time-series plots, heatmaps, and box plots for insights.

## File	Description
- InsiderThreat_CoreCMU.py	Analyzes email data using Isolation Forest, producing anomaly scores and visualizations like scatter and time-series plots.
- InsiderThreat_DfuseCMU.py	Extends core analysis with psychometric data and user-PC graph analysis, including additional visualizations (e.g., heatmaps, box plots).
- requirements.txt	Lists Python dependencies for running the scripts.

## Requirements
- Python 3.8+
- Dependencies listed in requirements.txt:
  - pandas>=1.5.0
  - numpy>=1.23.0
  - scikit-learn>=1.2.0
  - seaborn>=0.12.0
  - matplotlib>=3.6.0
  - networkx>=2.8.0
  - tqdm>=4.64.0
