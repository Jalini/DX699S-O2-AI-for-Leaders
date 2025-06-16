# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 16:34:07 2025

@author: jalin
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Step 1: Set your working directory
os.chdir(r"C:\Users\jalin\Documents\data\AILeader")
print("Working directory set to:", os.getcwd())

##################################################
#First I do the analysis for Hypertension

# Load the dataset
df_hypertension = pd.read_csv("hypertension_dataset.csv")  
# Set visualization style
sns.set(style="whitegrid")

# -------------------------------------
# DEFINE NUMERICAL AND CATEGORICAL COLUMNS
# -------------------------------------
numerical_columns = [
    'Age', 'BMI', 'Cholesterol', 'Systolic_BP', 'Diastolic_BP',
    'Alcohol_Intake', 'Salt_Intake', 'Sleep_Duration', 'Heart_Rate',
    'LDL', 'HDL', 'Triglycerides', 'Glucose'
]

categorical_columns = [
    'Gender', 'Smoking_Status', 'Education_Level',
    'Physical_Activity_Level', 'Employment_Status', 'Hypertension'
]

# -------------------------------------
# 1. HISTOGRAMS + KDE + VIOLIN + SWARM for all numeric features
# -------------------------------------
print("Generating Histogram Variants...")
for col in numerical_columns:
    plt.figure(figsize=(14, 4))

    # Histogram
    plt.subplot(1, 4, 1)
    sns.histplot(df_hypertension[col], kde=False)
    plt.title(f'Histogram: {col}')

    # KDE
    plt.subplot(1, 4, 2)
    sns.kdeplot(df_hypertension[col])
    plt.title(f'KDE Plot: {col}')

    # Violin
    plt.subplot(1, 4, 3)
    sns.violinplot(y=df_hypertension[col])
    plt.title(f'Violin Plot: {col}')

    # Swarm plot (sampled for performance)
    plt.subplot(1, 4, 4)
    sns.swarmplot(x='Hypertension', y=col, data=df_hypertension.sample(500, random_state=42))
    plt.title(f'Swarm Plot: {col} vs Hypertension')

    plt.tight_layout()
    plt.show()

# -------------------------------------
# 2. GROUPED HISTOGRAMS by 'Hypertension'
# -------------------------------------
print("Generating Grouped Histograms...")
for col in ['Age', 'BMI', 'Cholesterol', 'Systolic_BP']:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_hypertension, x=col, hue='Hypertension', kde=True, element='step', stat='density', common_norm=False)
    plt.title(f'Grouped Histogram of {col} by Hypertension')
    plt.tight_layout()
    plt.show()

# -------------------------------------
# 3. BAR PLOTS for Categorical Variables
# -------------------------------------
print("Generating Bar Plots...")
for col in categorical_columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_hypertension, x=col, order=df_hypertension[col].value_counts().index)
    plt.title(f'Bar Plot of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -------------------------------------
# 4. MEAN, MEDIAN, MODE SUMMARY
# -------------------------------------
print("\n Summary Statistics (Mean, Median, Mode):\n")
summary_stats = df_hypertension[numerical_columns].agg(['mean', 'median']).T
summary_stats['mode'] = [df_hypertension[col].mode().iloc[0] for col in summary_stats.index]
print(summary_stats)

#########################################
# Next I do the same analysis for CKD

sns.set(style="whitegrid")

# --------------------------------------
# Load Dataset
# --------------------------------------
df_kidney = pd.read_csv("Chronic_Kidney_Dsease_data.csv")  

# --------------------------------------
# Define Columns
# --------------------------------------
# Drop PatientID and identify numeric/categorical columns
numerical_cols_kidney = df_kidney.select_dtypes(include=['float64', 'int64']).drop(columns=['PatientID']).columns.tolist()
categorical_cols_kidney = df_kidney.select_dtypes(include=['object']).columns.tolist()

# --------------------------------------
# 1. Histogram Variants (for each numeric column)
# --------------------------------------
print("Generating histogram variants for numerical columns...")
for col in numerical_cols_kidney[:]:  # Adjust range to [:] for all columns if desired
    plt.figure(figsize=(14, 4))

    # Histogram
    plt.subplot(1, 4, 1)
    sns.histplot(df_kidney[col], kde=False)
    plt.title(f'Histogram: {col}')

    # KDE
    plt.subplot(1, 4, 2)
    sns.kdeplot(df_kidney[col])
    plt.title(f'KDE Plot: {col}')

    # Violin
    plt.subplot(1, 4, 3)
    sns.violinplot(y=df_kidney[col])
    plt.title(f'Violin Plot: {col}')

    # Swarm Plot by Diagnosis
    plt.subplot(1, 4, 4)
    sns.swarmplot(x='Diagnosis', y=col, data=df_kidney.sample(500, random_state=42))
    plt.title(f'Swarm Plot: {col} by Diagnosis')

    plt.tight_layout()
    plt.show()

# --------------------------------------
#  2. Grouped Histograms by 'Diagnosis'
# --------------------------------------
print("Generating grouped histograms...")
for col in ['Age', 'BMI', 'SystolicBP', 'GFR']:
    if col in df_kidney.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df_kidney, x=col, hue='Diagnosis', kde=True, element='step', stat='density', common_norm=False)
        plt.title(f'Grouped Histogram of {col} by Diagnosis')
        plt.tight_layout()
        plt.show()

# --------------------------------------
# 3. Bar Plots for Categorical Variables
# --------------------------------------
print("Generating bar plots for categorical variables...")
for col in categorical_cols_kidney:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_kidney, x=col, order=df_kidney[col].value_counts().index)
    plt.title(f'Bar Plot of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --------------------------------------
# 4. Summary Statistics (Mean, Median, Mode)
# --------------------------------------
print("Calculating summary statistics...")
summary_kidney = df_kidney[numerical_cols_kidney].agg(['mean', 'median']).T
summary_kidney['mode'] = [df_kidney[col].mode().iloc[0] for col in summary_kidney.index]

# Display in Jupyter/Spyder
print("Summary Statistics (Mean, Median, Mode):")
print(summary_kidney)

# Optionally save to CSV
summary_kidney.to_csv("kidney_summary_stats.csv")
print("Summary statistics saved to 'kidney_summary_stats.csv'")

############################
# Finally I do analysis for Diabetic 50:50 dataset
# Set style
sns.set(style="whitegrid")

# --------------------------------------
# Load Diabetics Dataset (50:50)
# --------------------------------------
df_diabetes = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")  

# --------------------------------------
# Identify Columns
# --------------------------------------
numerical_cols_diabetes = df_diabetes.columns.tolist()
target_col = 'Diabetes_binary'

# --------------------------------------
# 1. Histogram, KDE, Violin, Swarm
# --------------------------------------
print(" Generating histogram variants...")
for col in numerical_cols_diabetes[:]:  
    if col == target_col:
        continue  # Skip target variable

    plt.figure(figsize=(14, 4))

    # Histogram
    plt.subplot(1, 4, 1)
    sns.histplot(df_diabetes[col], kde=False)
    plt.title(f'Histogram: {col}')

    # KDE
    plt.subplot(1, 4, 2)
    sns.kdeplot(df_diabetes[col])
    plt.title(f'KDE Plot: {col}')

    # Violin
    plt.subplot(1, 4, 3)
    sns.violinplot(y=df_diabetes[col])
    plt.title(f'Violin Plot: {col}')

    # Swarm Plot
    plt.subplot(1, 4, 4)
    sns.swarmplot(x=target_col, y=col, data=df_diabetes.sample(500, random_state=42))
    plt.title(f'Swarm Plot: {col} by {target_col}')

    plt.tight_layout()
    plt.show()

# --------------------------------------
# 2. Grouped Histograms by Diabetes
# --------------------------------------
print(" Generating grouped histograms...")
for col in ['BMI', 'GenHlth', 'PhysHlth', 'Age']:
    if col in df_diabetes.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df_diabetes, x=col, hue=target_col, kde=True, element='step', stat='density', common_norm=False)
        plt.title(f'Grouped Histogram of {col} by {target_col}')
        plt.tight_layout()
        plt.show()

# --------------------------------------
# 3. Bar Plots for Categorical Features
# --------------------------------------
print(" Generating bar plots...")
for col in ['HighBP', 'HighChol', 'Smoker', 'PhysActivity', 'Sex']:
    if col in df_diabetes.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df_diabetes, x=col)
        plt.title(f'Bar Plot of {col}')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

# --------------------------------------
# 4. Summary Statistics
# --------------------------------------
print(" Calculating summary statistics...")
summary_diabetes = df_diabetes[numerical_cols_diabetes].agg(['mean', 'median']).T
summary_diabetes['mode'] = [df_diabetes[col].mode().iloc[0] for col in summary_diabetes.index]

# Display and save to CSV
print("\n Summary Statistics (Mean, Median, Mode):")
print(summary_diabetes)

summary_diabetes.to_csv("diabetes_5050_summary_stats.csv")
print(" Summary statistics saved to 'diabetes_5050_summary_stats.csv'")