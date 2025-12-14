# Quantitative Food Processing Classification & Numerical Modeling

## Project Overview
This project addresses the limitations of qualitative food classification systems (like NOVA) by developing a **Quantitative Numerical Model** to assess food processing intensity. By analyzing nutritional data, the system calculates a **Processing Intensity Score (PIS)** to classify foods into six distinct **CHEF classes** (0-5), offering a more granular and objective evaluation of ultra-processed foods.

## Aim & Objective
The primary objective is to build and validate a machine learning-based framework that utilizes nutrient composition data (from 65 to 102 nutrient markers) to accurately predict the degree of processing and assign a corresponding CHEF classification level.

## Repository Structure
*   **`BTP new/Numerical Model Food Processing/`**: Contains the core Machine Learning models, nutrient datasets (`102_Nutrients`, `65_Nutrients`, `FDA_Nutrients`), and model training notebooks.
*   **`new_algo/`**: Houses the baseline heuristic algorithm for comparison, utilizing keyword-based detection for NOVA classification.
*   **`CHEF_Classification.ipynb`**: Implements the logic to map the continuous Processing Intensity Score (PIS) to discrete CHEF classes (0–5).
*   **`BTP_report_main (1).pdf`**: The official Bachelor Thesis Project report detailing literature review, methodology, and full results.

## Methodology
The project employs a data-driven approach using:
*   **Feature Engineering**: Extraction of 65 to 102 nutrient markers per food item.
*   **Machine Learning Models**: Training and evaluation of multiple classifiers including **LightGBM, XGBoost, Random Forest, SVM, and MLP**.
*   **Sampling Techniques**: Utilization of **SMOTE** and **Stratified Cross-Validation** to handle class imbalances and ensure robust model performance.
*   **Scoring System**: Derivation of a continuous PIS metric that correlates with industrial processing levels.

## Outcome
*   Successfully developed a robust **Processing Intensity Score (PIS)** capable of distinguishing fine-grained processing levels.
*   The **LightGBM and XGBoost models** achieved the highest performance, reaching approximately **93% accuracy** on the 102-nutrient dataset.
*   Demonstrated that a quantitative, nutrient-based model offers superior resolution compared to traditional, broad-category classification systems.
