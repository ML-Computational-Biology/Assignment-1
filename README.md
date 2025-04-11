# Assignment-1: Predicting BMI from Metagenomic Data

This project focuses on predicting the Body Mass Index (BMI) of individuals using metagenomic features from gut microbiome data. It is part of the "ML for Computational Biology" course assignment and includes data exploration, feature selection, model training, evaluation, and hyperparameter optimization.

## Project Structure

```plaintext
Assignment-1/
├── data/                 # Cleaned datasets (development and evaluation)
├── notebooks/            # Jupyter notebooks for exploration and analysis
├── src/                  # Source code (functions, model training, evaluation)
├── models/               # Trained model instances (baseline, FS, tuned)
├── models_optuna/        # Models trained with Optuna hyperparameter tuning
├── final_models/         # Best model per regressor and overall winner
├── eval_results/         # Pickle file with evaluation metrics
Implemented Models

    Elastic Net

    Support Vector Regression (SVR)

    Bayesian Ridge Regression

Each model is evaluated under the following configurations:

    Baseline: All features with default hyperparameters.

    FS (Feature Selection): Using Lasso to select the most predictive features.

    FS + Tuning: Feature-selected models with hyperparameter optimization using Grid Search or Optuna.
