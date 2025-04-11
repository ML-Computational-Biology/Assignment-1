import os
import joblib
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import json
import optuna

from sklearn.linear_model import ElasticNet, BayesianRidge, LassoCV
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict


# =========================
# GLOBAL CONSTANTS
# =========================
CV_FOLDS = 5
N_ITER = 1000
RANDOM_STATE = 42

class ModelTrainer:
    def __init__(self, X_dev, y_dev, X_eval=None, y_eval=None, random_state=RANDOM_STATE):
        self.X_dev = X_dev
        self.y_dev = y_dev
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.random_state = random_state

        self.baseline_models = {
            'ElasticNet': ElasticNet(),
            'SVR': SVR(),
            'BayesianRidge': BayesianRidge()
        }

        self.baseline_cv_results = {}
        self.baseline_eval_results = {}
        self.selected_features_idx = None
        self.X_dev_selected = None
        self.X_eval_selected = None
        self.fs_eval_results = {}
        self.tuned_models = {}
        self.tuned_eval_results = {}

    # =========================
    # COMMON UTILITY FUNCTIONS
    # =========================
    def _save_model(self, model, filename='final_model.joblib', model_dir='../models'):
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, filename)
        joblib.dump(model, path)
        print(f"Model saved to {path}")

    def _evaluate_model_bootstrap(self, model, X_eval, y_eval, n_iterations=N_ITER):
        rng = np.random.RandomState(self.random_state)
        n = len(X_eval)
        rmse_list, mae_list, r2_list = [], [], []

        for _ in range(n_iterations):
            indices = rng.choice(n, size=n, replace=True)
            X_sample = X_eval[indices]
            y_sample = y_eval[indices]
            y_pred = model.predict(X_sample)

            rmse_list.append(float(np.sqrt(mean_squared_error(y_sample, y_pred))))
            mae_list.append(float(mean_absolute_error(y_sample, y_pred)))
            r2_list.append(float(r2_score(y_sample, y_pred)))

        def summarize(metric_list):
            return {
                'mean': round(np.mean(metric_list), 4),
                'median': round(np.median(metric_list), 4),
                '95% CI': (round(float(np.percentile(metric_list, 2.5)), 4), round(float(np.percentile(metric_list, 97.5)), 4)
                          )

            }

        summary = {
            'RMSE': summarize(rmse_list),
            'MAE': summarize(mae_list),
            'R2': summarize(r2_list)
        }
        
        raw = {
        'rmse_list': [round(float(x), 4) for x in rmse_list],
        'mae_list': [round(float(x), 4) for x in mae_list],
        'r2_list': [round(float(x), 4) for x in r2_list]
        }


     


        return summary, raw

    # =========================
    # 1. BASELINE MODELS
    # =========================
    def train_and_evaluate_baseline_models(self, model_dir="../models"):
        for name, model in self.baseline_models.items():
            rmse_cv = np.mean(np.sqrt(-cross_val_score(model, self.X_dev, self.y_dev, scoring='neg_mean_squared_error', cv=CV_FOLDS)))
            self.baseline_cv_results[name] = rmse_cv
            print(f"{name} baseline CV RMSE: {rmse_cv:.4f}")

            print(f"\nTraining {name} on all features...")
            model.fit(self.X_dev, self.y_dev)
            self._save_model(model, filename=f"{name}_baseline.joblib", model_dir=model_dir)

            print(f"Evaluating {name} on evaluation set...")
            summary, raw = self._evaluate_model_bootstrap(model, self.X_eval, self.y_eval)
            self.baseline_eval_results[name] = {'summary': summary, 'raw': raw}

            print(f"--- {name} Evaluation Summary ---")
            print(f"RMSE:  mean = {summary['RMSE']['mean']}, median = {summary['RMSE']['median']}, 95% CI = {summary['RMSE']['95% CI']}")
            print(f"MAE:   mean = {summary['MAE']['mean']}, median = {summary['MAE']['median']}, 95% CI = {summary['MAE']['95% CI']}")
            print(f"R²:    mean = {summary['R2']['mean']}, median = {summary['R2']['median']}, 95% CI = {summary['R2']['95% CI']}")

        return self.baseline_eval_results

    # =========================
    # 2. FEATURE SELECTION
    # =========================
    def select_features_with_lasso(self):
        lasso = LassoCV(cv=CV_FOLDS, random_state=self.random_state)
        lasso.fit(self.X_dev, self.y_dev)
        selector = SelectFromModel(lasso, prefit=True)
        self.selected_features_idx = selector.get_support(indices=True)
        self.X_dev_selected = selector.transform(self.X_dev)

        if self.X_eval is not None:
            self.X_eval_selected = self.X_eval[:, self.selected_features_idx]

        print(f"Selected {self.X_dev_selected.shape[1]} features out of {self.X_dev.shape[1]}")
        return self.X_dev_selected, self.selected_features_idx

    # =========================
    # 3. FS MODELS (NO TUNING)
    # =========================
    def train_and_evaluate_fs_models(self, model_dir="../models"):
        fs_models = {
            'ElasticNet': ElasticNet(),
            'SVR': SVR(),
            'BayesianRidge': BayesianRidge()
        }

        for name, model in fs_models.items():
            print(f"\nTraining {name} on selected features...")
            model.fit(self.X_dev_selected, self.y_dev)
            self._save_model(model, filename=f"{name}_FS.joblib", model_dir=model_dir)

            print(f"Evaluating {name} FS model on evaluation set...")
            summary, raw = self._evaluate_model_bootstrap(model, self.X_eval_selected, self.y_eval)
            self.fs_eval_results[name] = {'summary': summary, 'raw': raw}

            print(f"--- {name} (FS) Evaluation Summary ---")
            print(f"RMSE:  mean = {summary['RMSE']['mean']}, median = {summary['RMSE']['median']}, 95% CI = {summary['RMSE']['95% CI']}")
            print(f"MAE:   mean = {summary['MAE']['mean']}, median = {summary['MAE']['median']}, 95% CI = {summary['MAE']['95% CI']}")
            print(f"R²:    mean = {summary['R2']['mean']}, median = {summary['R2']['median']}, 95% CI = {summary['R2']['95% CI']}")

        return self.fs_eval_results

    # =========================
    # 4. MODEL TUNING
    # =========================
    def train_and_evaluate_tuned_models(self, param_grids, model_dir="../models"):
        for name, params in param_grids.items():
            print(f"\nTuning {name}...")
            model = self.baseline_models[name]
            grid = GridSearchCV(model, params, scoring=make_scorer(mean_squared_error, greater_is_better=False), cv=CV_FOLDS, n_jobs=-1)
            grid.fit(self.X_dev_selected, self.y_dev)

            best_model = grid.best_estimator_
            self.tuned_models[name] = best_model
            self._save_model(best_model, filename=f"{name}_FS_TUNED.joblib", model_dir=model_dir)

            print(f"Evaluating tuned {name} model on evaluation set...")
            summary, raw = self._evaluate_model_bootstrap(best_model, self.X_eval_selected, self.y_eval)
            self.tuned_eval_results[name] = {'summary': summary, 'raw': raw}

            print(f"--- {name} (FS + Tuning) Evaluation Summary ---")
            print(f"RMSE:  mean = {summary['RMSE']['mean']}, median = {summary['RMSE']['median']}, 95% CI = {summary['RMSE']['95% CI']}")
            print(f"MAE:   mean = {summary['MAE']['mean']}, median = {summary['MAE']['median']}, 95% CI = {summary['MAE']['95% CI']}")
            print(f"R²:    mean = {summary['R2']['mean']}, median = {summary['R2']['median']}, 95% CI = {summary['R2']['95% CI']}")

        return self.tuned_eval_results


    # =========================
    # 4. COMPARISON AND PLOTTING
    # =========================
    def compare_model_stages(self, save_path="../eval_results/all_eval_results.pkl"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        all_eval_results = {}
        for model_name in self.baseline_models:
            all_eval_results[model_name] = {
                'Baseline': self.baseline_eval_results[model_name]['raw'],
                'FS': self.fs_eval_results.get(model_name, {}).get('raw', {}),
                'Tuned': self.tuned_eval_results.get(model_name, {}).get('raw', {})
            }

        with open(save_path, "wb") as f:
            pickle.dump(all_eval_results, f)
        print(f"Saved evaluation results to {save_path}")

        return all_eval_results

    def plot_metric_boxplots(self, metric_name, results_dict, title):
        data = []
        for model_name, stages in results_dict.items():
            for stage, metrics in stages.items():
                if metric_name.lower() in metrics:
                    for value in metrics[metric_name.lower()]:
                        data.append({
                            'Model': model_name,
                            'Stage': stage,
                            'Value': value
                        })
        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Model', y='Value', hue='Stage', data=df)
        plt.title(f'{title} Comparison by Model and Stage')
        plt.ylabel(metric_name.upper())
        plt.grid(True)
        plt.show()

    # =========================
    # 5. FINAL PIPELINE MODEL
    # =========================
   

    def get_best_model_by_rmse(self):
        best_model_name = None
        lowest_rmse = float('inf')

        for model_name, result in self.tuned_eval_results.items():
            rmse_mean = result['summary']['RMSE']['mean']
            print(f"{model_name}: RMSE = {rmse_mean:.4f}")
            if rmse_mean < lowest_rmse:
                lowest_rmse = rmse_mean
                best_model_name = model_name

        print(f"\nBest overall model based on evaluation RMSE: {best_model_name}")

        # Show best model hyperparameters (clean and sorted)
        best_model = self.tuned_models.get(best_model_name)
        if best_model is not None:
            params = best_model.get_params()
            sorted_params = dict(sorted(params.items()))
            print("\nBest model hyperparameters:")
            print(json.dumps(sorted_params, indent=4))

        return best_model_name


    def move_best_models_to_final(self, model_names, source_dir='../models', target_dir='../final_models'):
        os.makedirs(target_dir, exist_ok=True)
        for name in model_names:
            src = os.path.join(source_dir, f'{name}_FS_TUNED.joblib')
            dst = os.path.join(target_dir, f'{name}.joblib')
            if os.path.exists(src):
                joblib.dump(joblib.load(src), dst)
                print(f"Copied {src} to {dst}")
            else:
                print(f"Model {src} not found.")

    # =========================
    # 6. TRAIN WINNER MODEL ON FULL DATASET WITH PIPELINE
    # =========================
    def train_and_save_winner_model(self, model_name, filename='winner.joblib', model_dir='../final_models'):
        if model_name not in self.tuned_models:
            raise ValueError(f"Model '{model_name}' not found in tuned_models. Make sure to tune and evaluate models first.")

        model = self.tuned_models[model_name]
        X_all = np.concatenate([self.X_dev_selected, self.X_eval_selected], axis=0)
        y_all = np.concatenate([self.y_dev, self.y_eval], axis=0)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])

        pipeline.fit(X_all, y_all)
        self._save_model(pipeline, filename=filename, model_dir=model_dir)
        print(f"Winner model ('{model_name}') trained on full dataset and saved with preprocessing included.")
        import optuna

#----------------------------------------------------------
## optional 1
#---------------------------------------------------------


    def train_and_evaluate_tuned_models_optuna(self, search_spaces, n_trials=50, timeout=600, model_dir="../models"):
        for model_name, space in search_spaces.items():
            print(f"\nTuning {model_name} with Optuna...")

            def objective(trial):
                params = {}
                for param_name, spec in space.items():
                    if spec[0] == 'loguniform':
                        params[param_name] = trial.suggest_float(param_name, spec[1], spec[2], log=True)
                    elif spec[0] == 'uniform':
                        params[param_name] = trial.suggest_float(param_name, spec[1], spec[2])
                    elif spec[0] == 'int':
                        params[param_name] = trial.suggest_int(param_name, spec[1], spec[2])
                    elif spec[0] == 'categorical':
                        params[param_name] = trial.suggest_categorical(param_name, spec[1])
                    elif spec[0] == 'fixed':
                        params[param_name] = spec[1]

                if model_name == 'ElasticNet':
                    model = ElasticNet(**params)
                elif model_name == 'SVR':
                    model = SVR(**params)
                elif model_name == 'BayesianRidge':
                    model = BayesianRidge(**params)
                else:
                    raise ValueError(f"Unsupported model: {model_name}")

                preds = cross_val_predict(model, self.X_dev_selected, self.y_dev, cv=CV_FOLDS)
                return mean_squared_error(self.y_dev, preds)

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials, timeout=timeout)

            best_params = study.best_params
            print(f"Best hyperparameters for {model_name}: {best_params}")

            # Train final model with best params
            if model_name == 'ElasticNet':
                best_model = ElasticNet(**best_params)
            elif model_name == 'SVR':
                best_model = SVR(**best_params)
            elif model_name == 'BayesianRidge':
                best_model = BayesianRidge(**best_params)

            best_model.fit(self.X_dev_selected, self.y_dev)
            self.tuned_models[model_name] = best_model
            self._save_model(best_model, filename=f"{model_name}_FS_TUNED.joblib", model_dir=model_dir)

            print(f"Evaluating tuned {model_name} model on evaluation set...")
            summary, raw = self._evaluate_model_bootstrap(best_model, self.X_eval_selected, self.y_eval)
            self.tuned_eval_results[model_name] = {'summary': summary, 'raw': raw}

            print(f"--- {model_name} (FS + Optuna Tuning) Evaluation Summary ---")
            print(f"RMSE:  mean = {summary['RMSE']['mean']}, median = {summary['RMSE']['median']}, 95% CI = {summary['RMSE']['95% CI']}")
            print(f"MAE:   mean = {summary['MAE']['mean']}, median = {summary['MAE']['median']}, 95% CI = {summary['MAE']['95% CI']}")
            print(f"R²:    mean = {summary['R2']['mean']}, median = {summary['R2']['median']}, 95% CI = {summary['R2']['95% CI']}")

        return self.tuned_eval_results

