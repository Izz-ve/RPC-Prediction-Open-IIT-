"""
CreditNirvana PS2: Model Training Module
========================================

Train and evaluate RPC prediction models.
Models included:
  - Logistic Regression (baseline, interpretable)
  - XGBoost (gradient boosting, strong performance)
  - LightGBM (fast, memory efficient)

Usage:
    trainer = ModelTrainer(X, y, splits)
    results = trainer.train_all_models()
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    average_precision_score, precision_score, recall_score
)
import logging
from typing import Dict, Tuple
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. Install with: pip install lightgbm")


class ModelTrainer:
    """Train and evaluate RPC prediction models"""

    def __init__(self, X: pd.DataFrame, y: pd.Series, splits: pd.Series,
                 class_weight: Dict = None, random_state: int = 42):
        """
        Args:
            X: Feature matrix
            y: Target (1 = RPC, 0 = not RPC)
            splits: Train/val/test assignment for each sample
            class_weight: Class weights for imbalance (optional)
            random_state: Random seed
        """
        self.X = X
        self.y = y
        self.splits = splits
        self.class_weight = class_weight or {0: 1.0, 1: 1.0}
        self.random_state = random_state

        # Split data
        self.X_train = X[splits == 'train']
        self.y_train = y[splits == 'train']

        self.X_val = X[splits == 'validation']
        self.y_val = y[splits == 'validation']

        self.X_test = X[splits == 'test']
        self.y_test = y[splits == 'test']

        self.models = {}
        self.results = {}

        logger.info(f"Train set: {len(self.y_train)} ({self.y_train.mean()*100:.1f}% positive)")
        logger.info(f"Val set: {len(self.y_val)} ({self.y_val.mean()*100:.1f}% positive)")
        logger.info(f"Test set: {len(self.y_test)} ({self.y_test.mean()*100:.1f}% positive)")

    def train_logistic_regression(self, C: float = 1.0) -> Tuple[LogisticRegression, Dict]:
        """Train Logistic Regression baseline"""
        logger.info("\n" + "="*80)
        logger.info("Training Logistic Regression...")
        logger.info("="*80)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_val_scaled = scaler.transform(self.X_val)
        X_test_scaled = scaler.transform(self.X_test)

        # Train
        model = LogisticRegression(
            C=C,
            class_weight='balanced',
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs'
        )
        model.fit(X_train_scaled, self.y_train)

        # Evaluate
        results = self._evaluate_model(
            model, X_train_scaled, X_val_scaled, X_test_scaled,
            self.y_train, self.y_val, self.y_test,
            'Logistic Regression'
        )

        self.models['LogisticRegression'] = (model, scaler)
        self.results['LogisticRegression'] = results

        return model, results

    def train_xgboost(self, **xgb_params) -> Tuple:
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost not available. Install with: pip install xgboost")
            return None, None

        logger.info("\n" + "="*80)
        logger.info("Training XGBoost...")
        logger.info("="*80)

        # Default params
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'random_state': self.random_state,
            'eval_metric': 'logloss',
            'verbose': 0,
        }
        params.update(xgb_params)

        # Scale class weights
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        params['scale_pos_weight'] = scale_pos_weight

        # Train
        model = xgb.XGBClassifier(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            early_stopping_rounds=20,
            verbose=False
        )

        # Evaluate
        results = self._evaluate_model(
            model, self.X_train, self.X_val, self.X_test,
            self.y_train, self.y_val, self.y_test,
            'XGBoost'
        )

        self.models['XGBoost'] = model
        self.results['XGBoost'] = results

        return model, results

    def train_lightgbm(self, **lgb_params) -> Tuple:
        """Train LightGBM model"""
        if not LIGHTGBM_AVAILABLE:
            logger.error("LightGBM not available. Install with: pip install lightgbm")
            return None, None

        logger.info("\n" + "="*80)
        logger.info("Training LightGBM...")
        logger.info("="*80)

        # Default params
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary',
            'random_state': self.random_state,
            'verbose': -1,
        }
        params.update(lgb_params)

        # Scale class weights
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        params['scale_pos_weight'] = scale_pos_weight

        # Train
        model = lgb.LGBMClassifier(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            early_stopping_rounds=20,
        )

        # Evaluate
        results = self._evaluate_model(
            model, self.X_train, self.X_val, self.X_test,
            self.y_train, self.y_val, self.y_test,
            'LightGBM'
        )

        self.models['LightGBM'] = model
        self.results['LightGBM'] = results

        return model, results

    def _evaluate_model(self, model, X_train, X_val, X_test, y_train, y_val, y_test, name):
        """Evaluate model on all sets"""

        # Predictions
        y_train_pred_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_train)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)

        y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
        y_val_pred = (y_val_pred_proba >= 0.5).astype(int)
        y_test_pred = (y_test_pred_proba >= 0.5).astype(int)

        # Metrics
        results = {
            'train': {
                'auc': roc_auc_score(y_train, y_train_pred_proba),
                'ap': average_precision_score(y_train, y_train_pred_proba),
                'f1': f1_score(y_train, y_train_pred),
                'precision': precision_score(y_train, y_train_pred),
                'recall': recall_score(y_train, y_train_pred),
            },
            'val': {
                'auc': roc_auc_score(y_val, y_val_pred_proba),
                'ap': average_precision_score(y_val, y_val_pred_proba),
                'f1': f1_score(y_val, y_val_pred),
                'precision': precision_score(y_val, y_val_pred),
                'recall': recall_score(y_val, y_val_pred),
            },
            'test': {
                'auc': roc_auc_score(y_test, y_test_pred_proba),
                'ap': average_precision_score(y_test, y_test_pred_proba),
                'f1': f1_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred),
                'recall': recall_score(y_test, y_test_pred),
            },
            'predictions': {
                'train': y_train_pred_proba,
                'val': y_val_pred_proba,
                'test': y_test_pred_proba,
            }
        }

        # Log results
        logger.info(f"\n{name} Results:")
        logger.info("-" * 60)
        for split in ['train', 'val', 'test']:
            logger.info(f"\n{split.upper()}:")
            for metric, value in results[split].items():
                logger.info(f"  {metric:12s}: {value:.4f}")

        return results

    def train_all_models(self) -> Dict:
        """Train all available models"""
        logger.info("\n" + "="*80)
        logger.info("STARTING MODEL TRAINING")
        logger.info("="*80)

        # Logistic Regression
        self.train_logistic_regression()

        # XGBoost
        if XGBOOST_AVAILABLE:
            self.train_xgboost()

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            self.train_lightgbm()

        return self.results

    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models"""
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)

        comparison = []
        for name, results in self.results.items():
            comparison.append({
                'Model': name,
                'Train AUC': results['train']['auc'],
                'Val AUC': results['val']['auc'],
                'Test AUC': results['test']['auc'],
                'Val AP': results['val']['ap'],
                'Test AP': results['test']['ap'],
                'Test F1': results['test']['f1'],
            })

        df = pd.DataFrame(comparison).sort_values('Test AUC', ascending=False)
        logger.info("\n" + df.to_string(index=False))

        return df

    def get_best_model(self):
        """Get best model by test AUC"""
        best_name = None
        best_auc = -1

        for name, results in self.results.items():
            if results['test']['auc'] > best_auc:
                best_auc = results['test']['auc']
                best_name = name

        logger.info(f"\n✓ Best model: {best_name} (Test AUC: {best_auc:.4f})")
        return best_name, self.models[best_name]

    def save_model(self, model_name: str, filepath: str):
        """Save model to disk"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return False

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.models[model_name], f)
            logger.info(f"✓ Model saved: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import build_features

    # Load and build features
    data = load_data()
    X, y, splits, df_full, fe = build_features(data)

    # Get class weights
    class_weights = fe.get_class_weights(y)

    # Train models
    trainer = ModelTrainer(X, y, splits, class_weight=class_weights)
    results = trainer.train_all_models()

    # Compare
    comparison = trainer.compare_models()

    # Get best
    best_name, best_model = trainer.get_best_model()
