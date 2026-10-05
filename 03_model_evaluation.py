"""
CreditNirvana PS2: Model Evaluation & Insights
==============================================

Deep-dive analysis of model predictions.
Includes:
  - Feature importance (for tree-based models)
  - Calibration analysis
  - Error analysis
  - Per-segment performance
  - Threshold optimization

Usage:
    evaluator = ModelEvaluator(trainer, X, y, splits)
    evaluator.analyze_best_model()
"""

import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib/seaborn not installed. Plotting disabled.")


class ModelEvaluator:
    """Deep evaluation of RPC prediction models"""

    def __init__(self, trainer, X: pd.DataFrame, y: pd.Series,
                 splits: pd.Series, original_df: pd.DataFrame = None):
        """
        Args:
            trainer: ModelTrainer instance with trained models
            X: Feature matrix
            y: Target labels
            splits: Train/val/test assignment
            original_df: Original dataframe with phone_id, account_id for analysis
        """
        self.trainer = trainer
        self.X = X
        self.y = y
        self.splits = splits
        self.original_df = original_df

        self.X_test = X[splits == 'test']
        self.y_test = y[splits == 'test']

        logger.info("ModelEvaluator initialized")

    def analyze_best_model(self):
        """Full analysis of the best performing model"""
        logger.info("\n" + "="*80)
        logger.info("BEST MODEL ANALYSIS")
        logger.info("="*80)

        best_name, best_model = self.trainer.get_best_model()

        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            self.analyze_feature_importance(best_model)

        # Calibration
        self.analyze_calibration(best_model)

        # Threshold analysis
        self.analyze_threshold(best_model)

        # Per-segment performance
        if self.original_df is not None:
            self.analyze_segment_performance(best_model)

    def analyze_feature_importance(self, model, top_n: int = 15):
        """Analyze feature importance for tree-based models"""
        logger.info("\n" + "="*80)
        logger.info("FEATURE IMPORTANCE")
        logger.info("="*80)

        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None

        # Get feature importance
        importances = model.feature_importances_
        feature_names = self.X.columns

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'importance_pct': importances / importances.sum() * 100
        }).sort_values('importance', ascending=False)

        logger.info(f"\nTop {top_n} Features:")
        logger.info(importance_df.head(top_n).to_string(index=False))

        # Plot if available
        if PLOTTING_AVAILABLE:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df.head(top_n), x='importance_pct', y='feature', palette='viridis')
            plt.xlabel('Importance (%)')
            plt.title('Top Features for RPC Prediction')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
            logger.info("✓ Feature importance plot saved: feature_importance.png")

        return importance_df

    def analyze_calibration(self, model):
        """Analyze prediction calibration"""
        logger.info("\n" + "="*80)
        logger.info("CALIBRATION ANALYSIS")
        logger.info("="*80)

        # Get predictions
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(self.X_test)

        # Compute calibration
        prob_true, prob_pred = calibration_curve(self.y_test, y_pred_proba, n_bins=10)

        logger.info("\nCalibration Curve (Predicted Prob -> Actual Prob):")
        for i, (pred, true) in enumerate(zip(prob_pred, prob_true)):
            logger.info(f"  Bin {i+1}: Predicted {pred:.3f} -> Actual {true:.3f} (diff: {abs(pred-true):.3f})")

        # Check for over/under-confidence
        mean_diff = np.mean(np.abs(prob_pred - prob_true))
        logger.info(f"\nMean calibration error: {mean_diff:.4f}")
        if mean_diff < 0.05:
            logger.info("✓ Model is well-calibrated (error < 0.05)")
        elif mean_diff < 0.10:
            logger.info("⚠️ Model is reasonably calibrated (error 0.05-0.10)")
        else:
            logger.info("❌ Model is poorly calibrated (error > 0.10)")

        # Plot if available
        if PLOTTING_AVAILABLE:
            plt.figure(figsize=(8, 6))
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            plt.plot(prob_pred, prob_true, 'o-', label='Model Calibration')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('calibration_curve.png', dpi=100, bbox_inches='tight')
            logger.info("✓ Calibration curve plot saved: calibration_curve.png")

        return prob_true, prob_pred

    def analyze_threshold(self, model):
        """Analyze optimal prediction threshold"""
        logger.info("\n" + "="*80)
        logger.info("THRESHOLD OPTIMIZATION")
        logger.info("="*80)

        # Get predictions
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(self.X_test)

        # Test different thresholds
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred_proba)

        # F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]

        logger.info(f"\nOptimal Threshold Analysis:")
        logger.info(f"  Default threshold (0.5): F1 = {f1_scores[np.searchsorted(thresholds, 0.5)]:.4f}")
        logger.info(f"  Optimal threshold: {best_threshold:.3f}")
        logger.info(f"  Optimal F1 score: {best_f1:.4f}")

        # Show precision/recall at optimal
        y_pred_opt = (y_pred_proba >= best_threshold).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score
        opt_precision = precision_score(self.y_test, y_pred_opt)
        opt_recall = recall_score(self.y_test, y_pred_opt)
        opt_f1 = f1_score(self.y_test, y_pred_opt)

        logger.info(f"\nAt threshold {best_threshold:.3f}:")
        logger.info(f"  Precision: {opt_precision:.4f}")
        logger.info(f"  Recall: {opt_recall:.4f}")
        logger.info(f"  F1: {opt_f1:.4f}")

        return best_threshold

    def analyze_segment_performance(self, model):
        """Analyze performance by phone source and relation"""
        logger.info("\n" + "="*80)
        logger.info("SEGMENT PERFORMANCE ANALYSIS")
        logger.info("="*80)

        if self.original_df is None or len(self.original_df) == 0:
            logger.warning("Original dataframe not available for segment analysis")
            return

        # Get test predictions
        y_test_idx = self.splits[self.splits == 'test'].index
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(self.X_test)

        # Get original data for test set
        test_df = self.original_df.loc[y_test_idx].copy()
        test_df['predicted_prob'] = y_pred_proba
        test_df['actual'] = self.y_test.values

        # Performance by source
        logger.info("\nPerformance by Phone Source:")
        source_perf = test_df.groupby('source').agg({
            'predicted_prob': ['mean', 'count'],
            'actual': ['mean', 'sum']
        }).round(4)
        logger.info(source_perf.to_string())

        # Performance by relation
        logger.info("\nPerformance by Relation Recorded:")
        relation_perf = test_df.groupby('relation_recorded').agg({
            'predicted_prob': ['mean', 'count'],
            'actual': ['mean', 'sum']
        }).round(4)
        logger.info(relation_perf.to_string())


class ModelInference:
    """Make predictions on new data and generate actionable insights"""

    def __init__(self, model, scaler=None):
        """
        Args:
            model: Trained model
            scaler: Fitted StandardScaler (for LogisticRegression)
        """
        self.model = model
        self.scaler = scaler

    def predict_phone_quality(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Predict RPC probability for phones.

        Returns DataFrame with:
          - phone_id
          - predicted_rpc_prob: P(will reach borrower)
          - predicted_rpc: Binary prediction at threshold
          - rpc_quality: 'high' / 'medium' / 'low'
        """
        # Scale if needed
        X_pred = self.scaler.transform(X) if self.scaler else X

        # Predict
        proba = self.model.predict_proba(X_pred)[:, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(X_pred)
        pred = (proba >= threshold).astype(int)

        # Quality labels
        quality = pd.cut(proba, bins=[0, 0.33, 0.67, 1.0], labels=['low', 'medium', 'high'])

        result = pd.DataFrame({
            'predicted_rpc_prob': proba,
            'predicted_rpc': pred,
            'rpc_quality': quality
        })

        return result

    def rank_for_skiptrace(self, phone_features: pd.DataFrame,
                           expected_recovery: pd.Series,
                           skip_trace_cost: float = 95.0) -> pd.DataFrame:
        """
        Rank phones for skip-tracing by expected value.

        Value of Information = (1 - predicted_rpc_prob) × expected_recovery - skip_trace_cost

        Args:
            phone_features: Feature matrix for phones to rank
            expected_recovery: Expected collection amount for each account
            skip_trace_cost: Cost per skip-trace (default ₹95)

        Returns: DataFrame ranked by value, with recommendation
        """
        # Predict RPC probability
        proba = self.model.predict_proba(phone_features)[:, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(phone_features)

        # Probability that a skip-trace would find a valid number
        prob_invalid = 1 - proba

        # Expected value of skip-trace
        value_of_info = (prob_invalid * expected_recovery) - skip_trace_cost

        # Recommendation
        recommendation = pd.cut(value_of_info,
                                bins=[-np.inf, 0, 100, np.inf],
                                labels=['Skip - Not Worth', 'Skip - Low Priority', 'Skip - High Priority'])

        result = pd.DataFrame({
            'predicted_rpc_prob': proba,
            'prob_invalid': prob_invalid,
            'expected_recovery': expected_recovery,
            'value_of_skiptrace': value_of_info,
            'recommendation': recommendation
        }).sort_values('value_of_info', ascending=False)

        return result


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import build_features
    from model_training import ModelTrainer

    # Load and build
    data = load_data()
    X, y, splits, df_full, fe = build_features(data)
    class_weights = fe.get_class_weights(y)

    # Train
    trainer = ModelTrainer(X, y, splits, class_weight=class_weights)
    trainer.train_all_models()

    # Evaluate best
    evaluator = ModelEvaluator(trainer, X, y, splits, original_df=df_full)
    evaluator.analyze_best_model()
