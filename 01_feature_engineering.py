"""
CreditNirvana PS2: Feature Engineering Module
==============================================

Build features from raw data for RPC prediction model.
Handles:
  - Phone-level features (source, relation, priority)
  - Account-level features (DPD, bureau score, ability-to-pay)
  - Dial history features (network response patterns, success rate)
  - Temporal features (time-of-day, day-of-week)

Usage:
    fe = FeatureEngineer(data)
    X, y = fe.build_training_data()
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Build features for RPC prediction"""

    def __init__(self, data: dict):
        """
        Args:
            data: Dictionary of loaded DataFrames from DataLoader
        """
        self.data = data
        self.accounts = data['accounts']
        self.phones = data['phones']
        self.dial_attempts = data['dial_attempts']
        self.addresses = data['addresses']
        self.field_visits = data['field_visits']
        self.verified_contacts = data['verified_contact_points']
        self.splits = data['splits']

        logger.info("FeatureEngineer initialized")

    def build_training_data(self):
        """Build complete feature set with labels"""
        logger.info("Building training data...")

        # Start with verified contacts (ground truth)
        df = self.verified_contacts.copy()

        # Create binary target: 1 if borrower_number, 0 otherwise
        df['is_rpc'] = (df['verified_status'] == 'borrower_number').astype(int)

        # Add phone features
        df = df.merge(self.phones[['phone_id', 'source', 'relation_recorded', 'priority_slot']],
                     on='phone_id', how='left')

        # Add account features
        df = df.merge(self.accounts[['account_id', 'bucket_start', 'dpd_start', 'outstanding',
                                      'bureau_score_band', 'ability_to_pay_estimate', 'portfolio',
                                      'income_type', 'preferred_language']],
                     on='account_id', how='left')

        # Add dial history features
        df = self._add_dial_history_features(df)

        # Add train/val/test split
        df = df.merge(self.splits[['account_id', 'split']], on='account_id', how='left')

        # Encode categorical variables
        df = self._encode_categoricals(df)

        logger.info(f"✓ Training data built: {len(df)} rows × {len(df.columns)} columns")
        logger.info(f"  Positive (RPC): {df['is_rpc'].sum()} ({df['is_rpc'].mean()*100:.1f}%)")
        logger.info(f"  Negative: {(1-df['is_rpc']).sum()} ({(1-df['is_rpc']).mean()*100:.1f}%)")

        # Separate features and target
        X = df.drop(['phone_id', 'account_id', 'verified_status', 'is_rpc', 'verified_date', 'split'], axis=1)
        y = df['is_rpc']
        splits = df['split']

        return X, y, splits, df

    def _add_dial_history_features(self, df):
        """Extract features from call history for each phone"""
        logger.info("Extracting dial history features...")

        dial = self.dial_attempts.copy()

        # Convert timestamps
        dial['attempt_ts'] = pd.to_datetime(dial['attempt_ts'])
        dial['hour'] = dial['attempt_ts'].dt.hour
        dial['day_of_week'] = dial['attempt_ts'].dt.dayofweek

        # Group by phone_id to get history
        phone_history = dial.groupby('phone_id').agg({
            'attempt_id': 'count',  # total attempts
            'network_response': lambda x: (x == 'answered').sum(),  # answer count
            'disposition': lambda x: (x.str.contains('rpc', case=False, na=False)).sum(),  # RPC count
            'talk_duration_s': 'mean',  # avg talk duration
            'ring_duration_s': 'mean',  # avg ring duration
        }).rename(columns={
            'attempt_id': 'total_attempts',
            'network_response': 'answer_count',
            'disposition': 'rpc_count',
            'talk_duration_s': 'avg_talk_duration',
            'ring_duration_s': 'avg_ring_duration'
        }).reset_index()

        # Calculate success rates
        phone_history['answer_rate'] = phone_history['answer_count'] / phone_history['total_attempts']
        phone_history['rpc_rate'] = phone_history['rpc_count'] / phone_history['total_attempts']

        # Get latest network response
        latest_attempt = dial.sort_values('attempt_ts').groupby('phone_id').tail(1)[['phone_id', 'network_response']].rename(
            columns={'network_response': 'latest_network_response'}
        )
        phone_history = phone_history.merge(latest_attempt, on='phone_id', how='left')

        # Merge into main df
        df = df.merge(phone_history, on='phone_id', how='left')

        # Fill NaN (phones with no dial history)
        df['total_attempts'] = df['total_attempts'].fillna(0)
        df['answer_count'] = df['answer_count'].fillna(0)
        df['rpc_count'] = df['rpc_count'].fillna(0)
        df['answer_rate'] = df['answer_rate'].fillna(0)
        df['rpc_rate'] = df['rpc_rate'].fillna(0)
        df['avg_talk_duration'] = df['avg_talk_duration'].fillna(0)
        df['avg_ring_duration'] = df['avg_ring_duration'].fillna(0)
        df['latest_network_response'] = df['latest_network_response'].fillna('no_history')

        logger.info(f"✓ Dial history features added")
        return df

    def _encode_categoricals(self, df):
        """Encode categorical variables"""
        logger.info("Encoding categorical features...")

        # One-hot encode with drop_first to avoid collinearity
        categorical_cols = ['source', 'relation_recorded', 'bucket_start', 'bureau_score_band',
                           'portfolio', 'income_type', 'preferred_language', 'latest_network_response']

        for col in categorical_cols:
            if col in df.columns:
                # Get dummies
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(col, axis=1)

        # Convert boolean columns to int
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df[col] = df[col].astype(int)

        logger.info(f"✓ Categorical encoding complete. Final shape: {df.shape}")
        return df

    def get_feature_importance_baseline(self, X, y):
        """Simple feature importance using correlation with target"""
        logger.info("Computing feature correlations...")

        # For numeric columns, compute correlation with target
        X_numeric = X.select_dtypes(include=[np.number])
        correlations = X_numeric.corrwith(y).abs().sort_values(ascending=False)

        logger.info("\nTop 15 Features (Correlation with RPC):")
        print(correlations.head(15).to_string())

        return correlations

    def get_class_weights(self, y):
        """Compute class weights to handle imbalance"""
        from sklearn.utils.class_weight import compute_class_weight

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        logger.info(f"\nClass Weights (for imbalance handling):")
        logger.info(f"  Class 0 (Not RPC): {class_weight_dict[0]:.3f}")
        logger.info(f"  Class 1 (RPC): {class_weight_dict[1]:.3f}")

        return class_weight_dict


def build_features(data: dict):
    """Convenience function to build features"""
    fe = FeatureEngineer(data)
    X, y, splits, df_full = fe.build_training_data()
    return X, y, splits, df_full, fe


if __name__ == "__main__":
    from data_loader import load_data

    # Load data
    data = load_data()

    # Build features
    X, y, splits, df_full, fe = build_features(data)

    # Print summary
    print("\n" + "="*80)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*80)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Target balance: {y.mean()*100:.1f}% positive")
    print(f"\nSplit distribution:\n{splits.value_counts()}")

    # Feature importance
    print("\n" + "="*80)
    fe.get_feature_importance_baseline(X, y)

    # Class weights
    class_weights = fe.get_class_weights(y)
