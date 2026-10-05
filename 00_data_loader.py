"""
CreditNirvana PS2: Data Loading Module
=====================================

Load and preprocess all 11 CSV files for modeling.
Handles missing values, data type conversions, and basic validation.

Usage:
    loader = DataLoader()
    accounts, phones, dial_attempts, addresses, field_visits = loader.load_all()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Load and manage all CreditNirvana PS2 datasets"""

    def __init__(self, base_path: str = r"F:\ishu\omni route"):
        self.base_path = Path(base_path)
        self.shared_path = self.base_path / "shared-20261005T153334Z-1-001" / "shared"

        # Define file paths
        self.files = {
            'accounts': self.shared_path / 'accounts.csv',
            'phones': self.base_path / 'phones.csv',
            'addresses': self.shared_path / 'addresses.csv',
            'dial_attempts': self.shared_path / 'dial_attempts.csv',
            'field_visits': self.shared_path / 'field_visits.csv',
            'payments': self.shared_path / 'payments.csv',
            'agents': self.shared_path / 'agents.csv',
            'lenders': self.shared_path / 'lenders.csv',
            'skip_traces': self.base_path / 'skip_traces.csv',
            'verified_contact_points': self.base_path / 'verified_contact_points.csv',
            'splits': self.shared_path / 'splits.csv',
        }

        self.data = {}

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all 11 CSV files"""
        logger.info("Loading all data files...")

        for name, path in self.files.items():
            if not path.exists():
                logger.warning(f"File not found: {path}")
                continue

            try:
                df = pd.read_csv(path)
                self.data[name] = df
                logger.info(f"✓ {name}: {len(df):,} rows × {len(df.columns)} cols")
            except Exception as e:
                logger.error(f"✗ Error loading {name}: {e}")

        return self.data

    def get_summary(self) -> pd.DataFrame:
        """Print data summary"""
        summary = []
        for name, df in self.data.items():
            summary.append({
                'File': name,
                'Rows': len(df),
                'Columns': len(df.columns),
                'Memory (MB)': df.memory_usage(deep=True).sum() / 1024**2
            })

        summary_df = pd.DataFrame(summary)
        logger.info("\n" + summary_df.to_string(index=False))
        return summary_df

    def get_missing_values(self, name: str) -> pd.Series:
        """Get missing value counts for a dataset"""
        if name not in self.data:
            logger.error(f"Dataset {name} not loaded")
            return None

        df = self.data[name]
        missing = df.isnull().sum()
        return missing[missing > 0]

    def get_dtypes(self, name: str) -> pd.Series:
        """Get data types for a dataset"""
        if name not in self.data:
            logger.error(f"Dataset {name} not loaded")
            return None

        return self.data[name].dtypes

    def validate_keys(self) -> bool:
        """Validate that key columns exist and relationships are valid"""
        logger.info("Validating data relationships...")

        # Check accounts-phones relationship
        accounts = self.data['accounts']
        phones = self.data['phones']

        account_ids_accounts = set(accounts['account_id'].unique())
        account_ids_phones = set(phones['account_id'].unique())

        orphan_phones = account_ids_phones - account_ids_accounts
        if orphan_phones:
            logger.warning(f"⚠️ {len(orphan_phones)} phones reference non-existent accounts")

        # Check dial_attempts-phones relationship
        dial_attempts = self.data['dial_attempts']
        phone_ids_dial = set(dial_attempts['phone_id'].unique())
        phone_ids_phones = set(phones['phone_id'].unique())

        orphan_dials = phone_ids_dial - phone_ids_phones
        if orphan_dials:
            logger.warning(f"⚠️ {len(orphan_dials)} dial attempts reference non-existent phones")

        logger.info("✓ Validation complete")
        return True


def load_data(base_path: str = r"F:\ishu\omni route") -> Dict[str, pd.DataFrame]:
    """Convenience function to load all data at once"""
    loader = DataLoader(base_path)
    data = loader.load_all()
    loader.get_summary()
    loader.validate_keys()
    return data


if __name__ == "__main__":
    # Quick test
    data = load_data()

    # Print sample of each dataset
    print("\n" + "="*80)
    print("SAMPLE DATA")
    print("="*80)

    for name, df in data.items():
        print(f"\n{name.upper()} (first 3 rows):")
        print(df.head(3))
        print(f"Shape: {df.shape}")
