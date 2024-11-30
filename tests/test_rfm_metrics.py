import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.features.rfm_metrics import RFMCalculator

@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    return pd.DataFrame({
        'customer_id': [1, 1, 2, 3, 3, 3],
        'transaction_date': [
            datetime(2024, 1, 1),
            datetime(2024, 1, 15),
            datetime(2024, 1, 20),
            datetime(2024, 1, 1),
            datetime(2024, 1, 10),
            datetime(2024, 2, 1)
        ],
        'amount': [100, 200, 150, 300, 400, 500]
    })

def test_calculate_rfm():
    """Test RFM calculation functionality."""
    calculator = RFMCalculator()
    transactions = sample_transactions()
    analysis_date = datetime(2024, 2, 15)
    
    rfm = calculator.calculate_rfm(transactions, analysis_date)
    
    assert len(rfm) == 3  # Should have 3 unique customers
    assert all(col in rfm.columns for col in ['recency', 'frequency', 'monetary_sum', 'monetary_avg'])
    
    # Check customer 1's metrics
    customer_1 = rfm[rfm['customer_id'] == 1]
    assert customer_1['frequency'].iloc[0] == 2
    assert customer_1['monetary_sum'].iloc[0] == 300
    assert customer_1['monetary_avg'].iloc[0] == 150

def test_add_rfm_scores():
    """Test RFM scoring functionality."""
    calculator = RFMCalculator()
    transactions = sample_transactions()
    rfm = calculator.calculate_rfm(transactions)
    rfm_scored = calculator.add_rfm_scores(rfm)
    
    assert all(col in rfm_scored.columns for col in ['R_score', 'F_score', 'M_score', 'RFM_Score'])
    assert all(1 <= score <= 4 for score in rfm_scored['R_score'])
    assert all(1 <= score <= 4 for score in rfm_scored['F_score'])
    assert all(1 <= score <= 4 for score in rfm_scored['M_score'])

def test_get_customer_segment():
    """Test customer segmentation functionality."""
    calculator = RFMCalculator()
    transactions = sample_transactions()
    rfm = calculator.calculate_rfm(transactions)
    rfm_scored = calculator.add_rfm_scores(rfm)
    segmented = calculator.get_customer_segment(rfm_scored)
    
    assert 'Customer_Segment' in segmented.columns
    assert all(isinstance(segment, str) for segment in segmented['Customer_Segment'])
