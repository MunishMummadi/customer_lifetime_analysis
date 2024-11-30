import pandas as pd
import numpy as np
from datetime import datetime
import logging
import yaml

logger = logging.getLogger(__name__)

class RFMCalculator:
    """Calculates RFM (Recency, Frequency, Monetary) metrics for customers."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the RFM calculator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def calculate_rfm(self, transactions: pd.DataFrame, analysis_date: datetime = None) -> pd.DataFrame:
        """Calculate RFM metrics for each customer."""
        if analysis_date is None:
            analysis_date = transactions['transaction_date'].max() + pd.Timedelta(days=1)
        
        logger.info(f"Calculating RFM metrics as of {analysis_date}")
        
        # Calculate recency, frequency, and monetary value for each customer
        rfm = transactions.groupby('customer_id').agg({
            'transaction_date': lambda x: (analysis_date - x.max()).days,  # Recency
            'customer_id': 'count',  # Frequency
            'amount': ['sum', 'mean']  # Monetary
        })
        
        # Flatten column names
        rfm.columns = ['recency', 'frequency', 'monetary_sum', 'monetary_avg']
        rfm = rfm.reset_index()
        
        # Add T (time since first purchase)
        first_purchase = transactions.groupby('customer_id')['transaction_date'].min()
        rfm['T'] = (analysis_date - first_purchase).dt.days
        
        logger.info("RFM metrics calculated successfully")
        return rfm
    
    def add_rfm_scores(self, rfm: pd.DataFrame, quartiles: bool = True) -> pd.DataFrame:
        """Add RFM scores (1-4) for each metric."""
        if quartiles:
            # Use quartiles for scoring
            r_labels = range(4, 0, -1)  # 4 is best (lowest recency)
            f_labels = range(1, 5)  # 4 is best (highest frequency)
            m_labels = range(1, 5)  # 4 is best (highest monetary)
            
            r_quartiles = pd.qcut(rfm['recency'], q=4, labels=r_labels)
            f_quartiles = pd.qcut(rfm['frequency'], q=4, labels=f_labels)
            m_quartiles = pd.qcut(rfm['monetary_avg'], q=4, labels=m_labels)
            
            rfm['R_score'] = r_quartiles
            rfm['F_score'] = f_quartiles
            rfm['M_score'] = m_quartiles
        else:
            # Use manual breaks (customize as needed)
            rfm['R_score'] = pd.cut(rfm['recency'], 
                                  bins=[0, 7, 30, 90, float('inf')],
                                  labels=[4, 3, 2, 1])
            
            rfm['F_score'] = pd.cut(rfm['frequency'],
                                  bins=[0, 2, 5, 10, float('inf')],
                                  labels=[1, 2, 3, 4])
            
            rfm['M_score'] = pd.cut(rfm['monetary_avg'],
                                  bins=[0, 100, 250, 500, float('inf')],
                                  labels=[1, 2, 3, 4])
        
        # Calculate RFM Score
        rfm['RFM_Score'] = rfm['R_score'].astype(str) + \
                          rfm['F_score'].astype(str) + \
                          rfm['M_score'].astype(str)
        
        return rfm
    
    def get_customer_segment(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Assign customer segments based on RFM scores."""
        def segment_customer(row):
            if row['RFM_Score'] in ['444', '434', '443', '433']:
                return 'Best Customers'
            elif row['R_score'] == 4:
                return 'Recent Customers'
            elif row['F_score'] == 4:
                return 'Loyal Customers'
            elif row['M_score'] == 4:
                return 'Big Spenders'
            elif row['R_score'] == 1:
                return 'Lost Customers'
            else:
                return 'Average Customers'
        
        rfm['Customer_Segment'] = rfm.apply(segment_customer, axis=1)
        return rfm
