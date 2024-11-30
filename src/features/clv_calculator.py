import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter, GammaGammaFitter
import logging
import yaml

logger = logging.getLogger(__name__)

class CLVCalculator:
    """Calculates Customer Lifetime Value using probabilistic models."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the CLV calculator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.bgf_model = None
        self.ggf_model = None
    
    def fit_models(self, rfm: pd.DataFrame):
        """Fit BG/NBD and Gamma-Gamma models."""
        logger.info("Fitting BG/NBD model...")
        self.bgf_model = BetaGeoFitter(
            penalizer_coef=self.config['clv_calculation']['penalizer_coef']
        )
        self.bgf_model.fit(
            rfm['frequency'],
            rfm['recency'],
            rfm['T']
        )
        
        # Only fit Gamma-Gamma model for customers with purchases
        mask = rfm['frequency'] > 0
        
        logger.info("Fitting Gamma-Gamma model...")
        self.ggf_model = GammaGammaFitter(
            penalizer_coef=self.config['clv_calculation']['penalizer_coef']
        )
        self.ggf_model.fit(
            rfm.loc[mask, 'frequency'],
            rfm.loc[mask, 'monetary_avg']
        )
    
    def predict_clv(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Predict customer lifetime value."""
        if self.bgf_model is None or self.ggf_model is None:
            raise ValueError("Models must be fit before predicting CLV")
        
        # Predict future transactions
        logger.info("Predicting future transactions...")
        predicted_transactions = self.bgf_model.predict(
            self.config['clv_calculation']['time_period'],
            rfm['frequency'],
            rfm['recency'],
            rfm['T']
        )
        
        # Calculate expected average profit
        logger.info("Calculating customer lifetime value...")
        clv = self.ggf_model.customer_lifetime_value(
            self.bgf_model,
            rfm['frequency'],
            rfm['recency'],
            rfm['T'],
            rfm['monetary_avg'],
            time=self.config['clv_calculation']['time_period'],
            discount_rate=self.config['clv_calculation']['discount_rate']
        )
        
        # Add predictions to RFM dataframe
        rfm['predicted_transactions'] = predicted_transactions
        rfm['clv'] = clv
        
        return rfm
    
    def get_clv_segments(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Segment customers based on CLV predictions."""
        rfm['clv_segment'] = pd.qcut(
            rfm['clv'],
            q=4,
            labels=['Low Value', 'Medium Value', 'High Value', 'Top Value']
        )
        
        return rfm
    
    def get_clv_summary(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for CLV segments."""
        summary = rfm.groupby('clv_segment').agg({
            'clv': ['count', 'mean', 'min', 'max', 'sum'],
            'frequency': 'mean',
            'monetary_avg': 'mean'
        })
        
        # Calculate percentage of total CLV
        summary[('clv', 'percentage')] = (
            summary[('clv', 'sum')] / summary[('clv', 'sum')].sum() * 100
        )
        
        return summary
