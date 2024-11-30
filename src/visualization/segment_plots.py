import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CustomerAnalyticsVisualizer:
    """Creates visualizations for customer analytics insights."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the visualizer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set style parameters
        plt.style.use(self.config['visualization']['style'])
        self.figure_size = self.config['visualization']['figure_size']
        self.dpi = self.config['visualization']['dpi']
        
        # Create figures directory if it doesn't exist
        self.figures_dir = Path(self.config['paths']['figures_dir'])
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_rfm_distributions(self, rfm_data: pd.DataFrame):
        """Plot distributions of RFM metrics."""
        fig, axes = plt.subplots(1, 3, figsize=self.figure_size)
        
        # Recency distribution
        sns.histplot(data=rfm_data, x='recency', ax=axes[0])
        axes[0].set_title('Recency Distribution')
        axes[0].set_xlabel('Days Since Last Purchase')
        
        # Frequency distribution
        sns.histplot(data=rfm_data, x='frequency', ax=axes[1])
        axes[1].set_title('Frequency Distribution')
        axes[1].set_xlabel('Number of Purchases')
        
        # Monetary distribution
        sns.histplot(data=rfm_data, x='monetary_avg', ax=axes[2])
        axes[2].set_title('Monetary Distribution')
        axes[2].set_xlabel('Average Purchase Value')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'rfm_distributions.png', dpi=self.dpi)
        plt.close()
    
    def plot_segment_characteristics(self, rfm_data: pd.DataFrame):
        """Plot characteristics of customer segments."""
        plt.figure(figsize=self.figure_size)
        
        segment_stats = rfm_data.groupby('Customer_Segment').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary_avg': 'mean'
        }).round(2)
        
        sns.heatmap(segment_stats, annot=True, cmap=self.config['visualization']['color_palette'])
        plt.title('Segment Characteristics')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'segment_characteristics.png', dpi=self.dpi)
        plt.close()
    
    def plot_clv_distribution(self, rfm_data: pd.DataFrame):
        """Plot CLV distribution and segments."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size)
        
        # CLV distribution
        sns.histplot(data=rfm_data, x='clv', ax=ax1)
        ax1.set_title('Customer Lifetime Value Distribution')
        ax1.set_xlabel('Predicted CLV')
        
        # CLV by segment
        sns.boxplot(data=rfm_data, x='clv_segment', y='clv', ax=ax2)
        ax2.set_title('CLV by Segment')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'clv_analysis.png', dpi=self.dpi)
        plt.close()
    
    def plot_retention_matrix(self, transactions: pd.DataFrame):
        """Create customer retention matrix."""
        # Calculate retention by cohort
        cohort_data = self._prepare_cohort_data(transactions)
        retention_matrix = self._calculate_retention_matrix(cohort_data)
        
        plt.figure(figsize=self.figure_size)
        sns.heatmap(retention_matrix, annot=True, fmt='.0%', cmap='YlOrRd')
        plt.title('Customer Retention by Cohort')
        plt.xlabel('Months Since First Purchase')
        plt.ylabel('Cohort')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'retention_matrix.png', dpi=self.dpi)
        plt.close()
    
    def _prepare_cohort_data(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Prepare transaction data for cohort analysis."""
        transactions = transactions.copy()
        transactions['cohort'] = transactions['transaction_date'].dt.to_period('M')
        transactions['purchase_month'] = transactions['transaction_date'].dt.to_period('M')
        
        return transactions
    
    def _calculate_retention_matrix(self, cohort_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate retention rates for cohort analysis."""
        # Count unique customers by cohort and purchase month
        grouping = cohort_data.groupby(['cohort', 'purchase_month'])['customer_id'].nunique()
        cohort_data = grouping.reset_index()
        
        # Calculate periods since first purchase
        cohort_data['period_number'] = (
            cohort_data.purchase_month - cohort_data.cohort).apply(lambda x: x.n)
        
        # Create retention matrix
        retention_matrix = cohort_data.pivot_table(
            index='cohort',
            columns='period_number',
            values='customer_id',
            aggfunc='sum'
        )
        
        # Calculate retention rates
        retention_matrix = retention_matrix.divide(retention_matrix[0], axis=0)
        
        return retention_matrix
