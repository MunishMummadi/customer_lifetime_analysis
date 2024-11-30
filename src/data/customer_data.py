import pandas as pd
import numpy as np
from datetime import datetime
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class CustomerDataGenerator:
    """Generates synthetic customer transaction data for analysis."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data generator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        np.random.seed(self.config['data']['random_seed'])
        self.n_customers = self.config['data']['n_customers']
    
    def generate_customer_base(self) -> pd.DataFrame:
        """Generate basic customer demographic data."""
        customers = pd.DataFrame({
            'customer_id': range(self.n_customers),
            'first_purchase': pd.date_range(
                start=self.config['data']['date_range']['start'],
                periods=self.n_customers,
                freq='D'
            ).map(lambda x: x + pd.Timedelta(days=np.random.randint(0, 30))),
            'age': np.random.normal(45, 15, self.n_customers).round().clip(18, 90),
            'gender': np.random.choice(['M', 'F'], self.n_customers),
            'income_segment': np.random.choice(['Low', 'Medium', 'High'], self.n_customers,
                                            p=[0.3, 0.5, 0.2])
        })
        return customers
    
    def generate_transactions(self, customers: pd.DataFrame) -> pd.DataFrame:
        """Generate transaction history for customers."""
        transactions = []
        
        for _, customer in customers.iterrows():
            # Number of transactions follows a Poisson distribution
            n_transactions = np.random.poisson(5)
            
            if n_transactions > 0:
                for _ in range(n_transactions):
                    # Days since first purchase follows exponential distribution
                    days_since_first = np.random.exponential(30)
                    transaction_date = customer['first_purchase'] + pd.Timedelta(days=days_since_first)
                    
                    # Transaction amount based on income segment
                    base_amount = {
                        'Low': 50,
                        'Medium': 100,
                        'High': 200
                    }[customer['income_segment']]
                    
                    amount = np.random.lognormal(
                        np.log(base_amount),
                        0.5
                    )
                    
                    transactions.append({
                        'customer_id': customer['customer_id'],
                        'transaction_date': transaction_date,
                        'amount': round(amount, 2),
                        'category': np.random.choice(
                            ['Electronics', 'Clothing', 'Food', 'Home', 'Other'],
                            p=[0.2, 0.3, 0.25, 0.15, 0.1]
                        )
                    })
        
        return pd.DataFrame(transactions)
    
    def generate_dataset(self) -> tuple:
        """Generate complete customer and transaction datasets."""
        logger.info("Generating customer base data...")
        customers = self.generate_customer_base()
        
        logger.info("Generating transaction history...")
        transactions = self.generate_transactions(customers)
        
        logger.info(f"Generated {len(customers)} customers and {len(transactions)} transactions")
        return customers, transactions
    
    def save_data(self, customers: pd.DataFrame, transactions: pd.DataFrame):
        """Save generated data to files."""
        data_dir = Path(self.config['paths']['data_dir'])
        data_dir.mkdir(exist_ok=True)
        
        customers.to_csv(data_dir / 'customers.csv', index=False)
        transactions.to_csv(data_dir / 'transactions.csv', index=False)
        logger.info("Data saved successfully")

class CustomerDataLoader:
    """Loads and preprocesses customer data for analysis."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data loader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_data(self) -> tuple:
        """Load customer and transaction data from files."""
        data_dir = Path(self.config['paths']['data_dir'])
        
        customers = pd.read_csv(
            data_dir / 'customers.csv',
            parse_dates=['first_purchase']
        )
        
        transactions = pd.read_csv(
            data_dir / 'transactions.csv',
            parse_dates=['transaction_date']
        )
        
        return customers, transactions
    
    def preprocess_data(self, customers: pd.DataFrame, transactions: pd.DataFrame) -> tuple:
        """Preprocess the data for analysis."""
        # Remove duplicates
        transactions = transactions.drop_duplicates()
        
        # Sort transactions by date
        transactions = transactions.sort_values('transaction_date')
        
        # Ensure all transactions have valid customer IDs
        transactions = transactions[
            transactions['customer_id'].isin(customers['customer_id'])
        ]
        
        return customers, transactions
