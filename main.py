import logging
import yaml
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.data.customer_data import CustomerDataGenerator, CustomerDataLoader
from src.features.rfm_metrics import RFMCalculator
from src.features.clv_calculator import CLVCalculator
from src.visualization.segment_plots import CustomerAnalyticsVisualizer

def setup_logging(config):
    """Set up logging configuration."""
    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format']
    )
    logger = logging.getLogger(__name__)
    return logger

def generate_report(rfm_data: pd.DataFrame, output_path: Path):
    """Generate summary report in markdown format."""
    report = f"""
# Customer Lifetime Value Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
Total Customers Analyzed: {len(rfm_data)}

## Customer Segments
{rfm_data['Customer_Segment'].value_counts().to_markdown()}

## CLV Analysis
- Average CLV: ${rfm_data['clv'].mean():.2f}
- Median CLV: ${rfm_data['clv'].median():.2f}
- Total Expected CLV: ${rfm_data['clv'].sum():.2f}

## Segment Performance
{rfm_data.groupby('Customer_Segment').agg({
    'clv': ['count', 'mean', 'sum'],
    'frequency': 'mean',
    'monetary_avg': 'mean'
}).round(2).to_markdown()}

## Recommendations
1. Focus retention efforts on High-Value customers
2. Develop activation campaigns for Recent Customers
3. Create reactivation programs for Lost Customers
4. Implement upselling strategies for Loyal Customers
"""
    
    with open(output_path / 'analysis_report.md', 'w') as f:
        f.write(report)

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting Customer Lifetime Value Analysis")
    
    try:
        # Generate or load data
        data_path = Path(config['paths']['data_dir']) / 'customers.csv'
        if not data_path.exists():
            logger.info("Generating new customer data...")
            data_generator = CustomerDataGenerator()
            customers, transactions = data_generator.generate_dataset()
            data_generator.save_data(customers, transactions)
        else:
            logger.info("Loading existing customer data...")
            data_loader = CustomerDataLoader()
            customers, transactions = data_loader.load_data()
            customers, transactions = data_loader.preprocess_data(customers, transactions)
        
        # Calculate RFM metrics
        logger.info("Calculating RFM metrics...")
        rfm_calculator = RFMCalculator()
        rfm_data = rfm_calculator.calculate_rfm(transactions)
        rfm_data = rfm_calculator.add_rfm_scores(rfm_data)
        rfm_data = rfm_calculator.get_customer_segment(rfm_data)
        
        # Calculate CLV
        logger.info("Calculating Customer Lifetime Value...")
        clv_calculator = CLVCalculator()
        clv_calculator.fit_models(rfm_data)
        rfm_data = clv_calculator.predict_clv(rfm_data)
        rfm_data = clv_calculator.get_clv_segments(rfm_data)
        
        # Create visualizations
        logger.info("Generating visualizations...")
        visualizer = CustomerAnalyticsVisualizer()
        visualizer.plot_rfm_distributions(rfm_data)
        visualizer.plot_segment_characteristics(rfm_data)
        visualizer.plot_clv_distribution(rfm_data)
        visualizer.plot_retention_matrix(transactions)
        
        # Generate report
        logger.info("Generating analysis report...")
        reports_dir = Path(config['paths']['reports_dir'])
        reports_dir.mkdir(exist_ok=True)
        generate_report(rfm_data, reports_dir)
        
        # Save final results
        logger.info("Saving results...")
        rfm_data.to_csv(reports_dir / 'customer_analysis_results.csv', index=False)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()
