"""
Forecasting Module
KDD PHASE 6: Trend Analysis

Time series forecasting for research trends:
- ARIMA forecasting
- Exponential smoothing
- Trend prediction
- Confidence intervals
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendForecaster:
    """
    Forecast research trends using time series models
    
    KDD Phase 6: Trend Forecasting
    Objective: Predict future trend patterns
    
    Models:
    - ARIMA: AutoRegressive Integrated Moving Average
    - Exponential Smoothing: For trends with seasonality
    """

    def __init__(self):
        """
        Initialize forecaster
        """
        self.models = {}
        self.forecasts = {}
        logger.info("Trend forecaster initialized")

    def prepare_timeseries(self, timeline: Dict[str, int]) -> Tuple[pd.Series, List[str]]:
        """
        Prepare time series data for forecasting
        
        Args:
            timeline: Dictionary mapping dates to frequencies
            
        Returns:
            Tuple of (pandas Series, list of dates)
        """
        # Sort by date
        dates = sorted(timeline.keys())
        values = [timeline[date] for date in dates]
        
        # Convert to pandas Series
        ts = pd.Series(values, index=pd.to_datetime(dates))
        
        logger.info(f"Time series prepared: {len(ts)} observations")
        return ts, dates

    def forecast_arima(self, timeline: Dict[str, int], 
                      periods: int = 12, order: Tuple = (1, 1, 1)) -> Dict:
        """
        Forecast using ARIMA model
        
        ARIMA Parameters:
        - p: AR (AutoRegressive) order
        - d: I (Integrated) order (differencing)
        - q: MA (Moving Average) order
        
        Args:
            timeline: Time series dictionary
            periods: Number of periods to forecast
            order: ARIMA order tuple (p, d, q)
            
        Returns:
            Dictionary with forecast results
        """
        try:
            ts, dates = self.prepare_timeseries(timeline)
            
            logger.info(f"Fitting ARIMA{order} model...")
            model = ARIMA(ts, order=order)
            fitted_model = model.fit()
            
            # Forecast
            forecast_result = fitted_model.get_forecast(steps=periods)
            forecast_values = forecast_result.predicted_mean
            confidence_intervals = forecast_result.conf_int()
            
            # Generate future dates
            last_date = pd.to_datetime(dates[-1])
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
            
            results = {
                'model': 'ARIMA',
                'order': order,
                'forecast_values': forecast_values.values,
                'forecast_dates': future_dates.strftime('%Y-%m-%d').tolist(),
                'confidence_intervals': confidence_intervals.values,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
            logger.info(f"ARIMA forecast completed: AIC={fitted_model.aic:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"ARIMA forecasting error: {str(e)}")
            return {'error': str(e)}

    def forecast_exponential_smoothing(self, timeline: Dict[str, int], 
                                      periods: int = 12) -> Dict:
        """
        Forecast using Exponential Smoothing
        
        Args:
            timeline: Time series dictionary
            periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            ts, dates = self.prepare_timeseries(timeline)
            
            if len(ts) < 4:
                logger.warning("Insufficient data for exponential smoothing")
                return {'error': 'Insufficient data'}
            
            logger.info("Fitting Exponential Smoothing model...")
            model = ExponentialSmoothing(ts, trend='add', seasonal=None)
            fitted_model = model.fit()
            
            # Forecast
            forecast_values = fitted_model.forecast(steps=periods)
            
            # Generate future dates
            last_date = pd.to_datetime(dates[-1])
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
            
            results = {
                'model': 'Exponential Smoothing',
                'forecast_values': forecast_values.values,
                'forecast_dates': future_dates.strftime('%Y-%m-%d').tolist(),
                'smoothing_level': fitted_model.params['smoothing_level'],
                'smoothing_trend': fitted_model.params['smoothing_trend']
            }
            
            logger.info("Exponential Smoothing forecast completed")
            return results
            
        except Exception as e:
            logger.error(f"Exponential Smoothing error: {str(e)}")
            return {'error': str(e)}

    def evaluate_forecast(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """
        Evaluate forecast accuracy
        
        Metrics:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - MAPE: Mean Absolute Percentage Error
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary with error metrics
        """
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        logger.info(f"Forecast evaluation: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        return metrics

    def compare_models(self, timeline: Dict[str, int], test_size: float = 0.2) -> Dict:
        """
        Compare multiple forecasting models
        
        Args:
            timeline: Time series dictionary
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with model comparisons
        """
        ts, dates = self.prepare_timeseries(timeline)
        split_point = int(len(ts) * (1 - test_size))
        
        train_ts = ts[:split_point]
        test_ts = ts[split_point:]
        
        comparisons = {}
        
        # ARIMA
        try:
            arima_model = ARIMA(train_ts, order=(1, 1, 1))
            arima_fit = arima_model.fit()
            arima_pred = arima_fit.forecast(steps=len(test_ts))
            arima_metrics = self.evaluate_forecast(test_ts.values, arima_pred.values)
            comparisons['ARIMA'] = arima_metrics
        except Exception as e:
            logger.warning(f"ARIMA comparison failed: {str(e)}")
        
        # Exponential Smoothing
        try:
            es_model = ExponentialSmoothing(train_ts, trend='add', seasonal=None)
            es_fit = es_model.fit()
            es_pred = es_fit.forecast(steps=len(test_ts))
            es_metrics = self.evaluate_forecast(test_ts.values, es_pred.values)
            comparisons['Exponential Smoothing'] = es_metrics
        except Exception as e:
            logger.warning(f"Exponential Smoothing comparison failed: {str(e)}")
        
        logger.info(f"Model comparison completed: {len(comparisons)} models evaluated")
        return comparisons

    def get_forecast_report(self, forecast_results: Dict) -> str:
        """
        Generate forecast report
        
        Args:
            forecast_results: Dictionary from forecast method
            
        Returns:
            Formatted forecast report
        """
        report = "=" * 60 + "\n"
        report += "FORECAST REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Model: {forecast_results.get('model', 'Unknown')}\n"
        report += f"Forecast Period: {len(forecast_results.get('forecast_values', []))} periods\n"
        
        if 'forecast_values' in forecast_results:
            values = forecast_results['forecast_values']
            report += f"\nForecast Summary:\n"
            report += f"  Mean: {np.mean(values):.2f}\n"
            report += f"  Min: {np.min(values):.2f}\n"
            report += f"  Max: {np.max(values):.2f}\n"
        
        report += "\n" + "=" * 60 + "\n"
        return report
