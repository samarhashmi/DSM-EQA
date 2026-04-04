import numpy as np
from scipy import stats
from typing import Dict, Tuple

class StatisticalAnalysis:
    """Statistical analysis utilities."""
    
    @staticmethod
    def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute residuals."""
        return y_true - y_pred
    
    @staticmethod
    def normality_test(residuals: np.ndarray) -> Dict[str, float]:
        """
        Shapiro-Wilk normality test.
        
        Args:
            residuals: Residuals
            
        Returns:
            Test statistic and p-value
        """
        stat, p_value = stats.shapiro(residuals)
        return {'statistic': stat, 'p_value': p_value}
    
    @staticmethod
    def heteroscedasticity_test(y_pred: np.ndarray, residuals: np.ndarray) -> Dict[str, float]:
        """
        Breusch-Pagan heteroscedasticity test.
        
        Args:
            y_pred: Predictions
            residuals: Residuals
            
        Returns:
            Test statistic and p-value
        """
        # Simplified BP test
        n = len(residuals)
        residuals_sq = residuals ** 2
        X = np.column_stack([np.ones(n), y_pred])
        
        # Linear regression on squared residuals
        coeffs = np.linalg.lstsq(X, residuals_sq, rcond=None)[0]
        fitted = X @ coeffs
        ss_res = np.sum((residuals_sq - fitted) ** 2)
        ss_tot = np.sum((residuals_sq - np.mean(residuals_sq)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        test_stat = n * r2
        p_value = 1 - stats.chi2.cdf(test_stat, df=1)
        
        return {'statistic': test_stat, 'p_value': p_value}
    
    @staticmethod
    def confidence_interval(predictions: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence intervals.
        
        Args:
            predictions: Model predictions
            confidence: Confidence level
            
        Returns:
            Tuple of (lower, upper) bounds
        """
        std_error = np.std(predictions) / np.sqrt(len(predictions))
        margin = 1.96 * std_error if confidence == 0.95 else 2.576 * std_error
        
        return predictions - margin, predictions + margin
