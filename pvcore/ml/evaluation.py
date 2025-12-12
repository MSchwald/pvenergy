import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

from typing import Callable
from dataclasses import dataclass

@dataclass
class EvaluationMethod:
    """Template for creating evaluation methods for trained ML models"""
    name: str
    method: Callable[..., pd.Series]
    _result: pd.Series = None

    def evaluate(self, model, X_test, y_test, y_pred):
        result = self.method(model, X_test, y_test, y_pred)
        if isinstance(result, pd.Series):
            self._result = result
        else:
            self._result = pd.Series([result], index = [self.name])
        return self._result
    def print_result(self):
        if self._result is None:
            print(f"Method {self.name} has not been evaluated on test data yet.")
            return 
        print(self.name, self._result)
        
class EVALUATIONS:
    """Methods to analyze the performance of trained ML models"""
    def rmse_method(model, X_test, y_test, y_pred) -> pd.Series:
        return np.sqrt(mean_squared_error(y_test, y_pred))
                       
    def r2_method(model, X_test, y_test, y_pred) -> pd.Series:
        return r2_score(y_test, y_pred)
        
    def feature_importance_method(model, X_test, y_test, y_pred) -> pd.Series:
        df = pd.DataFrame({
            'Feature': X_test.columns.tolist(),
            'Importance': model.feature_importances_
        }).sort_values(by = 'Importance', ascending = False)
        return pd.Series(df['Importance'].values, index = df['Feature']) 

    RMSE = EvaluationMethod(
        name = "rmse",
        method = rmse_method     
    )
    R2 = EvaluationMethod(
        name = "r2",
        method = r2_method
    )
    FEATURE_IMPORTANCE = EvaluationMethod(
        name = "feature_importance",
        method = feature_importance_method
    )
    
ALL_EVALUATIONS = tuple(eval for eval in vars(EVALUATIONS).values() if isinstance(eval, EvaluationMethod)) 