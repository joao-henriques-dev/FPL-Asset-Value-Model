from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy.typing import NDArray

class CleanSheetPointsModel:
    """
    A predictive model for determining FPL clean sheet points per game for goalkeepers,
    defenders and midfielders based on expected goals conceded per 90 minutes (xGCp90) data.

    The module provides functionality to train a linear regression model
    and make predictions given a xGCp90 value using a Stochastic Gradient
    Descent Regressor (SGDRegressor) with feature scaling (StandardScaler).

    The model is trained using a training dataset consisting of 20 xGCp90
    data points and their corresponding clean sheet points per game.
    The data was obtained by FBRef's 2024-2025 season of the English Premier League,
    with extensions being planned to achieve a more representative sample for
    improved prediction accuracy to strengthen the model's credibility.

    Attributes:
        x_data (np.ndarray): Input features (xGCp90 values)
        y_data (np.ndarray): Target values (clean sheet points per match)
        scaler (StandardScaler): Scaler object for normalising data for more accurate predictions
        sgdr (SGDRegressor): The regression model object to train and make predictions with
        is_trained (bool): Indicates if the model is trained or not
    """

    def __init__(self, x_data: NDArray[np.float64], y_data: NDArray[np.float64]):
        """
        Initialises the Clean Sheet Points Model object with the
        data to achieve a realistic plot for accurately predicting
        the clean sheet points per game based on xGCp90 data.

        Args:
          x_data (np.ndarray (m,)): Training data with m examples with 1 feature each
            (xGCp90 values for each PL team)
          y_data (np.ndarray (m,)): Target values (clean sheet points per game for each PL team)
        """
        self.x_data = x_data.reshape(-1, 1)
        self.y_data = y_data
        self.scaler = StandardScaler()
        self.sgdr = SGDRegressor(max_iter=1000, random_state=42)
        self.is_trained = False

    def train_model(self) -> None:
        """
        Trains the model with the training data by using a Stochastic Gradient
        Descent Regressor (SGDRegressor) with feature scaling (StandardScaler).
        """
        x_data_norm = self.scaler.fit_transform(self.x_data)
        self.sgdr.fit(x_data_norm, self.y_data)
        self.is_trained = True

    def predict(self, xgcp90: float) -> float:
        """
        Checks if the model is trained before making a prediction.
        If the model is already trained, a prediction is made using
        the trained model with feature scaling applied to the given input value.
        If the model is not trained yet, it will call the 'CleanSheetPointsModel.train_model()'
        method to train the model first before making a prediction.

        Args:
            xgcp90 (float): Input feature value to be used for prediction

        Returns:
            y_pred (float): Predicted value for the input feature
            None: If the model is not trained
        """
        if not self.is_trained:
            CleanSheetPointsModel.train_model(self)

        xgcp90_norm = self.scaler.transform([[xgcp90]])
        y_pred = self.sgdr.predict(xgcp90_norm)
        return float(y_pred[0])