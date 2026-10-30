import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class PolinomijalnaRegresijaRegularizovana(BaseEstimator, RegressorMixin):
    def __init__(self, stepen=1, lambda_reg=0.0):
      
        self.stepen = stepen
        self.lambda_reg = lambda_reg
        self.koeficijenti_ = None

    def _konstruisi_matricu_dizajna(self, X):
        X = np.asarray(X).ravel()
        N = len(X)
        M = self.stepen
        
        Phi = np.ones((N, M + 1))
        for j in range(1, M + 1):
            Phi[:, j] = X ** j
            
        return Phi

    def fit(self, X, y):
       
        X = np.asarray(X)
        y = np.asarray(y)
        
        Phi = self._konstruisi_matricu_dizajna(X)
        N, broj_parametara = Phi.shape
        
        Phi_T = Phi.T
        A = np.dot(Phi_T, Phi)
        b = np.dot(Phi_T, y)
        
        I = np.eye(broj_parametara)
        I[0, 0] = 0 
        
        A_reg = A + self.lambda_reg * I
        
        self.koeficijenti_ = np.linalg.solve(A_reg, b)
        
        return self

    def predict(self, X):
        if self.koeficijenti_ is None:
            raise ValueError("Model još uvek nije obučen!")
        Phi = self._konstruisi_matricu_dizajna(X)
        return np.dot(Phi, self.koeficijenti_)

    def score(self, X, y):
        y = np.asarray(y)
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return mse