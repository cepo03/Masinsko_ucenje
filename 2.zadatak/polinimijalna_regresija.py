import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class PolinomijalnaRegresija(BaseEstimator, RegressorMixin):
    def __init__(self, stepen=1):
        
        self.stepen = stepen
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
        
        Phi_T = Phi.T
        A = np.dot(Phi_T, Phi)
        b = np.dot(Phi_T, y)
        
     
        self.koeficijenti_ = np.linalg.solve(A, b)
        
        return self

    def predict(self, X):
       
        if self.koeficijenti_ is None:
            raise ValueError("Model jos uvek nije obucen! Pozovite metodu 'fit' pre predikcije.")
            
        Phi = self._konstruisi_matricu_dizajna(X)
        
        y_pred = np.dot(Phi, self.koeficijenti_)
        return y_pred

    def score(self, X, y):
        
        y = np.asarray(y)
        y_pred = self.predict(X)
        
        
        mse = np.mean((y - y_pred) ** 2)
        return mse