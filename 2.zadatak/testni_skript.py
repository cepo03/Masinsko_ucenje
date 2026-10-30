import pandas as pd
import numpy as np
from polinomijalna_regresija import PolinomijalnaRegresija


podaci = {
    'X': [520, 610, 700, 820, 910, 1030, 1120, 1250, 1390, 1510],
    'Y': [3.1, 3.8, 4.0, 5.1, 5.3, 6.0, 6.2, 5.9, 5.1, 4.1]
}
df = pd.DataFrame(podaci)


X = df['X'].values
y = df['Y'].values


M = 2
model = PolinomijalnaRegresija(stepen=M)


model.fit(X, y)
print(f"Procenjeni koeficijenti w za M = {M}:")
print(model.koeficijenti_)


y_pred = model.predict(X)
print("\nPredikcije biomase (Y):")
print(y_pred)

mse_vrednost = model.score(X, y)
print(f"\nSrednja kvadratna greska (MSE) za M = {M}: {mse_vrednost:.4f}")