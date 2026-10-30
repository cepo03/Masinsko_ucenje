import numpy as np
import pandas as pd
from polinomijalna_regresija_reg import PolinomijalnaRegresijaRegularizovana

podaci = {
    'X': [520, 610, 700, 820, 910, 1030, 1120, 1250, 1390, 1510],
    'Y': [3.1, 3.8, 4.0, 5.1, 5.3, 6.0, 6.2, 5.9, 5.1, 4.1]
}
df = pd.DataFrame(podaci)
X = df['X'].values
y = df['Y'].values

nove_tacke_X = np.array([600, 900, 1300, 1600])

print("=" * 60)
print("POREDJENJE MODELA M = 9 (BEZ VS SA L2 REGULARIZACIJOM)")
print("=" * 60)

model_bez_reg = PolinomijalnaRegresijaRegularizovana(stepen=9, lambda_reg=0.0)
model_bez_reg.fit(X, y)
pred_bez = model_bez_reg.predict(nove_tacke_X)

model_sa_reg = PolinomijalnaRegresijaRegularizovana(stepen=9, lambda_reg=1.5)
model_sa_reg.fit(X, y)
pred_sa = model_sa_reg.predict(nove_tacke_X)

print("\n--- Model BEZ regularizacije (lambda = 0.0) ---")
print("Koeficijenti w:", np.round(model_bez_reg.koeficijenti_, 2))
print("Predikcije na novim tackama:", np.round(pred_bez, 3))
print("MSE na obuci:", model_bez_reg.score(X, y))

print("\n--- Model SA regularizacijom (lambda = 1.5) ---")
print("Koeficijenti w:", np.round(model_sa_reg.koeficijenti_, 2))
print("Predikcije na novim tackama:", np.round(pred_sa, 3))
print("MSE na obuci:", model_sa_reg.score(X, y))