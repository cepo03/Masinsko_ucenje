import numpy as np
import pandas as pd
from polinomijalna_regresija import PolinomijalnaRegresija


podaci = {
    'X': [520, 610, 700, 820, 910, 1030, 1120, 1250, 1390, 1510],
    'Y': [3.1, 3.8, 4.0, 5.1, 5.3, 6.0, 6.2, 5.9, 5.1, 4.1]
}
df = pd.DataFrame(podaci)
X = df['X'].values
y = df['Y'].values

nove_tacke_X = np.array([600, 900, 1300, 1600])

stepeni_M = [1, 2, 3, 5, 9]

print("=" * 70)
print("REZULTATI ZADATKA 3")
print("=" * 70)

rezultati_novih_tacaka = {}

for M in stepeni_M:
    model = PolinomijalnaRegresija(stepen=M)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mse = model.score(X, y) 
    E_w = mse / 2.0         
    
    predikcije_nove = model.predict(nove_tacke_X)
    rezultati_novih_tacaka[f'M={M}'] = predikcije_nove
    
    print(f"\n--- Model stepena M = {M} ---")
    print(f"Vektor parametara w: {np.round(model.koeficijenti_, 4)}")
    print(f"Funkcija greske E(w): {E_w:.4f}")
    print(f"MSE na obucavajucem skupu: {mse:.4f}")

df_nove = pd.DataFrame(rezultati_novih_tacaka, index=[f'X={x}mm' for x in nove_tacke_X])
print("\n" + "=" * 70)
print("PREDIKCIJE BILJNE BIOMASE (Y) ZA NOVE TACKE PADAVINA (X):")
print("=" * 70)
print(np.round(df_nove, 3))



"""
DISKUSIJA REZULTATA I ANALIZA MODELA:

1. Kako se menja oblik polinoma sa porastom stepena M:
   - Za M = 1 (linearni model): Oblik je krut i linearan, ne moze da prati zakrivljenost.
   - Za M = 2 i M = 3: Kriva prirodno prati opsti trend podataka (rast biomase 
     do srednjih vrednosti padavina, a zatim blagi pad).
   - Za M = 9: Kriva postaje izuzetno "talasasta" i ekstremno se prilagodjava 
     svakoj pojedinacnoj tacki merenja.

2. Kako se menja greska na skupu za obuku:
   - Povecanjem stepena M, greska na trening skupu (MSE) monotono opada.
   - Za M = 9 (gde je broj parametara jednak broju tacaka N=10), matrica postaje 
     kvadratna, polinom prolazi tacno kroz svaku tacku, pa je MSE blizu nule.

3. Kako se menjaju predikcije na novim tackama i problemi prilagodjavanja:
   - Nedovoljna prilagodjenost (Underfitting): Javlja se kod M = 1. Model nema 
     dovoljan kapacitet da opise podatke, pa je greska velika i na obuci i na testu.
   - Preterana prilagodjenost (Overfitting): Javlja se kod M = 9. Model je "zapamtio" 
     sum iz trening podataka. Izmedju tacaka dolazi do Rungeovog fenomena 
     (ogromnih oscilacija), dok na novim tackama (npr. X = 1600 mm) daje 
     nerealne i ekstremne vrednosti biomase.
   
4. Zakljucak:
   - Najbolji balans izmedju fleksibilnosti i generalizacije na nepoznatim 
     podacima daju modeli umerenog stepena (M = 2 ili M = 3).
"""