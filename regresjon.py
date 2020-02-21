#Dette skriptet gjør en Linær regresjon, med usikkerhet
#skrevet av Olav Milian Schmitt Gran

import numpy as np

def lineaer_regresjon(x, y):
    # Beregning av a0 og a1
    N = len(x)  # finner N ved lengden av listen for x
    # hjelpestørelser
    Sx, Sy, Sxx, Sxy = sum(x), sum(y), sum(x ** 2), sum(x * y)
    Delta = N * Sxx - Sx ** 2
    # Beregning
    a0 = (Sy * Sxx - Sx * Sxy) / Delta
    a1 = (N * Sxy - Sx * Sy) / Delta
    print('a1 = ', a1, 'a0 = ', a0)

    # Beregnede y-verdier for Regresjon
    yb = a0 + a1 * x

    # Beregning av Da1 og Da0
    # Hjelpestørrelser
    Dy = y - yb
    S = sum(Dy ** 2)
    Da0 = np.sqrt(1 / (N - 2) * (S * Sxx) / Delta)
    Da1 = np.sqrt(N / (N - 2) * S / Delta)
    print('Da1 = ', Da1, 'Da0 = ', Da0)

    #returnerer a0 og a1
    return a0, a1, yb, Dy