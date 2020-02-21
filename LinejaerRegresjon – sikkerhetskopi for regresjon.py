#Dette skriptet gjør en Linær regresjon, med usikkerhet
#skrevet av Olav Milian Schmitt Gran

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.backends.backend_pdf import PdfPages

# Initialiserer pen visning av uttrykkene
sp.init_printing()

# Plotteparametre for C% fC% store, tydelige plott som utnytter tilgjengelig skjermareal
fontsize = 20
newparams = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
             'lines.linewidth': 2, 'lines.markersize': 7,
             'figure.figsize': (16, 7), 'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 'legend.fontsize': fontsize,
            'legend.handlelength': 1.5}
plt.rcParams.update(newparams)

#Les inn fil
data = np.loadtxt('data.dat', delimiter =' ,')
xe = data [ : ,0] #x-verdier eksprimentell
ye = data [ : ,1]  #y-verdier eksprimetell

#Beregning av a0 og a1
N = len(xe) #finner N ved lengden av listen for x
#hjelpestørelser
Sx, Sy, Sxx, Sxy = sum(xe), sum(ye), sum(xe**2), sum(xe*ye)
Delta = N * Sxx - Sx**2
#Beregning
a0 = (Sy*Sxx - Sx*Sxy) / Delta
a1 = (N*Sxy-Sx*Sy) / Delta
print('a1 = ', round(a1, 5), 'a0 = ', round(a0, 3))

#Beregnede y-verdier for Regresjon
yb = a0 + a1*xe

#Beregning av Da1 og Da0
    #Hjelpestørrelser
Dy = ye -yb
S = sum(Dy**2)
Da0 = np.sqrt(1 / (N-2) * (S * Sxx) / Delta)
Da1 = np.sqrt(N / (N-2) * S / Delta)
print('Da1 = ', round(Da1, 6), 'a0 = ', round(Da0, 5))

# Plotter
    #with PdfPages('Kort spole.pdf') as pdf:
plt.figure(1)                # første figure
plt.plot(xe, ye, 'rx', label='(xi,yi)', alpha=0.5)              # plottter xe mot ye i røde kryss
plt.plot(xe, yb, label='y = a0 +a1*x', alpha=0.5)               # plotter xe mot yb med blå linje
plt.ylim([0, 1.1*np.max(ye)])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('RegLin')
plt.legend(loc='upper left');

plt.figure(2) # andre figur
plt.plot(xe, Dy, 'rx', label='(xi,Dyi)', alpha=0.5)              # plottter xe mot Dy i røde kryss
plt.ylim([0, 1.1*np.max(Dy)])
plt.xlabel('$x$')
plt.ylabel('$Dy$')
plt.title('x mot Dy')
plt.legend(loc='upper left');
plt.show()
#pdf.savefig()  # lagrer firgur til pdf
#plt.close()