#Dette skriptet gjør en Linær regresjon, med usikkerhet
#skrevet av Olav Milian Schmitt Gran

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from regresjon import lineaer_regresjon

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
#xe, ye = np.loadtxt('test.txt', unpack = True,  delimiter =' ')
xe = np.loadtxt('testiter.txt', unpack = True,  delimiter =' ')
ye = np.loadtxt('testvar.txt', unpack = True,  delimiter =' ')
 #x-verdier eksprimentell
 #y-verdier eksprimetell
k = 0
n = 0
xv = []
yv = []

for i in range(len(xe)):
    if xe[i] == 30:
        k += 1
        yv.append(ye[i])
    if ye[i] != 0:
        n += 1
        xv.append(xe[i])

print(k, n)
print(xv, yv)


#a0, a1, yb, Dy
a0, a1, yb, Dy = lineaer_regresjon(xe, ye)


# Plotter
plt.figure(1)                # første figure
plt.plot(xe, ye, 'rx', label='(xi,yi)', alpha=0.5)              # plottter xe mot ye i røde kryss
plt.plot(xe, yb, label='y = a0 +a1*x', alpha=0.5)               # plotter xe mot yb med blå linje
plt.ylim([-100, 1.5*np.max(ye)])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('RegLin')
plt.legend(loc='upper left');
#plt.savefig('RegLin.pdf')

plt.figure(2) # andre figur
plt.plot(xe, Dy, 'rx', label='(xi,Dyi)', alpha=0.5)              # plottter xe mot Dy i røde kryss
plt.plot(xe, yb - yb)
plt.ylim([1.1*np.min(Dy), 1.1*np.max(Dy)])
plt.xlabel('$x$')
plt.ylabel('$Dy$')
plt.title('x mot Dy')
plt.legend(loc='upper left');
#plt.savefig('RegLinDy')

plt.show()

