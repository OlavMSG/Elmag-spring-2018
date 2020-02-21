#Dette skriptet gjør en Sin regresjon, med usikkerhet
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
theta, ye = np.loadtxt('theta.txt', unpack=True,  delimiter=' ,')
 #x-verdier eksprimentell
theta = np.radians(theta)
 #y-verdier eksprimetell
sintheta = np.sin(theta)

#a0, a1, yb, Dy
a0, a1, yb, Dy = lineaer_regresjon(sintheta, ye)

#thetaFit og ybFit
mintheta = np.min(theta)
maxtheta = np.max(theta)
thetaFit = np.linspace(mintheta, maxtheta, 100)
ybFit = a0 + a1 * np.sin(thetaFit)


# Plotter
plt.figure(1)                # første figure
plt.plot(theta, ye, 'rx', label='(thetai,yi)', alpha=0.5)              # plottter theta mot ye i røde kryss
plt.plot(thetaFit, ybFit, label=' y = a0 + a1*sinthetaFit', alpha=0.5)               # plotter theta mot yb med blå linje
plt.ylim([0, 1.1*np.max(ye)])
plt.xlabel('$theta$')
plt.ylabel('$y$')
plt.title('RegSin')
plt.legend(loc='upper left');
#plt.savefig('theta.pdf')

plt.figure(2) # andre figur
plt.plot(sintheta, ye , 'rx', label='(sinthetai,yi)', alpha=0.5)  # plottter sintheta mot ye i røde kryss
plt.plot(sintheta, yb, label='y = a0 + a1*sintheta', alpha=0.5)  # plotter sintheta mot yb med blå linje
plt.ylim([0, 1.1*np.max(ye)])
plt.xlabel('$sintheta$')
plt.ylabel('$y$')
plt.title('sintheta mot y')
plt.legend(loc='upper left');
#plt.savefig('SinthetaRegLin.pdf')

plt.figure(3) # trede figur
plt.plot(sintheta, Dy, 'rx', label='(sinthetai,Dyi)', alpha=0.5)              # plottter sintheta mot Dy i røde kryss
plt.plot(sintheta, yb - yb)
plt.ylim([1.1*np.min(Dy), 1.1*np.max(Dy)])
plt.xlabel('$sintheta$')
plt.ylabel('$Dy$')
plt.title('sintheta mot Dy')
plt.legend(loc='upper left');
#plt.savefig('SinthetaRegLinDM.pdf')

plt.show()
