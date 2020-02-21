import numpy as np
from matplotlib import pyplot as plt
import sympy as sp
from matplotlib.backends.backend_pdf import PdfPages

# Initialiserer pen visning av uttrykkene
sp.init_printing()

# Plotteparametre for å få store, tydelige plott som utnytter tilgjengelig skjermareal
fontsize = 20
newparams = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
             'lines.linewidth': 2, 'lines.markersize': 7,
             'figure.figsize': (16, 7), 'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 'legend.fontsize': fontsize,
            'legend.handlelength': 1.5}
plt.rcParams.update(newparams)

mu0, I, R, N, l, x, z, a = sp.symbols('mu0 I R N l x z a')

Bs = mu0*I*N/(2*l)*(z/(sp.sqrt(z**2+R**2))+(l-z)/(sp.sqrt(((l-z)**2)+R**2)))
dBsbd = [sp.diff(Bs, z), sp.diff(Bs, I), sp.diff(Bs, R), sp.diff(Bs, l)]

ze, Be = np.loadtxt('Målte verdier Solenoide.txt', unpack=True, delimiter = ' ,')
#posisjonen eksperimentielt#Målte B-verdier
ze = ze + 0.15e-2
zb = np.linspace(ze[0], ze[-1], 200) #posisjon for en beregnet kurve


Bb = [Bs.subs([(N, 368), (I, 1), (mu0, 4*np.pi*1e-7), (R, 0.05), (z, zbi), (l, 0.392)])*1e4 for zbi in zb]
Bb = np.array(Bb).astype(np.float64)
Bb_forrev = [Bs.subs([(N, 368), (I, 1), (mu0, 4*np.pi*1e-7), (R, 0.05), (z, zei), (l, 0.392)])*1e4 for zei in ze]
Bb_forrev = np.array(Bb_forrev).astype(np.float64)

deltaz = 0.001
deltaI = 0.01
deltaR = 0.001
deltaL = 0.001

deltas = [deltaz, deltaI, deltaR, deltaL]
deltaBe = np.around(Be*0.004 + 0.001*100 + 0.01, decimals=2)
deltaBb = np.zeros(zb.shape)

for i in range(len(zb)):
    deltaBb[i] = np.sum([(dBsbd[j].subs([(N, 368), (I, 1), (mu0, 4*np.pi*1e-7), (R, 0.05),
                                        (z, zb[i]), (l, 0.392)])*deltas[j])**2 for j in range(len(deltas))])
    deltaBb[i] = np.sqrt(deltaBb[i])*1e4

#relativ usikerhet i prosent
DB_rev_e = (Be-Bb_forrev)/Bb_forrev*100
deltaBe_rev = deltaBe / Be *100
deltaBb_rev = deltaBb / Bb *100

# Plotter
plt.figure(1)
plt.fill_between(zb, Bb - 1*deltaBb, Bb + 1*deltaBb,
                     label='Beregnet kurve\n med usikkerhet', alpha=0.5)
plt.errorbar(ze, Be, yerr=deltaBe, fmt='r.', label='Måledata')
plt.title('Solenoide')
plt.ylim([0, 1.1*np.max(Bb + deltaBb)])
plt.xlim([1.1*np.min(ze), 1.1*np.max(ze)])
plt.xlabel('$z$ [m]')
plt.ylabel('$B$ [Gauss]')
plt.legend(loc='upper right');
plt.savefig('Solenoide_v4.pdf')

plt.figure(2)
plt.fill_between(zb, - 1*deltaBb_rev, 1*deltaBb_rev,
                    label='Relativ usikkerhet i beregnet ', alpha=0.5)
#plt.errorbar(ze, DB_rev_e, yerr=deltaBe_rev, fmt='r.', label='Avvik i prosent')
plt.plot(ze, DB_rev_e, 'rx', label='Avvik i prosent', alpha=0.5)
plt.plot(zb, Bb-Bb, label='Avvik=0')
plt.title('Kort spole relativ usikkerhet i prosent')
plt.ylim([-1.1*np.max(deltaBb_rev), 3.1*np.max(DB_rev_e)])
plt.xlabel('$z$ [m]')
plt.ylabel('Avvik [%]')
plt.legend(loc='upper center');
plt.savefig('Solenoide_v4_relativusikkerhet.pdf')
plt.show()

plt.close()
