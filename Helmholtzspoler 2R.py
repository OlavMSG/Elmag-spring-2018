import numpy as np
from matplotlib import pyplot as plt
import sympy as sp

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

mu0, I, R, N, l, x, z, a = sp.symbols('mu0 I R N l x z a')

#Generell formel som kan brukes for a=2R, a=R og a=R/2
Bhhs = N*mu0*I/(2*R)*((1+((x-a/2)/R)**2)**(-3/2)+(1+((x+a/2)/R)**2)**(-3/2))
dBbd = [sp.diff(Bhhs, x), sp.diff(Bhhs, I), sp.diff(Bhhs, R), sp.diff(Bhhs, a)]

xe, Be = np.loadtxt('Målte verdier Helmholtzspoler a=2R.txt', unpack=True, delimiter = ' ,')
#posisjonen eksperimentielt#Målte B-verdier
xe = xe +0.1e-2
xb = np.linspace(xe[0], xe[-1], 200) #posisjon for en beregnet kurve
a_R = 0.07 * 2

Bb = [Bhhs.subs([(N, 330), (I, 1), (mu0, 4*np.pi*1e-7), (R, 0.07), (x, xbi), (a, a_R)])*1e4 for xbi in xb]
Bb_forrev = [Bhhs.subs([(N, 330), (I, 1), (mu0, 4*np.pi*1e-7), (R, 0.07), (x, xei), (a, a_R)])*1e4 for xei in xe]

Bb = np.array(Bb).astype(np.float64)
Bb_forrev = np.array(Bb_forrev).astype(np.float64)

deltaxb = 0.001  # [m]
deltaIb = 0.01  # [A]
deltaRb = 0.001  # [m]
deltaa = 0.005

deltas = [deltaxb, deltaIb, deltaRb, deltaa]
deltaBe = np.around(Be*0.004 + 0.001*100 + 0.01, decimals=2)
deltaBb = np.zeros(xb.shape)

for i in range(len(xb)):
    deltaBb[i] = np.sum([(dBbd[j].subs([(N, 330), (I, 1), (mu0, 4*np.pi*1e-7), (R, 0.07),
                                        (x, xb[i]), (a, a_R)])*deltas[j])**2 for j in range(len(deltas))])
    deltaBb[i] = np.sqrt(deltaBb[i])*1e4

#relativ usikerhet i prosent
DB_rev_e = (Be-Bb_forrev)/Bb_forrev*100
deltaBe_rev = deltaBe / Be *100
deltaBb_rev = deltaBb / Bb *100

# Plotter
plt.figure(1)
plt.fill_between(xb, Bb - 1*deltaBb, Bb + 1*deltaBb,
                     label='Beregnet kurve \n med usikkerhet', alpha=0.5)
plt.errorbar(xe, Be, yerr=deltaBe, fmt='r.', label='Måledata')
plt.title('Helmholtzspole a=2R')
plt.ylim([0, 1.1*np.max(Bb + deltaBb)])
plt.xlabel('$x$ [m]')
plt.ylabel('$B$ [Gauss]')
plt.legend(loc='upper left');
plt.savefig('Helmholtzspole_a=2R_v3.pdf')

plt.figure(2)
plt.fill_between(xb, - 1*deltaBb_rev, 1*deltaBb_rev,
                    label='Relativ usikkerhet i beregnet ', alpha=0.5)
#plt.errorbar(xe, DB_rev_e, yerr=deltaBe_rev, fmt='r.', label='Avvik i prosent')
plt.plot(xe, DB_rev_e, 'rx', label='Avvik i prosent', alpha=0.5)
plt.plot(xb, Bb-Bb, label='Avvik=0')
plt.title('Helmholtzspole_a=2R relativ usikkerhet i prosent')
plt.ylim([-1.1*np.max(deltaBb_rev), 4.4*np.max(DB_rev_e)])
plt.xlabel('$x$ [m]')
plt.ylabel('Avvik [%]')
plt.legend(loc='upper center');
plt.savefig('Helmholtzspole_a=2R_v3_relativusikkerhet.pdf')

plt.show()
plt.close()


