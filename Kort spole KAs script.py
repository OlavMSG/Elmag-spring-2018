
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

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

# Definerer variablene som inngår i uttrykkene for Biot-Savarts lov for alle geometriene
mu0, I, R, N, l, x, z = sp.symbols('mu0 I R N l x z')

kort_spole = (N*I*mu0)/(2*R)*(1 + (x/R)**2)**(-3/2)

dBbd = [sp.diff(kort_spole, x), sp.diff(kort_spole, I), sp.diff(kort_spole, R)]

xe, Be = np.loadtxt('Beregned_verdi_kort_spole.txt', unpack=True, delimiter = ' ,')
#posisjonen eksperimentielt#Målte B-verdier
xe = xe + 0
xb = np.linspace(xe[0],xe[-1], 200) #posisjon for en beregnet kurve


# Beregnede magnetfeltstyrker i Gauss (ganger med 10**4)
Bb = [kort_spole.subs([(N, 330), (I, 1), (mu0, 4*np.pi*1e-7), (R, 0.07), (x, xbi)])*1e4
      for xbi in xb]
Bb_forrev = [kort_spole.subs([(N, 330), (I, 1), (mu0, 4*np.pi*1e-7), (R, 0.07), (x, xei)])*1e4
      for xei in xe]

# Gjør om hvert element i Bb til numpy float-verdier
Bb = np.array(Bb).astype(np.float64)
Bb_forrev = np.array(Bb_forrev).astype(np.float64)

# Usikkerheter for beregnet kurve (tenk ut disse selv! Valg av disse må begrunnes i rapporten!!)
deltaxb = 0.001  # [m]
deltaIb = 0.01  # [A]
deltaRb = 0.001  # [m]

# Legger dem i en liste i samme rekkefølge (! VIKTIG!) som dBed
deltas = [deltaxb, deltaIb, deltaRb]

# Usikkerhet for gaussmeteret for eksperimentelle måledata (som oppgitt i kap. 3.3 i labheftet)
deltaBe = np.around(Be*0.004 + 0.001*100 + 0.01, decimals=2)  # [Gauss]

# Definerer et array av samme form som Be
deltaBb = np.zeros(xb.shape)

# Beregner usikkerhetene for hver av måleposisjonene. Én iterasjon, én deltaB(x_i).
for i in range(len(xb)):
    
    # Benytter list comprehension her, se f.eks.
    # http://www.secnetix.de/olli/Python/list_comprehensions.hawk
    # Eksempel syntaks: variabel = [en operasjon for i in range(3)]
    # Hva vi gjør (fra innerst til ytterst i linjen):
    #   1. Med dBkd[j].subs() bytter vi ut variablene N, I, mu0, R og x med tallverdier. x endrer seg
    #      for hver i-iterasjon i den ytre løkken.
    #   2. Kvadrerer
    #   3. Summerer de tre leddene ()^2
    deltaBb[i] = np.sum([(dBbd[j].subs([(N, 330), (I, 1), (mu0, 4*np.pi*1e-7), (R, 0.07),
                                        (x, xb[i])])*deltas[j])**2 for j in range(len(deltas))])
    # Så tar vi roten og gjør om fra Tesla til Gauss
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
plt.title('Kort spole')
plt.ylim([0, 1.1*np.max(Bb + deltaBb)])
plt.xlabel('$x$ [m]')
plt.ylabel('$B$ [Gauss]')
plt.legend(loc='upper left');
plt.savefig('Kort_spole_v3.pdf')

plt.figure(2)
plt.fill_between(xb, - 1*deltaBb_rev, 1*deltaBb_rev,
                    label='Relativ usikkerhet i beregnet ', alpha=0.5)
#plt.errorbar(xe, DB_rev_e, yerr=deltaBe_rev, fmt='r.', label='Avvik i prosent')
plt.plot(xe, DB_rev_e, 'rx', label='Avvik i prosent', alpha=0.5)
plt.plot(xb, Bb-Bb, label='Avvik=0')
plt.title('Kort spole relativ usikkerhet i prosent')
plt.ylim([-1.1*np.max(deltaBb_rev), 1.5*np.max(DB_rev_e)])
plt.xlabel('$x$ [m]')
plt.ylabel('Avvik [%]')
plt.legend(loc='upper center');
plt.savefig('Kort_spole_v3_relativusikkerhet.pdf')
plt.show()

plt.close()