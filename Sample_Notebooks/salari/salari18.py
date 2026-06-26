import numpy as np
import matplotlib.pyplot as plt
from scipy import *
import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker


# Grace Brannigan 4/2018 rev 9/2018
# Accompanies "A streamlined, general approach for computing ligand binding free energies and its application to GPCR-bound cholesterol" by R. Salari, T. Joseph, R. Lohia, J. Henin, G. Brannigan
# Script used to calculate quantities in Table 5 and 6 and predictions in Figure 3; also plots chemical potential and derivative with respect to cholesterol concentration

def mix_model(x, P0, h0):
    # quadratic mixture model
    # P0: ideal bulk/gas partition coefficient;
    # h0: enthalpy of mixing
    return P0 * np.exp(-(1 - x) ** 2 * h0 / kT)


def inv_mix_model(x, invP0, h0):
    # quadratic mixture model;
    # invP0: ideal gas/bulk partition coefficient;
    # h0: enthalpy of mixing
    return invP0 * np.exp((1 - x) ** 2 * h0 / kT)


def get_x50(x, y):
    # determine x value for whsich y = 1/2
    x50 = x[(np.abs(y - 0.5)).argmin()]
    if ((np.max(y) < 0.5) | (np.min(y) > 0.5)):
        print("Warning: Predicted occupancy does not cross 0.5, will print x for which occupancy is closest to 0.5:")
    return x50


################### Plotting PARAMETERS
logxscale = 1  # for shared logarithmic x-axis, set to 0 for linear
xdecades, xmax, nx = 12., 1.0, 200
x = (10 ** (np.linspace(-xdecades, 0, nx)) * xmax).reshape(nx, 1)  # desired concentration range for prediction
np.set_printoptions(precision=5)

################### PARAMETERS used in MD Simulation
kT = 0.59  # kBT in kcal/mol
totLipid = 230.  # desired total number of lipids per receptor
alpha = 0.3  # fraction of total number in bulk restraint volume
sim_x = np.array(
    [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])  # cholesterol fractions used during decoupling from membrane
thetaR = 0.14 * np.pi  # maximum angle for bulk orientation restraint in radians
zR = 11.  # height of bulk restraint in Angstrom
A = 3600.  # typical box area for bulk simulation in Angstrom^2
rR = 5.  # radius of coarse restraint in Angstrom

################### RESULTS from MD Simulation

# AFEP Delta G values for DECOUPLING from membrane in kcal/mol, averaged across 3 replicas, for cholesterol fractions in x
AFEP1 = np.array([18.1, 18.2, 18.5, 18.5, 18.8, 18.6, 18.9, 19.1, 18.9])
# deltaG2 standard error in kcal/mol
AFEP1_err = np.array([0.3, 0.3, 0.3, 0.4, 0.7, 0.2, 0.3, 0.5, 0.4]) / sqrt(2)
# RFEP Delta G values in kcal/mol for imposing DBC restraints on decoupled cholesterol, was equivalent within precision for all 3 bound configurations
RFEP = 4.6
# AFEP Delta G values for DECOUPLING from protein for 3 GPCRs in kcal/mol, 3 replicas each
AFEP_3D4S = np.array([34.4, 33.9, 34.5])
AFEP_4CN3 = np.array([25.6, 25.2, 27.0])
AFEP_5C1M = np.array([23.1, 25.7, 25.7])
# averaging across replicas
AFEP2 = np.array([np.average(AFEP_3D4S), np.average(AFEP_4CN3), np.average(AFEP_5C1M)])

###################Prediction: Bulk/Gas Partition coefficient & non-ideality of bulk##############

Px_sim = exp(AFEP1 / kT)  # Simulated values of the bulk/gas partition coefficient
inv_Px_sim = exp(-AFEP1 / kT)  # Simulated values of the gas/bulk partition coefficient
inv_Px_err = inv_Px_sim * (AFEP1_err / kT)  # Propagated Error
popt, pcov = curve_fit(inv_mix_model, sim_x,
                       inv_Px_sim)  # Fit values to the quadratic mixture model; numerically fitting the gas/bulk coefficient seems more robust than the bulk/gas coefficient
inv_P0, h0 = popt[0], popt[1]
P0 = 1 / inv_P0

print
"Fitted Ideal Gas/Bulk Partition Coefficient (PO): %.2g" % P0
print
"Fitted Enthalpy of Mixing in kcal/mol (h0): %.2g\n" % h0

inv_Px = inv_mix_model(x, inv_P0, h0)  # prediction based on curve fit parameters

###plot Gas-Bulk Partition Coefficient (Figure 3A, Row 2 of Table 5)
plt.subplot(5, 1, 1)
plt.ylabel(r'$1/P_x$')
plt.errorbar(sim_x, inv_Px_sim, yerr=inv_Px_err)
if logxscale:
    plt.semilogx(x, inv_Px, 'k')
else:
    plt.plot(x, inv_Px, 'k')
plt.gca().axes.get_xaxis().set_visible(False)

###Plot Chemical Potential
mu = kT * np.log(x) + h0 * (1 - x) ** 2
plt.subplot(5, 1, 2)
plt.ylabel(r'$\mu - \mu^0$' + '\n(kcal/mol)', multialignment='center')
if logxscale:
    plt.semilogx(x, mu)
else:
    plt.plot(x, mu)
plt.gca().axes.get_xaxis().set_visible(False)
tick_spacing = 3
plt.gca().axes.get_yaxis().set_major_locator(ticker.MultipleLocator(tick_spacing))

# Plot x Derivative of Chemical Potential
dmu = kT / x - 2 * h0 * (1 - x)
plt.subplot(5, 1, 3)
plt.ylabel(r'$d\mu/dx$' + '\n(kcal/mol)', multialignment='center')
if logxscale:
    plt.semilogx(x, dmu)
else:
    plt.plot(x, dmu)
plt.gca().axes.get_xaxis().set_visible(False)

#########################################Prediction: OCCUPATION PROBABILITIES######################
# Calculate quantities in Table 5 and Table 6
ratio1 = x * totLipid * alpha  # Table 5, Row 1
ratio2 = inv_Px  # Table 5, Row 2
ratio3 = (2. / 3. * np.pi) * (rR ** 3. / zR / A) / (1 - np.cos(thetaR))  # Table 5,Row 3
ratio4 = np.exp(-RFEP / kT)  # Table 6, Row 1
ratio5 = np.exp(AFEP2 / kT).reshape(1, 3)  # Table 6,Row 2

##Plot kappa
kappa = ratio1 * ratio2 * ratio3 * ratio4 * ratio5 / x  # kappa for each protein - Table 6, Row 4
plt.subplot(5, 1, 4)
plt.ylabel(r'$\log\kappa$')
if logxscale:
    plt.semilogx(x, np.log10(kappa))
else:
    plt.plot(x, np.log10(kappa))
plt.gca().axes.get_xaxis().set_visible(False)
tick_spacing = 3
plt.gca().axes.get_yaxis().set_major_locator(ticker.MultipleLocator(tick_spacing))

##Plot occupation probabilities (Figure 3B)
pocc = 1. / (1. + 1. / (x * kappa))
plt.subplot(5, 1, 5)
plt.ylabel(r'$p_{occ}$')
if logxscale:
    plt.semilogx(x, pocc)
else:
    plt.plot(x, pocc)
plt.xlabel(r'$x_{\mathrm{CHOL}}$')
plt.gca().legend(('beta2' + '-adrenergic', 'serotonin', 'mu' + '-opioid'), fontsize='x-small', loc=0)
plt.legend()
plt.show()

# Print half-saturation ratios for each protein
print
"\nCholesterol fractions for 50% occupancy (x50)"
print
"	beta2-adrenergic (3D4S):%.1g" % get_x50(x, pocc[:, 0])
print
"	5HT-2B (4NC3):%.1g" % get_x50(x, pocc[:, 1])
print
"	mu-opioid (5C1M):%.2g" % get_x50(x, pocc[:, 2])

# Output predictions to file
np.savetxt('predictions.csv', np.asarray(
    [np.log10(x), x, inv_Px, kappa[:, 0], kappa[:, 1], kappa[:, 2], pocc[:, 0], pocc[:, 1], pocc[:, 2]]).T,
           delimiter=',')
np.savetxt('normalized_data.csv', np.asarray([np.log10(sim_x), sim_x, inv_Px_sim / inv_P0, inv_Px_err / inv_P0]).T,
           delimiter=',')