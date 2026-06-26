import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from scipy.optimize import curve_fit

# Grace Brannigan 4/2018 rev 9/2018
# Accompanies "A streamlined, general approach for computing ligand binding free energies
# and its application to GPCR-bound cholesterol"
# by R. Salari, T. Joseph, R. Lohia, J. Henin, G. Brannigan
# Script used to calculate quantities in Table 5 and 6 and predictions in Figure 3;
# also plots chemical potential and derivative with respect to cholesterol concentration

# ---------------------------------------------------------------------------
# Load parameters from CSV files
# ---------------------------------------------------------------------------

_params = pd.read_csv("sim_params.csv", index_col="parameter")["value"]

kT = _params["kT"]
totLipid = _params["totLipid"]
alpha = _params["alpha"]
thetaR = _params["thetaR_pi_factor"] * np.pi
zR = _params["zR"]
A = _params["A"]
rR = _params["rR"]
RFEP = _params["RFEP"]
xdecades = _params["xdecades"]
xmax = _params["xmax"]
nx = int(_params["nx"])
logxscale = int(_params["logxscale"])

_membrane = pd.read_csv("membrane_afep.csv", index_col="quantity")

sim_x = _membrane.loc["sim_x"].values.astype(float)
AFEP1 = _membrane.loc["AFEP1"].values.astype(float)
AFEP1_err = _membrane.loc["AFEP1_err"].values.astype(float)

_protein = pd.read_csv("protein_afep.csv", index_col="protein")

replica_cols = ["replica_1", "replica_2", "replica_3"]
AFEP_3D4S = _protein.loc["beta2_adrenergic", replica_cols].values.astype(float)
AFEP_4CN3 = _protein.loc["serotonin", replica_cols].values.astype(float)
AFEP_5C1M = _protein.loc["mu_opioid", replica_cols].values.astype(float)
labels = _protein["label"].tolist()

# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------


def mix_model(x, P0, h0):
    # Quadratic mixture model
    # P0: ideal bulk/gas partition coefficient
    # h0: enthalpy of mixing
    return P0 * np.exp(-((1 - x) ** 2) * h0 / kT)


def inv_mix_model(x, invP0, h0):
    # Quadratic mixture model
    # invP0: ideal gas/bulk partition coefficient
    # h0: enthalpy of mixing
    return invP0 * np.exp((1 - x) ** 2 * h0 / kT)


def get_x50(x, y):
    """Return x value for which y is closest to 0.5."""
    x50 = x[(np.abs(y - 0.5)).argmin()].item()
    if (np.max(y) < 0.5) or (np.min(y) > 0.5):
        print(
            "Warning: Predicted occupancy does not cross 0.5, "
            "will print x for which occupancy is closest to 0.5:"
        )
    return x50


# ---------------------------------------------------------------------------
# Concentration range for prediction
# ---------------------------------------------------------------------------

x = (10 ** (np.linspace(-xdecades, 0, nx)) * xmax).reshape(nx, 1)
np.set_printoptions(precision=5)

# ---------------------------------------------------------------------------
# Bulk/gas partition coefficient & non-ideality of bulk
# ---------------------------------------------------------------------------

Px_sim = np.exp(AFEP1 / kT)
inv_Px_sim = np.exp(-AFEP1 / kT)
inv_Px_err = inv_Px_sim * (AFEP1_err / kT)

popt, pcov = curve_fit(inv_mix_model, sim_x, inv_Px_sim)
inv_P0, h0 = popt[0], popt[1]
P0 = 1 / inv_P0

print(f"Fitted Ideal Gas/Bulk Partition Coefficient (P0): {P0:.2g}")
print(f"Fitted Enthalpy of Mixing in kcal/mol (h0): {h0:.2g}\n")

inv_Px = inv_mix_model(x, inv_P0, h0)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

### Gas-Bulk Partition Coefficient (Figure 3A, Row 2 of Table 5)
plt.subplot(5, 1, 1)
plt.ylabel(r"$1/P_x$")
plt.errorbar(sim_x, inv_Px_sim, yerr=inv_Px_err)
if logxscale:
    plt.semilogx(x, inv_Px, "k")
else:
    plt.plot(x, inv_Px, "k")
plt.gca().axes.get_xaxis().set_visible(False)

### Chemical Potential
mu = kT * np.log(x) + h0 * (1 - x) ** 2
plt.subplot(5, 1, 2)
plt.ylabel(r"$\mu - \mu^0$" + "\n(kcal/mol)", multialignment="center")
if logxscale:
    plt.semilogx(x, mu)
else:
    plt.plot(x, mu)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(3))

### x Derivative of Chemical Potential
dmu = kT / x - 2 * h0 * (1 - x)
plt.subplot(5, 1, 3)
plt.ylabel(r"$d\mu/dx$" + "\n(kcal/mol)", multialignment="center")
if logxscale:
    plt.semilogx(x, dmu)
else:
    plt.plot(x, dmu)
plt.gca().axes.get_xaxis().set_visible(False)

# ---------------------------------------------------------------------------
# Occupation probabilities (Tables 5 & 6)
# ---------------------------------------------------------------------------

AFEP2 = np.array([np.mean(AFEP_3D4S), np.mean(AFEP_4CN3), np.mean(AFEP_5C1M)])

ratio1 = x * totLipid * alpha
ratio2 = inv_Px
ratio3 = (2.0 / 3.0 * np.pi) * (rR**3.0 / zR / A) / (1 - np.cos(thetaR))
ratio4 = np.exp(-RFEP / kT)
ratio5 = np.exp(AFEP2 / kT).reshape(1, 3)

kappa = ratio1 * ratio2 * ratio3 * ratio4 * ratio5 / x

### kappa
plt.subplot(5, 1, 4)
plt.ylabel(r"$\log\kappa$")
if logxscale:
    plt.semilogx(x, np.log10(kappa))
else:
    plt.plot(x, np.log10(kappa))
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(3))

### Occupation probabilities (Figure 3B)
pocc = 1.0 / (1.0 + 1.0 / (x * kappa))
plt.subplot(5, 1, 5)
plt.ylabel(r"$p_{occ}$")
if logxscale:
    for col, label in zip(pocc.T, labels):
        plt.semilogx(x, col, label=label)
else:
    for col, label in zip(pocc.T, labels):
        plt.plot(x, col, label=label)
plt.xlabel(r"$x_{\mathrm{CHOL}}$")
plt.legend(fontsize="x-small", loc=0)
plt.savefig("test.pdf")

# ---------------------------------------------------------------------------
# Half-saturation ratios
# ---------------------------------------------------------------------------

print("\nCholesterol fractions for 50% occupancy (x50)")
for col, label in zip(pocc.T, labels):
    print(f"  {label}: {get_x50(x, col):.2g}")

# ---------------------------------------------------------------------------
# Output predictions
# ---------------------------------------------------------------------------

predictions_list = [
    np.log10(x),
    x,
    inv_Px,
    kappa[:, 0],
    kappa[:, 1],
    kappa[:, 2],
    pocc[:, 0],
    pocc[:, 1],
    pocc[:, 2],
]
predictions_flat = [np.asarray(v).flatten() for v in predictions_list]
np.savetxt("predictions.csv", np.asarray(predictions_flat).T, delimiter=",")

normalized_data_list = [
    np.log10(sim_x),
    sim_x,
    inv_Px_sim / inv_P0,
    inv_Px_err / inv_P0,
]
normalized_data_flat = [np.asarray(v).flatten() for v in normalized_data_list]
np.savetxt("normalized_data.csv", np.asarray(normalized_data_flat).T, delimiter=",")
