import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from blindpd.spectrum import multiplicity, conj_factor
from blindpd.tracking import hungarian
from blindpd.so5 import rep_name
from blindpd.ope_exact import me_value, z_den, insertion_vector
from blindpd.ope import match_states
import pandas as pd

def test():
    assert multiplicity(1, 0, 1) == 1 and multiplicity(1, 0, 0) == 0
    assert multiplicity(2, 0, 0) == 1 and multiplicity(2, 0, 2) == 1 and multiplicity(2, 0, 1) == 0
    assert multiplicity(1, 2, 1) == 1 and multiplicity(1, 2, 2) == 1 and multiplicity(1, 2, 3) == 1
    assert abs(conj_factor(0, 1) + np.sqrt(3)) < 1e-15 and conj_factor(2, 2) == 1
    C = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    a = dict(hungarian(C)); assert sum(C[i, j] for i, j in a.items()) == 5
    assert rep_name(8) == "35p" and rep_name(6) == "35" and rep_name(2) == "5"
    # v0.3.1: odd-l tensor pairing sign and signed S normalisation
    class _Od: beta = [2.0, 3.0]; vac = 8.0
    assert me_value("tensorG", [1.0, 0.25], _Od, 0.0, 1, 1, 1) == 0.75
    assert me_value("tensorG", [1.0, 0.25], _Od, 0.0, 1, 1, 2) == 1.25
    assert me_value("singlet", [1.0, 2.0], _Od, 1.0, 1, 1, 0) == 0.0
    Zt = {(5, "singlet"): -1.0, (5, "tensorG"): 0.5, (5, "gamma2"): 0.3}
    assert z_den(Zt, 5, "singlet") == -1.0 and z_den(Zt, 5, "tensorG") == 0.5
    # v0.3.2 (A1): truncation bound on the audit counterexample T=e1, z=(sqrt(.96), .2), O|x>=(1, 1000)
    T = np.array([[1.0], [0.0]]); z = np.array([np.sqrt(.96), .2]); Ox = np.array([1.0, 1000.0])
    cz = T.T @ z; r2 = 1 - float(cz @ cz); approx = float(cz @ (T.T @ Ox)); exact = float(z @ Ox)
    G = float(Ox @ Ox); C = float((T.T @ Ox) @ (T.T @ Ox)); bound = np.sqrt(r2) * np.sqrt(G - C)
    assert abs(exact - approx) <= bound + 1e-12 and bound > 199
    assert list(insertion_vector("tensorG", 3, _Od)) == [1.0, -1.0] and list(insertion_vector("singlet", 0, _Od)) == [2.0, 3.0]
    # v0.3.2: ambiguous energy match is not an identity
    prim = pd.DataFrame(dict(N=[5, 5], C2=[2, 2], L=[0, 0], P=[1, 1], Delta=[1.0, 1.0000001], phys_id=["a", "b"], track_id=["A", "B"]))
    rows = pd.DataFrame(dict(N=[5], X_C2=[2], X_L=[0], X_P=[1], X_Delta=[1.00000005], Z_C2=[2], Z_L=[0], Z_P=[1], Z_Delta=[1.0]))
    r = match_states(rows, prim)
    assert r.X_match.iloc[0] == "ambiguous_energy_match" and r.X_phys_id.iloc[0] is None
    print("UNIT TESTS PASS")

if __name__ == "__main__":
    test()
