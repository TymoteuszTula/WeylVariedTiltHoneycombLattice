# check_eigvectors.py

import numpy as np
from math import pi
import matplotlib.pyplot as plt
from create_hamiltonian import HamiltonianTilt

if __name__ == "__main__":

    case = "check eigvecs"

    if case == "check eigvecs":

        N = 1
        M = 3
        t = 1
        a = 1
        b = 4 * pi / 3 / a
        delta_nnn = np.zeros((N, M, 2, 3))

        t_bar = 0
        t_amp = 0.1
        p = 1
        try:
            t_freq = 2 * pi * p / a / (M - 1)
        except:
            t_freq = 1 
        delta_nn = HamiltonianTilt.generate_nn_vary_ttilda(N, M, t_bar, t_amp, t_freq, a)

        ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a)
        
        k1 = ham.k1
        k2 = np.array([-np.sqrt(3) / 2 * b, -b / 2])

        k_shift = 2 / 6 * (k1 + k2)

        k_max = 2/20
        k_num = 2000
        k_list = [k_shift + k_max * (i-k_num/2) / k_num * (k1 + k2) for i in range(k_num)]
        spectrum = []

        for i in range(k_num):
            ham.construct_ham_inK(k_list[i])
            ev = np.linalg.eigvals(ham.H)
            ev = np.sort(ev)
            spectrum.append(ev)

        spectrum = np.array(spectrum)

        fig, ax = plt.subplots()
        ax.plot(spectrum)
        
        plt.show()