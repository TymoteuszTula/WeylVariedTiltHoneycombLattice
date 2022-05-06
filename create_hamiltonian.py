# create_hamiltonian.py

import numpy as np
from math import pi
from cmath import exp, sin, cos, sqrt
import matplotlib.pyplot as plt

class HamiltonianTilt:

    def generate_nn_vary_ttilda(N, M, t_bar, t_til, t_amp, t_freq, a=1):

        rA = np.sqrt(3) * a

        t_nn = [[t_bar, t_til + t_amp * sin(t_freq * ((m+1/2) * rA)), t_bar, t_bar, 
                 t_til + t_amp * sin(t_freq * ((m-1/2) * rA)), t_bar] for m in range(M)]
        delta_nn = [[[t_nn[i], t_nn[i]] for i in range(M)] for j in range(N)]
        delta_nn = np.array(delta_nn)

        return delta_nn

    def generate_nnn_const(N, M, tp, tx):

        t_nnnA = [tx, tp, tx]
        t_nnnB = [tp, tx, tx]
        delta_nnn = [[[t_nnnA, t_nnnB] for j in range(M)] for i in range(N)]
        delta_nnn = np.array(delta_nnn)

        return delta_nnn

    def generate_nn_const(N, M, t_bar):

        t_nn = [t_bar, 0, t_bar, t_bar, 0, t_bar]
        delta_nn = [[[t_nn, t_nn] for j in range(M)] for i in range(N)]
        delta_nn = np.array(delta_nn)

        return delta_nn

    def __init__(self, N, M, t, delta_nn, delta_nnn, a=1):

        self.N = N
        self.M = M
        self.t = t
        self.a = a
        self.delta_nn = delta_nn
        self.delta_nnn = delta_nnn

        self.a1 = np.array([-a * np.sqrt(3) / 2, -3 * a / 2])
        self.a2 = np.array([a * np.sqrt(3), 0])

        b = 4 * pi / 3 / a

        self.k1 = np.array([0, -b])
        self.k2 = np.array([np.sqrt(3) / 2 * b, -b / 2])

        self.rA = np.array([0, -a / 2])
        self.rB = np.array([0, a / 2])

        self.added_gauge = False

        self.vdot = np.array([np.sqrt(3)/3, 1])

    def in_mapping_tostate(self, n, m, alpha):
        if alpha == "A":
            return int(2 * self.M * (n % self.N) + 2 * (m % self.M))
        elif alpha == "B":
            return int(2 * self.M * (n % self.N) + 2 * (m % self.M) + 1)

    def in_mapping_tocoord(self, i):
        modulo = i % (2 * self.M)
        n = (i - modulo) / (2 * self.M)
        if modulo % 2 == 1:
            m = (modulo - 1) / 2
            alpha = "B"
        else:
            m = modulo / 2
            alpha = "A"
        return int(n), int(m), alpha

    def Aintegral(self, t_bar, t_amp, t_freq, r_abs, r_rel):
        vdot = self.vdot
        t = self.t
        a = self.a

        div = np.dot(vdot, r_rel)
        sinarg = t_freq * np.dot(vdot, r_rel + r_abs)

        exparg = 1 / 2 * t_amp / t**2 / div * sin(sinarg) * (t_amp * sin(sinarg) - 2 * t_bar) * a * sqrt(3) / 2

        return exp(2 * pi * 1j * exparg)

    def Ay(self, t_bar, t_amp, t_freq, ra):
        vdot = self.vdot
        t = self.t
        sinarg = t_freq * np.dot(vdot, ra)

        ans = t_amp**2/t**2 * t_freq * sin(sinarg) * cos(sinarg) - t_bar * t_amp * t_freq / t**2 * cos(sinarg)
        return ans

    def dzeta(self, t_bar, t_amp, t_freq, ra):
        vdot = self.vdot
        t = self.t
        sinarg = t_freq * np.dot(vdot, ra)

        return 2 / t * (t_amp * sin(sinarg) - t_bar)

    def ddzeta(self, t_bar, t_amp, t_freq, ra):
        vdot = self.vdot
        t = self.t
        sinarg = t_freq * np.dot(vdot, ra)

        return 2 * t_amp * t_freq / t * cos(sinarg)

    def return_ts(self, n, m, t_bar, t_amp, t_freq):
        rA = self.rA
        rB = self.rB
        a1 = self.a1
        a2 = self.a2

        raA = rA + n * a1 + m * a2
        raB = rB + n * a1 + m * a2

        tsA = [self.Aintegral(t_bar, t_amp, t_freq, raA, rB),
               self.Aintegral(t_bar, t_amp, t_freq, raA, rB + a1 + a2),
               self.Aintegral(t_bar, t_amp, t_freq, raA, rB + a1)]

        tsB = [self.Aintegral(t_bar, t_amp, t_freq, raB, rA - a1),
               self.Aintegral(t_bar, t_amp, t_freq, raB, rA),
               self.Aintegral(t_bar, t_amp, t_freq, raB, rA - a2 - a1)]

        return np.array([tsA, tsB])

    def return_nn_new(self, n, m, t_bar, t_amp, t_freq):
        rA = self.rA
        rB = self.rB
        a1 = self.a1
        a2 = self.a2

        raA = rA + n * a1 + m * a2
        raB = rB + n * a1 + m * a2

        nn_new_A = [self.Aintegral(t_bar, t_amp, t_freq, raA, rA - a1),
                    self.Aintegral(t_bar, t_amp, t_freq, raA, rA + a2),
                    self.Aintegral(t_bar, t_amp, t_freq, raA, rA + a1 + a2),
                    self.Aintegral(t_bar, t_amp, t_freq, raA, rA + a1),
                    self.Aintegral(t_bar, t_amp, t_freq, raA, rA - a2),
                    self.Aintegral(t_bar, t_amp, t_freq, raA, rA - a1 - a2)]

        nn_new_B = [self.Aintegral(t_bar, t_amp, t_freq, raB, rB - a1),
                    self.Aintegral(t_bar, t_amp, t_freq, raB, rB + a2),
                    self.Aintegral(t_bar, t_amp, t_freq, raB, rB + a2 + a1),
                    self.Aintegral(t_bar, t_amp, t_freq, raB, rB + a1),
                    self.Aintegral(t_bar, t_amp, t_freq, raB, rB - a2),
                    self.Aintegral(t_bar, t_amp, t_freq, raB, rB - a2 - a1)]

        return np.array([nn_new_A, nn_new_B])

    def return_nnn_new(self, n, m, t_bar, t_amp, t_freq):
        rA = self.rA
        rB = self.rB
        a1 = self.a1
        a2 = self.a2

        raA = rA + n * a1 + m * a2
        raB = rB + n * a1 + m * a2

        nnn_new_A = [self.Aintegral(t_bar, t_amp, t_freq, raA, rB + a2),
                     self.Aintegral(t_bar, t_amp, t_freq, raA, rB + 2 * a1 + a2),
                     self.Aintegral(t_bar, t_amp, t_freq, raA, rB - a2)]

        nnn_new_B = [self.Aintegral(t_bar, t_amp, t_freq, raB, rA - 2 * a1 - a2),
                     self.Aintegral(t_bar, t_amp, t_freq, raB, rA + a2),
                     self.Aintegral(t_bar, t_amp, t_freq, raB, rA - a2)]

        return np.array([nnn_new_A, nnn_new_B])

    def add_gauge_sin_tilt(self, t_bar, t_amp, t_freq):

        N = self.N
        M = self.M
        rA = self.rA
        rB = self.rB
        a1 = self.a1
        a2 = self.a2

        t = self.t
        ts = np.array([[[[t, t, t], [t, t, t]] for m in range(M)] for n in range(N)], dtype=complex)
        onsite = np.array([[[1, 1] for m in range(M)] for n in range(N)], dtype=complex)
        delta_nn_new = self.delta_nn.copy().astype(dtype=complex)
        delta_nnn_new = self.delta_nnn.copy().astype(dtype=complex)

        for n in range(self.N):
            for m in range(self.M):
                raA = rA + n * a1 + m * a2
                raB = rB + n * a1 + m * a2

                ts[n,m] *= self.return_ts(n, m, t_bar, t_amp, t_freq)
                delta_nn_new[n, m] *= self.return_nn_new(n, m, t_bar, t_amp, t_freq)
                delta_nnn_new[n, m] *= self.return_nnn_new(n, m, t_bar, t_amp, t_freq)

                # onsite[n, m, 0] = (- 1/4 * self.Ay(t_bar, t_amp, t_freq, raA) * self.dzeta(t_bar, t_amp, t_freq, raA)
                #                    - 1/4 * self.ddzeta(t_bar, t_amp, t_freq, raA))
                # onsite[n, m, 1] = (- 1/4 * self.Ay(t_bar, t_amp, t_freq, raB) * self.dzeta(t_bar, t_amp, t_freq, raB)
                #                    - 1/4 * self.ddzeta(t_bar, t_amp, t_freq, raB))

                onsite[n, m, 0] = 0
                onsite[n, m, 1] = 0

                # # Define all rnms
                # rnm_NupA = np.dot(rB, vdot) + raA[0] + raA[1]
                # rnm_NdrA = np.dot(rB + a1 + a2, vdot) + raA[0] + raA[1]
                # rnm_NdlA = np.dot(rB + a1, vdot) + raA[0] + raA[1]

                # rnm_NupB = np.dot(rA, vdot) + raA[0] + raA[1]
                # rnm_NdrB = np.dot(rA + a1 + a2, vdot) + raA[0] + raA[1]
                # rnm_NdlB = np.dot(rA + a1, vdot) + raA[0] + raA[1]

                # rnm_NNurA = np.dot(rB, vdot) + raA[0] + raA[1]
                # rnm_NNrA = np.dot(rB + a1 + a2, vdot) + raA[0] + raA[1]
                # rnm_NNdrA = np.dot(rB + a1, vdot) + raA[0] + raA[1]
                # rnm_NNdlA = np.dot(rB, vdot) + raA[0] + raA[1]
                # rnm_NNlA = np.dot(rB + a1 + a2, vdot) + raA[0] + raA[1]
                # rnm_NNulA = np.dot(rB + a1, vdot) + raA[0] + raA[1]


        self.ts = ts
        self.onsite = onsite
        self.delta_nn = delta_nn_new
        self.delta_nnn = delta_nnn_new
        self.added_gauge = True
            
            
    def construct_ham_inK(self, k):

        N = self.N
        M = self.M
        t = self.t
        delta_nn = self.delta_nn
        delta_nnn = self.delta_nnn
        a1 = self.a1
        a2 = self.a2
        rA = self.rA
        rB = self.rB
        if self.added_gauge:
            onsite = self.onsite
            ts = self.ts

        H = []
        for i in range(2 * N * M):
            row_i = np.zeros((2 * N * M), dtype=complex)
            n_i, m_i, alpha = self.in_mapping_tocoord(i)


            if alpha == "A":
                jnmB = i + 1

                jnm1mA = self.in_mapping_tostate(n_i-1, m_i, "A")
                
                jnm1mm1A = self.in_mapping_tostate(n_i-1, m_i-1, "A")

                jnmm1A = self.in_mapping_tostate(n_i, m_i-1, "A")
                jnmm1B = self.in_mapping_tostate(n_i, m_i-1, "B")

                jnmp1A = self.in_mapping_tostate(n_i, m_i+1, "A")
                jnmp1B = self.in_mapping_tostate(n_i, m_i+1, "B")

                jnp1mA = self.in_mapping_tostate(n_i+1, m_i, "A")
                jnp1mB = self.in_mapping_tostate(n_i+1, m_i, "B")

                jnp1mp1A = self.in_mapping_tostate(n_i+1, m_i+1, "A")
                jnp1mp1B = self.in_mapping_tostate(n_i+1, m_i+1, "B")

                jnp2mp1B = self.in_mapping_tostate(n_i+2, m_i+1, "B")

                # t hoppings

                if self.added_gauge:
                    row_i[jnmB] += ts[n_i, m_i, 0][0] * exp(1j * np.dot(rA - rB, k))
                    row_i[jnp1mB] += ts[n_i, m_i, 0][2] * exp(1j * np.dot(rA - (a1 + rB), k))
                    row_i[jnp1mp1B] += ts[n_i, m_i, 0][1] * exp(1j * np.dot(rA - (a1 + a2 + rB), k))
                else:
                    row_i[jnmB] += t * exp(1j * np.dot(rA - rB, k))
                    row_i[jnp1mB] += t * exp(1j * np.dot(rA - (a1 + rB), k))
                    row_i[jnp1mp1B] += t * exp(1j * np.dot(rA - (a1 + a2 + rB), k))

                # delta_nn hoppings

                row_i[jnm1mA] += delta_nn[n_i, m_i, 0][0] * exp(1j * np.dot(rA - (-a1 + rA), k))
                row_i[jnmp1A] += delta_nn[n_i, m_i, 0][1]* exp(1j * np.dot(rA - (a2 + rA), k))
                row_i[jnp1mp1A] += delta_nn[n_i, m_i, 0][2] * exp(1j * np.dot(rA - (a1 + a2 + rA), k))
                row_i[jnp1mA] += delta_nn[n_i, m_i, 0][3] * exp(1j * np.dot(rA - (a1 + rA), k))
                row_i[jnmm1A] += delta_nn[n_i, m_i, 0][4] * exp(1j * np.dot(rA - (-a2 + rA), k))
                row_i[jnm1mm1A] += delta_nn[n_i, m_i, 0][5] * exp(1j * np.dot(rA - (-a1 -a2 + rA), k))

                # delta_nnn hoppings

                row_i[jnmp1B] += delta_nnn[n_i, m_i, 0][0] * exp(1j * np.dot(rA - (a2 + rB), k))
                row_i[jnp2mp1B] += delta_nnn[n_i, m_i, 0][1] * exp(1j * np.dot(rA - (2 * a1 + a2 + rB), k))
                row_i[jnmm1B] += delta_nnn[n_i, m_i, 0][2] * exp(1j * np.dot(rA - (-a2 + rB), k))

                # self term if gauge

                if self.added_gauge:
                    row_i[i] += onsite[n_i, m_i, 0]

            # Need to change so to include terms that can have different j but in the same cell.
                

            if alpha == "B":
                jnmA = i - 1

                jnm2mm1A = self.in_mapping_tostate(n_i-2, m_i-1, "A")

                jnm1mA = self.in_mapping_tostate(n_i-1, m_i, "A")
                jnm1mB = self.in_mapping_tostate(n_i-1, m_i, "B")
                
                jnm1mm1A = self.in_mapping_tostate(n_i-1, m_i-1, "A")
                jnm1mm1B = self.in_mapping_tostate(n_i-1, m_i-1, "B")

                jnmm1A = self.in_mapping_tostate(n_i, m_i-1, "A")
                jnmm1B = self.in_mapping_tostate(n_i, m_i-1, "B")

                jnmp1A = self.in_mapping_tostate(n_i, m_i+1, "A")
                jnmp1B = self.in_mapping_tostate(n_i, m_i+1, "B")

                jnp1mB = self.in_mapping_tostate(n_i+1, m_i, "B")

                jnp1mp1B = self.in_mapping_tostate(n_i+1, m_i+1, "B")

                # t_hoppings

                if self.added_gauge:
                    row_i[jnmA] += ts[n_i, m_i, 1][1] * exp(1j * np.dot(rB - rA, k))
                    row_i[jnm1mA] += ts[n_i, m_i, 1][0] * exp(1j * np.dot(rB - (-a1 + rA), k))
                    row_i[jnm1mm1A] += ts[n_i, m_i, 1][2] * exp(1j * np.dot(rB - (-a1 -a2 + rA), k))
                else:   
                    row_i[jnmA] += t * exp(1j * np.dot(rB - rA, k))
                    row_i[jnm1mA] += t * exp(1j * np.dot(rB - (-a1 + rA), k))
                    row_i[jnm1mm1A] += t * exp(1j * np.dot(rB - (-a1 -a2 + rA), k))

                # delta_nn hoppings
                
                row_i[jnm1mB] += delta_nn[n_i, m_i, 1][0] * exp(1j * np.dot(rB - (-a1 + rB), k))
                row_i[jnmp1B] += delta_nn[n_i, m_i, 1][1] * exp(1j * np.dot(rB - (a2 + rB), k))
                row_i[jnp1mp1B] += delta_nn[n_i, m_i, 1][2] * exp(1j * np.dot(rB - (a1 + a2 + rB), k))
                row_i[jnp1mB] += delta_nn[n_i, m_i, 1][3] * exp(1j * np.dot(rB - (a1 + rB), k))
                row_i[jnmm1B] += delta_nn[n_i, m_i, 1][4] * exp(1j * np.dot(rB - (-a2 + rB), k))
                row_i[jnm1mm1B] += delta_nn[n_i, m_i, 1][5] * exp(1j * np.dot(rB - (-a1 -a2 + rB), k))

                # delta_nnn hoppings

                row_i[jnm2mm1A] += delta_nnn[n_i, m_i, 1][0] * exp(1j * np.dot(rB - (-2 * a1 -a2 + rA), k))
                row_i[jnmp1A] += delta_nnn[n_i, m_i, 1][1] * exp(1j * np.dot(rB - (a2 + rA), k))
                row_i[jnmm1A] += delta_nnn[n_i, m_i, 1][2] * exp(1j * np.dot(rB - (-a2 + rA), k))

                # self term if gauge

                if self.added_gauge:
                    row_i[i] += onsite[n_i, m_i, 1]

            H.append(row_i)

        H = np.array(H)
        self.H = H

    def construct_ham(self):

        N = self.N
        M = self.M
        t = self.t
        delta_nn = self.delta_nn
        delta_nnn = self.delta_nnn

        H = []
        for i in range(2 * N * M):
            row_i = np.zeros((2 * N * M), dtype=complex)
            n_i, m_i, alpha = self.in_mapping_tocoord(i)


            if alpha == "A":
                jnmB = i + 1

                jnm1mA = self.in_mapping_tostate(n_i-1, m_i, "A")
                
                jnm1mm1A = self.in_mapping_tostate(n_i-1, m_i, "A")

                jnmm1A = self.in_mapping_tostate(n_i, m_i-1, "A")
                jnmm1B = self.in_mapping_tostate(n_i, m_i-1, "B")

                jnmp1A = self.in_mapping_tostate(n_i, m_i+1, "A")
                jnmp1B = self.in_mapping_tostate(n_i, m_i+1, "B")

                jnp1mA = self.in_mapping_tostate(n_i+1, m_i, "A")
                jnp1mB = self.in_mapping_tostate(n_i+1, m_i, "B")

                jnp1mp1A = self.in_mapping_tostate(n_i+1, m_i+1, "A")
                jnp1mp1B = self.in_mapping_tostate(n_i+1, m_i+1, "B")

                jnp2mp1B = self.in_mapping_tostate(n_i+2, m_i+1, "B")

                # t hoppings

                row_i[jnmB] += t
                row_i[jnp1mB] += t
                row_i[jnp1mp1B] += t

                # delta_nn hoppings

                row_i[jnm1mA] += delta_nn[n_i, m_i, 0][0]
                row_i[jnmp1A] += delta_nn[n_i, m_i, 0][1]
                row_i[jnp1mp1A] += delta_nn[n_i, m_i, 0][2]
                row_i[jnp1mA] += delta_nn[n_i, m_i, 0][3]
                row_i[jnmm1A] += delta_nn[n_i, m_i, 0][4]
                row_i[jnm1mm1A] += delta_nn[n_i, m_i, 0][5]

                # delta_nnn hoppings

                row_i[jnmp1B] += delta_nnn[n_i, m_i, 0][0]
                row_i[jnp2mp1B] += delta_nnn[n_i, m_i, 0][1]
                row_i[jnmm1B] += delta_nnn[n_i, m_i, 0][2]

            # Need to change so to include terms that can have different j but in the same cell.
                

            if alpha == "B":
                jnmA = i - 1

                jnm2mm1A = self.in_mapping_tostate(n_i-2, m_i-1, "A")

                jnm1mA = self.in_mapping_tostate(n_i-1, m_i, "A")
                jnm1mB = self.in_mapping_tostate(n_i-1, m_i, "B")
                
                jnm1mm1A = self.in_mapping_tostate(n_i-1, m_i, "A")
                jnm1mm1B = self.in_mapping_tostate(n_i-1, m_i, "B")

                jnmm1A = self.in_mapping_tostate(n_i, m_i-1, "A")
                jnmm1B = self.in_mapping_tostate(n_i, m_i-1, "B")

                jnmp1A = self.in_mapping_tostate(n_i, m_i+1, "A")
                jnmp1B = self.in_mapping_tostate(n_i, m_i+1, "B")

                jnp1mB = self.in_mapping_tostate(n_i+1, m_i, "B")

                jnp1mp1B = self.in_mapping_tostate(n_i+1, m_i+1, "B")

                # t_hoppings

                row_i[jnmA] += t
                row_i[jnm1mA] += t
                row_i[jnm1mm1A] += t

                # delta_nn hoppings
                
                row_i[jnm1mB] += delta_nn[n_i, m_i, 1][0]
                row_i[jnmp1B] += delta_nn[n_i, m_i, 1][1]
                row_i[jnp1mp1B] += delta_nn[n_i, m_i, 1][2]
                row_i[jnp1mB] += delta_nn[n_i, m_i, 1][3]
                row_i[jnmm1B] += delta_nn[n_i, m_i, 1][4]
                row_i[jnm1mm1B] += delta_nn[n_i, m_i, 1][5]

                # delta_nnn hoppings

                row_i[jnm2mm1A] += delta_nnn[n_i, m_i, 1][0]
                row_i[jnmp1A] += delta_nnn[n_i, m_i, 1][1]
                row_i[jnmm1A] += delta_nnn[n_i, m_i, 1][2]

            H.append(row_i)

        H = np.array(H)
        self.H = H





if __name__ == "__main__":
    
    case = "test4"

    if case == "test0":

        N = 1
        M = 1
        t = 1
        delta_nn = np.zeros((N, M, 2, 6))
        delta_nnn = np.zeros((N, M, 2, 3))

        ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn)

        indicies = [[ham.in_mapping_tostate(n, m, "A") for m in range(M)] for n in range(N)]
        coords = [[ham.in_mapping_tocoord(indicies[n][m]) for m in range(M)] for n in range(N)]

        print(indicies)
        print(coords)

    if case == "test1":

        N = 1
        M = 1
        t = 1
        a = 1
        delta_nn = np.zeros((N, M, 2, 6))
        delta_nnn = np.zeros((N, M, 2, 3))

        ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a)

        b = 2 * pi / 3 / a

        k1 = np.arange(-pi, pi, pi / 100)
        k2 = np.arange(-pi, pi, pi / 100)

        spectrum = np.zeros((len(k1), len(k2), 2))

        for i in range(len(k1)):
            for j in range(len(k2)):
                ham.construct_ham_inK(np.array(([k1[i], k2[j]])))
                ev = np.linalg.eigvals(ham.H)
                spectrum[i, j] = ev
    
        K1, K2 = np.meshgrid(k1, k2)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(K1, K2, spectrum[:,:,0], color="tab:blue", antialiased=False)
        surf = ax.plot_surface(K1, K2, spectrum[:,:,1], color="tab:orange", antialiased=False)

        plt.show()

    if case == "test2":

        # Constant Tilt
        N = 4
        M = 1
        t = 1
        a = 1
        delta_nnn = np.zeros((N, M, 2, 3))

        t_bar = 0.3
        t_til = 0
        t_A = [t_bar, t_til, t_bar, t_bar, t_til, t_bar]
        t_B = [t_bar, t_til, t_bar, t_bar, t_til, t_bar]
        delta_nn = [[[t_A, t_B] for j in range(M)] for i in range(N)]
        delta_nn = np.array(delta_nn)

        ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a)
        
        k1 = ham.k1
        k2 = ham.k2

        k_max = 0.5
        k_num = 2000
        k_list = [k_max * i / k_num *(k1 + k2) for i in range(k_num)]
        spectrum = []

        for i in range(k_num):
            ham.construct_ham_inK(k_list[i])
            ev = np.linalg.eigvals(ham.H)
            spectrum.append(ev)

        spectrum = np.array(spectrum)

        fig, ax = plt.subplots()
        ax.plot(spectrum)
        
        plt.show()

    if case == "test3":

        # Constant Tilt with shifted cones
        N = 1
        M = 4
        t = 1
        a = 1

        t_bar = 0
        t_til = 0
        t_A = [t_bar, t_til, t_bar, t_bar, t_til, t_bar]
        t_B = [t_bar, t_til, t_bar, t_bar, t_til, t_bar]
        delta_nn = [[[t_A, t_B] for j in range(M)] for i in range(N)]
        delta_nn = np.array(delta_nn)

        tp = 0
        tx = 0
        t_nnn_A = [tx, tp, tx]
        t_nnn_B = [tp, tx, tx]
        delta_nnn = [[[t_nnn_A, t_nnn_B] for j in range(M)] for i in range(N)]
        delta_nnn = np.array(delta_nnn)

        ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a)
        
        k1 = ham.k1
        k2 = ham.k2

        k_max = 0.5
        k_num = 2000
        k_list = [k_max * i / k_num *(k1 + k2) for i in range(k_num)]
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

    if case == "test4":

        # Constant Tilt with shifted cones
        N = 10
        M = 1
        t = 1
        a = 1
        b = 4 * pi / 3 / a
        delta_nnn = np.zeros((N, M, 2, 3))


        t_bar = 0
        t_amp = 0
        p = 3
        try:
            t_freq = 2 * pi * p / a / (M - 1)
        except:
            t_freq = 1 
        delta_nn = HamiltonianTilt.generate_nn_vary_ttilda(N, M, t_bar, t_amp, t_freq, a)

        ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a)
        
        k1 = ham.k1
        k2 = np.array([-np.sqrt(3) / 2 * b, -b / 2])

        k_shift = 2 / 6 * (k1 + k2)

        k_max = 2/6
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
