# class_show_curser.py

"""Code contains `UISpectrumTilted` class which """

from re import A
from matplotlib.axes import Axes
import numpy as np
from math import floor, pi, sqrt
from curses import wrapper
import curses as cs
from create_hamiltonian import HamiltonianTilt
from multiprocessing import Process
from class_show import UISpectrumTilted
import matplotlib.pyplot as plt
import scipy.linalg as scla
from mpl_toolkits.mplot3d import Axes3D

class UISpectrumTilted2(UISpectrumTilted):

    def __init__(self, stdscr=None):
        self.stdscr = stdscr
        self.show_lines = 41
        self.welcome_lines = 3

        self.N = 1
        self.M = 1
        self.t = 1
        self.a = 1
        self.t_bar = 0
        self.t_amp = 0
        self.t_til = 0
        self.tp = 0
        self.tx = 0
        self.p = 0
        self.kpath = "KZOOMIN"
        self.no_k = 100
        self.zoomin = 0.1
        self.is_plot_showing = False
        self.scale_bzone = False

    def draw_kpath(self):
        UISpectrumTilted.draw_kpath(self)

    def draw_kpath_an(self):
        UISpectrumTilted.draw_kpath_an(self)

    def draw_kpath_gauge(self):
        # Here goes the code for gauge field
        N = self.N
        M = self.M
        t = self.t
        tp = self.tp
        tx = self.tx
        t_bar = self.t_bar
        t_amp = self.t_amp
        try:
            t_freq = 2 * pi * self.p / self.a / M
        except:
            t_freq = 1
        delta_nnn = HamiltonianTilt.generate_nnn_const(N, M, tp, tx)
        delta_nn = HamiltonianTilt.generate_nn_const(N, M, t_bar)
        ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a=self.a)
        ham.add_gauge_sin_tilt(t_bar, t_amp, t_freq)
        
        eig_values = []
        
        k, ticks, label_ticks, add_ticks = self.get_k_path(ham)

        for i in range(len(k)):
            ham.construct_ham_inK(k[i])
            eigvals = np.linalg.eigvals(ham.H)
            eigvals = np.real(np.sort(eigvals))
            eig_values.append(eigvals)

        self.eig_values = eig_values

        # if self.is_plot_showing:
        #     self.pro.kill()

        self.pro = Process(target=self.plot_k_path, args=([eig_values, ticks, label_ticks, add_ticks]))
        self.pro.start()
        self.is_plot_showing = True

    def draw_kpath_folded(self):
        # draw kpath folded
        N = self.N
        M = self.M
        t = self.t
        tp = self.tp
        tx = self.tx
        t_bar = self.t_bar
        t_amp = self.t_amp
        t_til = self.t_til
        rA = np.sqrt(3) * self.a
        try:
            t_freq = 2 * pi * self.p / M / rA
        except:
            t_freq = 0
        delta_nnn = HamiltonianTilt.generate_nnn_const(N, M, tp, tx)
        delta_nn = HamiltonianTilt.generate_nn_vary_ttilda(N, M, t_bar, t_til, t_amp, t_freq, self.a)
        ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a=self.a)

        eig_values = []

        k1 = -ham.k1
        k2 = ham.k2
        k_res = 100

        k1f = k1 / N
        k2f = k2 / M

        kp1 = [k1 / 2 * i/k_res for i in range(k_res)]
        kp2 = [k1/2 + (k1/6 + k2/3) * i/k_res for i in range(k_res)]
        kp3 = [2*k1/3 + k2/3 + (-k1/3 + k2/3) * i/k_res for i in range(k_res)]
        kp4 = [k1/3 + 2*k2/3 + (-k1/3 - k2/6) * i/k_res for i in range(k_res)]
        kp5 = [k2/2 + (-k2/2) * i/k_res for i in range(k_res + 1)]
        k = kp1 + kp2 + kp3 + kp4 + kp5

        ticks = [0, k_res, 2 * k_res, 3 * k_res, 4 * k_res, 5 * k_res + 1]
        label_ticks = [r"$\Gamma$", "M", "K", "K'", "M'", r"$\Gamma$"]
        add_ticks = False

        for i in range(len(k)):
            ham.construct_ham_inK(self.project_to_smaller_wigner_cell(k1f, k2f, k[i]))
            eigvals = np.linalg.eigvals(ham.H)
            eigvals = np.real(np.sort(eigvals))
            eig_values.append(eigvals)

        self.pro = Process(target=self.plot_k_path, args=([eig_values, ticks, label_ticks, add_ticks]))
        self.pro.start()
        self.is_plot_showing = True

    def return_kpath_wignercell(self):
        # draw kpath folded
        N = self.N
        M = self.M
        t = self.t
        tp = self.tp
        tx = self.tx
        t_bar = self.t_bar
        t_amp = self.t_amp
        t_til = self.t_til
        rA = np.sqrt(3) * self.a
        try:
            t_freq = 2 * pi * self.p / M / rA
        except:
            t_freq = 0
        delta_nnn = HamiltonianTilt.generate_nnn_const(N, M, tp, tx)
        delta_nn = HamiltonianTilt.generate_nn_vary_ttilda(N, M, t_bar, t_til, t_amp, t_freq, self.a)
        ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a=self.a)

        eig_values = []

        k1 = -ham.k1
        k2 = ham.k2
        k_res = 100

        k1f = k1 / N
        k2f = k2 / M

        kp1 = [k1 / 2 * i/k_res for i in range(k_res)]
        kp2 = [k1/2 + (k1/6 + k2/3) * i/k_res for i in range(k_res)]
        kp3 = [2*k1/3 + k2/3 + (-k1/3 + k2/3) * i/k_res for i in range(k_res)]
        kp4 = [k1/3 + 2*k2/3 + (-k1/3 - k2/6) * i/k_res for i in range(k_res)]
        kp5 = [k2/2 + (-k2/2) * i/k_res for i in range(k_res + 1)]
        k = kp1 + kp2 + kp3 + kp4 + kp5

        k_folded_list = []

        for i in range(len(k)):
            k_folded = self.project_to_smaller_wigner_cell(k1f, k2f, k[i])
            k_folded_list.append(k_folded)
        
        return k, k_folded_list, k1, k2
        

    def project_to_smaller_wigner_cell(self, k1, k2, k_target):
        eps = 0.0000001
        N = self.N
        M = self.M
        k1_i = k1 * N
        k2_i = k2 * M
        lenk = 4 * pi / 3 / self.a

        # proj_k1 = sqrt(np.dot(k1_i, k_target)) / lenk
        # proj_k2 = sqrt(np.dot(k2_i, k_target)) / lenk

        proj_k2 = k_target[0] / k2_i[0]
        proj_k1 = (k_target[1] - k2_i[1] * proj_k2) / k1_i[1]

        m = floor(M * proj_k2 + eps)
        n = floor(N * proj_k1 + eps)

        return k_target - n * k1 - m * k2


    def plot_bzone_eq(self, en, k1, k2, k_res):

        X = np.array([[(k1[0] * i + k2[0] * j)/ k_res for j in range(k_res)] for i in range(k_res)])
        Y = np.array([[(k1[1] * i + k2[1] * j)/ k_res for j in range(k_res)] for i in range(k_res)])
        N = self.N
        M = self.M

        fig = plt.figure(0)
        ax = fig.gca(projection='3d')

        k1_i = k1 * N
        k2_i = k2 * M

        K = k1_i + (k2_i - k1_i) / 3
        KP = k1_i + 2 * (k2_i - k1_i) / 3

        tK = self.project_to_smaller_wigner_cell(k1, k2, K)
        tKP = self.project_to_smaller_wigner_cell(k1, k2, KP)

        # for e in range(2 * self.M * self.N):
        #     Z = en[:,:,e]
        #     ax.plot_surface(X, Y, Z)

        ZMIN = en[:,:,self.N*self.M-1]
        ZMAX = en[:,:,self.N*self.M]

        ax.plot_surface(X, Y, ZMIN)
        ax.plot_surface(X, Y, ZMAX)
        if self.scale_bzone:
            ax.set_xlim(left=0, right=4)
        ax.scatter([tK[0]], [tK[1]], 0, marker='o', c='b')
        ax.scatter([tKP[0]], [tKP[1]], 0, marker='o', c='r')
        ax.set_xlabel("KX")
        ax.set_ylabel("KY")
        ax.set_zlabel("Energy")

        plt.show()
        

    def draw_bzone_eq(self):
        N = self.N
        M = self.M
        t = self.t
        tp = self.tp
        tx = self.tx
        t_bar = self.t_bar
        t_amp = self.t_amp
        t_til = self.t_til
        rA = np.sqrt(3) * self.a
        try:
            t_freq = 2 * pi * self.p / rA / M
        except:
            t_freq = 0
        delta_nnn = HamiltonianTilt.generate_nnn_const(N, M, tp, tx)
        delta_nn = HamiltonianTilt.generate_nn_vary_ttilda(N, M, t_bar, t_til, t_amp, t_freq, self.a)
        ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a=self.a)

        k1 = -ham.k1 / N
        k2 = ham.k2 / M
        k_res = 100

        en = np.zeros((k_res, k_res, 2 * M * N))

        for i in range(k_res):
            for j in range(k_res):
                ham.construct_ham_inK(i / k_res * k1 + j / k_res * k2)
                eigvals = scla.eigh(ham.H, eigvals_only=True)
                en[i, j, :] = eigvals

        self.pro = Process(target=self.plot_bzone_eq, args=([en, k1, k2, k_res]))
        self.pro.start()
        self.is_plot_showing = True 



    def plot_dos(self, eig_values):
        plt.figure(0)
        plt.hist(eig_values, bins=1000, orientation='horizontal')
        plt.ylabel("DOS")
        plt.show()

    def draw_dos(self):
        N = self.N
        M = self.M
        t = self.t
        tp = self.tp
        tx = self.tx
        t_bar = self.t_bar
        t_amp = self.t_amp
        try:
            t_freq = 2 * pi * self.p / self.a / (M - 1)
        except:
            t_freq = 1
        delta_nnn = HamiltonianTilt.generate_nnn_const(N, M, tp, tx)
        delta_nn = HamiltonianTilt.generate_nn_vary_ttilda(N, M, t_bar, t_amp, t_freq)
        ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a=self.a)

        eig_values = []
        k1 = ham.k1
        k2 = ham.k2

        k = [[i/self.no_k * k1 + j/self.no_k * k2 for j in range(self.no_k)] for i in range(self.no_k)]

        for i in range(self.no_k):
            for j in range(self.no_k):
                ham.construct_ham_inK(k[i][j])
                eigvals = np.linalg.eigvals(ham.H)
                eigvals = np.real(eigvals)
                eig_values.append(eigvals)

        eig_values = np.array(eig_values)
        eig_values = eig_values.flatten()

        self.pro = Process(target=self.plot_dos, args=([eig_values]))
        self.pro.start()
        self.is_plot_showing = True

    def plot_2d_groundenergy(self, en, kx, ky):

        KX, KY = np.meshgrid(kx, ky)

        plt.figure(0)
        plt.contourf(KY, KX, en)
        plt.show()


    def draw_2d_en_level(self):

        N = self.N
        M = self.M
        t = self.t
        tp = self.tp
        tx = self.tx
        t_bar = self.t_bar
        t_amp = self.t_amp
        t_til = self.t_til
        rA = np.sqrt(3) * self.a
        try:
            t_freq = 2 * pi * self.p / M / rA
        except:
            t_freq = 1

        delta_nnn = HamiltonianTilt.generate_nnn_const(N, M, tp, tx)
        delta_nn = HamiltonianTilt.generate_nn_vary_ttilda(N, M, t_bar, t_til, t_amp, t_freq, a=self.a)
        ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a=self.a)

        kx = np.arange(-10, 10, 0.1)
        ky = np.arange(-10, 10, 0.1)

        en = np.zeros((kx.size, ky.size))

        for i, kx_i in enumerate(kx):
            for j, ky_j in enumerate(ky):
                ham.construct_ham_inK(np.array([kx_i, ky_j]))
                eigvals = scla.eigh(ham.H, eigvals_only=True, subset_by_index=(0, 0))
                en[i, j] = eigvals

        self.pro = Process(target=self.plot_2d_groundenergy, args=([en, kx, ky]))
        self.pro.start()
        self.is_plot_showing = True

    def run(self):
        # Redefinitions
        stdscr = self.stdscr
        welcome_lines = self.welcome_lines
        show_lines = self.show_lines
        pad_m = cs.newpad(welcome_lines, 100)

        # Print welcome
        pad_m.addstr(0, 0, "------------------------------------------")
        pad_m.addstr(1, 0, "-    Plot spectrum - tilted Weyl cones ---")
        pad_m.addstr(2, 0, "------------------------------------------")
        pad_m.refresh(0, 0, 0, 0, welcome_lines, 100)
        
        # Take input
        pad_i = cs.newpad(1000, 100)
        ccom = ""
        while True:
            y, x = pad_i.getyx()
            pad_i.addch(y, x, ">")
            pad_i.move(y, x+1)
            self.refresh_and_scroll(pad_i, y)
            cc = 0
            # Get line
            while True:
                s = pad_i.getch()
                if s == 10:
                    break
                ccom += chr(s)
                pad_i.addch(y, x+1+cc, s)
                cc += 1
                pad_i.move(y, x+1+cc)
                self.refresh_and_scroll(pad_i, y)

            # Make command
            self.make_command(ccom, pad_i, y)
            if ccom == "quit":
                break

            ccom = ""
            self.refresh_and_scroll(pad_i, y)            

            pad_i.move(y+2, x)

    def make_command(self, com, pad_i, y):
        
        if com == "quit":
            pad_i.addstr(y+1, 0, "Quit program")
        elif com == "draw":
            self.draw_kpath()
            pad_i.addstr(y+1, 0, "Drawing plot")
        elif com == "drawan":
            self.draw_kpath_an()
            pad_i.addstr(y+1, 0, "Drawing plot")
        elif com == "drawgauge":
            self.draw_kpath_gauge()
        elif com == "drawdos":
            self.draw_dos()
        elif com == "drawgs":
            self.draw_2d_en_level()
        elif com == "drawbzone":
            self.draw_bzone_eq()
        elif com == "drawfold":
            self.draw_kpath_folded()
        elif com == "scalebzone":
            self.scale_bzone = not self.scale_bzone
        elif com[:2] == "N=":
            self.N = int(com[2:])
            pad_i.addstr(y+1, 0, "N changed to " + str(self.N))
        elif com[:2] == "M=":
            self.M = int(com[2:])
            pad_i.addstr(y+1, 0, "M changed to " + str(self.M))
        elif com[:2] == "t=":
            self.t = float(com[2:])
            pad_i.addstr(y+1, 0, "t changed to " + str(self.t))
        elif com[:2] == "a=":
            self.a = float(com[2:])
            pad_i.addstr(y+1, 0, "a changed to " + str(self.a))
        elif com[:6] == "t_bar=":
            self.t_bar = float(com[6:])
            pad_i.addstr(y+1, 0, "t_bar changed to " + str(self.t_bar))
        elif com[:6] == "t_amp=":
            self.t_amp = float(com[6:])
            pad_i.addstr(y+1, 0, "t_amp changed to " + str(self.t_amp))
        elif com[:6] == "t_til=":
            self.t_til = float(com[6:])
            pad_i.addstr(y+1, 0, "t_til changed to " + str(self.t_til))
        elif com[:3] == "tp=":
            self.tp = float(com[3:])
            pad_i.addstr(y+1, 0, "tp changed to " + str(self.tp))
        elif com[:3] == "tx=":
            self.tx = float(com[3:])
            pad_i.addstr(y+1, 0, "tx changed to " + str(self.tx))
        elif com[:2] == "p=":
            self.p = int(com[2:])
            pad_i.addstr(y+1, 0, "p changed to " + str(self.p))
        elif com[:6] == "kpath=":
            self.kpath = str(com[6:])
            pad_i.addstr(y+1, 0, "kpath changed to " + str(self.kpath))
        elif com[:4] == "kno=":
            self.no_k = int(com[4:])
            pad_i.addstr(y+1, 0, "no_k changed to " + str(self.no_k))
        elif com[:5] == "zoom=":
            self.zoomin = float(com[5:])
            pad_i.addstr(y+1, 0, "zoomin changed to " + str(self.zoomin))

        else:
            pad_i.addstr(y+1, 0, "Command not found!")

    def refresh_and_scroll(self, pad_i, y):
        show_lines = self.show_lines
        welcome_lines = self.welcome_lines
        if y > show_lines - welcome_lines:
            pad_i.refresh(y - show_lines + welcome_lines, 0, welcome_lines+1, 0, show_lines+10, 100)
        else:
            pad_i.refresh(0, 0, welcome_lines+1, 0, show_lines+10, 100)



def main(stdscr):

    uiST = UISpectrumTilted2(stdscr)
    uiST.run()


if __name__ == "__main__":
    wrapper(main)
    
    