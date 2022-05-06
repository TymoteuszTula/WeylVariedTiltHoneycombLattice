# class_show.py

import numpy as np
from math import pi, inf
import matplotlib.pyplot as plt
import curses as cs
from create_hamiltonian import HamiltonianTilt
from multiprocessing import Process

class UISpectrumTilted:

    def __init__(self):
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
        self.no_k = 1000
        self.zoomin = 0.1
        self.stdscr = cs.initscr()
        self.stdscr = cs.newpad(1000, 100)
        cs.echo()
        self.stdscr.scrollok(True)
        self.stdscr.addstr(0,0, "Draw energy spectrum for tilted lattice system.")
        self.stdscr.addstr(1,0, "-----------------------------------------------")
        self.stdscr.addstr(2,0, "Set params")
        self.stdscr.move(3,0)
        self.is_plot_showing = False

    def plot_k_path(self, eig_values, ticks, labelticks, add_ticks):
        plt.figure(0)
        plt.xticks(ticks, labels=labelticks)
        #plt.vlines([0, 99, 199, 299], ymin=-100, ymax=100)
        for tick in ticks:
            plt.axvline(tick, c="k")
        if add_ticks:
            for i in range(20):
                pos = i * self.no_k / 20
                plt.axvline(pos, alpha=0.1, c="k")
        # plt.axvline(0, c='k')
        # plt.axvline(100, c='k')
        # plt.axvline(200, c='k')
        # plt.axvline(300, c='k')
        plt.plot(eig_values)
        plt.ylabel("Energy")
        plt.show()
        
    def show_params(self):
        stdscr = self.stdscr
        x, _ = stdscr.getyx()
        stdscr.addstr(x, 0, "N=" + str(self.N))
        stdscr.addstr(x+1, 0, "M=" + str(self.M))
        stdscr.addstr(x+2, 0, "t=" + str(self.t))
        stdscr.addstr(x+3, 0, "a=" + str(self.a))
        stdscr.addstr(x+4, 0, "t_bar=" + str(self.t_bar))
        stdscr.addstr(x+5, 0, "t_amp=" + str(self.t_amp))
        stdscr.addstr(x+6, 0, "tp=" + str(self.tp))
        stdscr.addstr(x+7, 0, "tx=" + str(self.tx))
        stdscr.addstr(x+8, 0, "p=" + str(self.p))
        stdscr.addstr(x+9, 0, "kpath=" + str(self.kpath))
        stdscr.move(x+10, 0)

    # def add_gauge(self):


    def close_figure(self):
        plt.close()

    def show_eigvectors(self, k, n):
        stdscr = self.stdscr
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
        delta_nn = HamiltonianTilt.generate_nn_vary_ttilda(N, M, t_bar, t_amp, t_freq)
        ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a=self.a)
        k1 = ham.k1
        b = 4 * pi / 3 / self.a
        k2 = np.array([-np.sqrt(3) / 2 * b, -b / 2])
        K_point = (k1 + k2) / 3

        ham.construct_ham_inK(K_point * (1 + k))
        _, v  = np.linalg.eigh(ham.H)
        x, y = stdscr.getyx()
        stdscr.addstr(x + 1, y + 1, str(v[:, n]))
        stdscr.move(x+2, 0)

    def get_k_path(self, ham):

        if self.kpath == "GKMG":
            k1 = ham.k1
            k2 = ham.k2
            kGK = [(k1 + k2) * i / 3 / self.no_k for i in range(self.no_k)]
            kKM = [(k1 + k2) / 3 + (-k1 / 3 + k2 / 6) * i / self.no_k for i in range(self.no_k)]
            kMG = [k2 / 2 - k2 / 2 * i / self.no_k for i in range(self.no_k)]
            k = kGK + kKM + kMG
            ticks = [0, self.no_k, 2 * self.no_k, 3 * self.no_k]
            label_ticks = [r"$\Gamma$", "K", "M", r"$\Gamma$"]
            add_ticks = False

        elif self.kpath == "GKKGY":
            b = 4 * pi / 3 / self.a
            k = [np.array([np.sqrt(3) * b / 3, 0]) * i / self.no_k for i in range(3 * self.no_k)]
            ticks = [0, self.no_k, 2 * self.no_k, 3 * self.no_k]
            label_ticks = [r"$\Gamma$", "K1", "K2", r"$\Gamma$"]
            add_ticks = False


        elif self.kpath == "GMG":
            k2 = ham.k2
            k = [k2 / 2 * i / self.no_k for i in range(2 * self.no_k)]
            ticks = [0, self.no_k, 2 * self.no_k]
            label_ticks = [r"$\Gamma$", "M", r"$\Gamma$"]
            add_ticks = False

        elif self.kpath == "GMSCALE":
            k1 = -ham.k1
            k2 = ham.k2 / self.M
            k3 = k1 + ham.k2
            k4 = (k1 + k3) / 2

            kGM1 = [k2/2 * i /self.no_k for i in range(self.no_k)]
            kM1K = [k2/2 + k4/2 * i/self.no_k for i in range(self.no_k)]
            kKM2 = [k2/2 + k4/2 - k2/2 * i/self.no_k for i in range(self.no_k)]
            kM2G = [k4/2 - k4/2 * i/self.no_k for i in range(self.no_k+1)]

            k = kGM1 + kM1K + kKM2 + kM2G
            ticks = [0, self.no_k, 2 * self.no_k, 3 * self.no_k, 4 * self.no_k]
            label_ticks = [r"$\Gamma$", r"$\bar{M_2}$", r"$\bar{K}$", r"$\bar{M_1}$", r"$\Gamma$"]
            add_ticks = False

        elif self.kpath == "GKBAR":
            k2 = ham.k2 / self.M
            k1 = -ham.k1
            k3 = k1 + ham.k2
            k4 = (k1 + k3) / 2
            k = [k2 + k4 * i /self.no_k for i in range(2 * self.no_k)]
            ticks = [0, self.no_k, 2 * self.no_k]
            label_ticks = [r"$\Gamma$", r"$\bar{K}$", r"$\Gamma$"]
            add_ticks = False

        elif self.kpath == "KZOOMIN":
            k1 = ham.k1
            b = 4 * pi / 3 / self.a
            k2 = np.array([-np.sqrt(3) / 2 * b, -b / 2])
            K_point = (k1 + k2) / 3
            zoom_in = 1 / 3 * self.zoomin
            k = [K_point + (i-self.no_k/2)/self.no_k * zoom_in * (k1 + k2) for i in range(self.no_k)]
            ticks = [0, self.no_k/2, self.no_k]
            label_ticks = [r"-{} $\Gamma$K".format(self.zoomin), r"K", r"{} $\Gamma$K".format(self.zoomin)]
            add_ticks = True

        return k, ticks, label_ticks, add_ticks

    def draw_kpath(self):
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
        delta_nn = HamiltonianTilt.generate_nn_vary_ttilda(N, M, t_bar, t_til, t_amp, t_freq, self.a)
        ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a=self.a)
        
        eig_values = []

        k, ticks, label_ticks, add_ticks = self.get_k_path(ham)
        
        for i in range(len(k)):
            ham.construct_ham_inK(k[i])
            eigvals = np.linalg.eigvals(ham.H)
            eigvals = np.real(np.sort(eigvals))
            eig_values.append(eigvals)

        self.eig_values = eig_values

        if self.is_plot_showing:
            self.pro.kill()

        self.pro = Process(target=self.plot_k_path, args=([eig_values, ticks, label_ticks, add_ticks]))
        self.pro.start()
        self.is_plot_showing = True

    def draw_kpath_an(self):
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
        delta_nn = HamiltonianTilt.generate_nn_vary_ttilda(N, M, t_bar, t_til, t_amp, t_freq, self.a)
        ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a=self.a)
        
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

    def check_command(self, a_list):
        if a_list[:2] == "N=":
            self.N = int(a_list[2:])
        elif a_list[:2] == "M=":
            self.M = int(a_list[2:])
        elif a_list[:2] == "t=":
            self.t = float(a_list[2:])
        elif a_list[:2] == "a=":
            self.a = float(a_list[2:])
        elif a_list[:6] == "t_bar=":
            self.t_bar = float(a_list[6:])
        elif a_list[:6] == "t_amp=":
            self.t_amp = float(a_list[6:])
        elif a_list[:3] == "tp=":
            self.tp = float(a_list[3:])
        elif a_list[:3] == "tx=":
            self.tx = float(a_list[3:])
        elif a_list[:2] == "p=":
            self.p = int(a_list[2:])
        elif a_list[:6] == "kpath=":
            self.kpath = str(a_list[6:])
        elif a_list[:4] == "kno=":
            self.no_k = int(a_list[4:])
        elif a_list[:5] == "zoom=":
            self.zoomin = float(a_list[5:])
        elif a_list == "par":
            self.show_params()
        elif a_list == "draw":
            self.draw_kpath()
        elif a_list == "draw_an":
            self.draw_kpath_an()
        elif a_list[:4] == "eigv":
            a_split = a_list.split(" ")
            self.show_eigvectors(float(a_split[1]), int(a_split[2]))
        elif a_list == "close":
            self.close_figure()
        else:
            x, _ = self.stdscr.getyx()
            self.stdscr.addstr(x, 0, "Command not found")
            self.stdscr.move(x+1, 0)


    def run(self):
        stdscr = self.stdscr

        a_list = ""

        while True:
            padpos = 0
            stdscr.refresh(padpos + 2, 0, 0, 0, 40, 80)
            a = stdscr.getch()
            # x, y = stdscr.getyx()
            # stdscr.addstr(x+1, 0, str(a))
            # stdscr.move(x+2, 0)
            if a == 10:
                x, y = stdscr.getyx()
                stdscr.addstr(x+1, 0, a_list)
                stdscr.move(x+2, 0)

                if a_list == "quit":
                    break
                else:
                    self.check_command(a_list)

                a_list = ""
                padpos += 2
                continue

            a_list += chr(a)



if __name__ == "__main__":

    app = UISpectrumTilted()
    app.run()
    cs.endwin()