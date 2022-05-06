# draw_cross_section.py

from email.base64mime import header_length
from matplotlib.collections import PathCollection
from class_show_curser import UISpectrumTilted2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import pi, sqrt
from create_hamiltonian import HamiltonianTilt
import scipy.linalg as scla
from matplotlib.widgets import Slider

def draw_kpath(M=2):
    ui_spectrum = UISpectrumTilted2()
    ui_spectrum.M = M
    k, k_folded_list, b2, b1 = ui_spectrum.return_kpath_wignercell()

    kx = [k[i][0] for i in range(len(k))]
    ky = [k[i][1] for i in range(len(k))]
    kfx = [k_folded_list[i][0] for i in range(len(k))]
    kfy = [k_folded_list[i][1] for i in range(len(k))]

    fig, ax = plt.subplots(1, 2)

    pol_xy = np.array([np.array([0,0]), b1, b1 + b2, b2])
    p = Polygon(pol_xy)
    p2 = Polygon(pol_xy)
    ax[0].add_patch(p)
    ax[1].add_patch(p2)

    xrange = b1[0]
    yrange = b2[1] - b1[1]

    # draw lines
    for i in range(1, M):
        x = b1[0] / M * i
        y1 = b1[1] / M * i
        y2 = b1[1] / M * i + b2[1]
        ax[0].plot([x, x], [y1, y2], linewidth=2, c='grey')
        ax[1].plot([x, x], [y1, y2], linewidth=2, c='grey')

    # draw arrows

    ax[0].arrow(0, 0, b1[0], b1[1], head_width=0.1, head_length=0.1, length_includes_head=True)
    ax[0].arrow(0, 0, b2[0], b2[1], head_width=0.1, head_length=0.1, length_includes_head=True)

    ax[1].arrow(0, 0, b1[0], b1[1], head_width=0.1, head_length=0.1, length_includes_head=True)
    ax[1].arrow(0, 0, b2[0], b2[1], head_width=0.1, head_length=0.1, length_includes_head=True)

    ax[0].set_xlim([-0.1 * xrange, b1[0] + 0.1 * xrange])
    ax[0].set_ylim([b1[1] - 0.1 * yrange, b2[1] + 0.1 * yrange])
    ax[1].set_xlim([-0.1 * xrange, b1[0] + 0.1 * xrange])
    ax[1].set_ylim([b1[1] - 0.1 * yrange, b2[1] + 0.1 * yrange])
    ax[0].plot(kx, ky, c='r', marker='*', ls='')
    ax[1].plot(kfx, kfy, c='b', marker='*', ls='')
    ax[0].set_xlabel(r"$k_x$")
    ax[0].set_ylabel(r"$k_y$")
    ax[1].set_xlabel(r"$k_x$")
    ax[1].set_ylabel(r"$k_y$")
    ax[0].set_title("Original path")
    ax[1].set_title("Path in folded part")
    plt.suptitle("Path between high symmetry points")

    #plt.show()

def return_eigenvalues_from_wigner(ui_spec):
    N = ui_spec.N
    M = ui_spec.M
    t = ui_spec.t
    tp = ui_spec.tp
    tx = ui_spec.tx
    t_bar = ui_spec.t_bar
    t_amp = ui_spec.t_amp
    t_til = ui_spec.t_til
    rA = np.sqrt(3) * ui_spec.a
    try:
        t_freq = 2 * pi * ui_spec.p / M / rA
    except:
        t_freq = 0
    delta_nnn = HamiltonianTilt.generate_nnn_const(N, M, tp, tx)
    delta_nn = HamiltonianTilt.generate_nn_vary_ttilda(N, M, t_bar, t_til, t_amp, t_freq, ui_spec.a)
    ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a=ui_spec.a)

    k_res = 100
    k1 = -ham.k1
    k2 = ham.k2

    en = np.zeros((k_res, k_res, 2 * M * N))

    k1f = k1 / N
    k2f = k2 / M

    X = np.array([[(k1[0] * i + k2[0] * j)/ k_res for j in range(k_res)] for i in range(k_res)])
    Y = np.array([[(k1[1] * i + k2[1] * j)/ k_res for j in range(k_res)] for i in range(k_res)])

    for i in range(k_res):
        for j in range(k_res):
            ham.construct_ham_inK(ui_spec.project_to_smaller_wigner_cell(k1f, k2f, i / k_res * k1 + j / k_res * k2))
            eigvals = scla.eigh(ham.H, eigvals_only=True)
            en[i, j, :] = eigvals

    return X, Y, en
            

def return_cross(t, width, height, b1, b2, result='X'):
    pos1 = b2 + (1-width)/2 * (-b2 + b1)
    pos2 = b2 + (1+width)/2 * (-b2 + b1)

    vec = -b2 + b1
    orth = np.array([vec[1], -vec[0]])

    X, Z = np.meshgrid(np.linspace(pos1[0] + t * orth[0], pos2[0] + t * orth[0]),
                        np.linspace(-height/2, height/2, 2))

    lenaorth = np.linalg.norm(t * orth)
    Y = -sqrt(3) * X + b2[1] - lenaorth * 2 * sqrt(3) / 3

    if result=="X":
        return X
    if result=="Y":
        return Y
    if result=="Z":
        return Z

    

def plot_cross_section_prev():
    a = 1
    b = 4 * pi / 3 / a
    b2 = np.array([0, b])
    b1 = np.array([np.sqrt(3) / 2 * b, -b / 2])

    # Add hamiltonian
    ui_spectrum = UISpectrumTilted2()
    M = 1
    N = 1
    ui_spectrum.M = M

    KX, KY, EN = return_eigenvalues_from_wigner(ui_spectrum)

    # size of patch
    width = 0.8
    height = 1

    pos1 = b2 + (1-width)/2 * (-b2 + b1)
    pos2 = b2 + (1+width)/2 * (-b2 + b1)

    X, Z = np.meshgrid(np.linspace(pos1[0], pos2[0], 2), np.linspace(-height/2, height/2, 2))

    Y = -sqrt(3) * X + b2[1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z)
    # plt.subplots_adjust(left=0.25, bottom=0.25)
    
    # # slider
    # axt = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    # t_slider = Slider(ax = axt, label="t", valmin=-0.1, valmax=0.1, valinit=0)

    # def update(val):
    #     surf.set_ydata(return_cross(t_slider.val, width, height, b1, b2, result="Y"))
    #     surf.set_xdata(return_cross(t_slider.val, width, height, b1, b2, result="X"))
    
    for i in range(2 * N * M):
        ax.plot_surface(KX, KY, EN[:, :, i], color='k')
    ax.set_zlim(-0.6 * height, 0.6 * height)

    #t_slider.on_changed(update)
    #plt.show()

def draw_en_from_pos1pos2(pos1, pos2, ui_spec, k_points):
    N = ui_spec.N
    M = ui_spec.M
    t = ui_spec.t
    tp = ui_spec.tp
    tx = ui_spec.tx
    t_bar = ui_spec.t_bar
    t_amp = ui_spec.t_amp
    t_til = ui_spec.t_til
    rA = np.sqrt(3) * ui_spec.a
    try:
        t_freq = 2 * pi * ui_spec.p / M / rA
    except:
        t_freq = 0
    delta_nnn = HamiltonianTilt.generate_nnn_const(N, M, tp, tx)
    delta_nn = HamiltonianTilt.generate_nn_vary_ttilda(N, M, t_bar, t_til, t_amp, t_freq, ui_spec.a)
    ham = HamiltonianTilt(N, M, t, delta_nn, delta_nnn, a=ui_spec.a)

    k1 = -ham.k1
    k2 = ham.k2

    k_res = k_points

    en = np.zeros((k_res+1, 2 * M * N))

    k1f = k1 / N
    k2f = k2 / M

    

    k = [pos1 + i/k_res * (-pos1 + pos2) for i in range(k_res+1)]
    for i in range(k_res+1):
        ham.construct_ham_inK(ui_spec.project_to_smaller_wigner_cell(k1f, k2f, k[i]))
        eigvals = scla.eigh(ham.H, eigvals_only=True)
        en[i, :] = eigvals

    return en, k


def plot_cross_section(M, width, height, k_points, close_on_oneWeyl):
    a = 1
    b = 4 * pi / 3 / a
    b2 = np.array([0, b])
    b1 = np.array([np.sqrt(3) / 2 * b, -b / 2])

    # Add hamiltonian
    ui_spectrum = UISpectrumTilted2()
    N = 1
    ui_spectrum.M = M

    if close_on_oneWeyl:
        pos1 = b2 + 1/3 * (-b2 + b1) - 1/3 * width * (-b2 + b1)
        pos2 = b2 + 1/3 * (-b2 + b1) + 1/3 * width * (-b2 + b1)
    else:
        pos1 = b2 + (1-width)/2 * (-b2 + b1)
        pos2 = b2 + (1+width)/2 * (-b2 + b1)

    vec = -b2 + b1
    orth = np.array([vec[1], -vec[0]])

    en, k = draw_en_from_pos1pos2(pos1, pos2, ui_spectrum, k_points)
    en = np.array(en)

    param = np.linspace(0, 1, len(k))

    fig, ax = plt.subplots()
    lines = ax.plot(param, en, ms=2, c='k', ls="", marker='s')
    ax.set_ylim(-height/2, height/2)
    ax.set_ylabel("Energy")
    ax.set_xlabel(r"$k_{parameterised}$")
    ax.set_title("Energy spectrum")

    plt.subplots_adjust(bottom=0.3)

    # Make four sliders

    axc = plt.axes([0.2, 0.16, 0.7, 0.03])
    c_slider = Slider(ax=axc, label='offset', valmin=-0.02, valmax=0.02, valinit=0)

    axttilda = plt.axes([0.2, 0.12, 0.7, 0.03])
    ttilda_slider = Slider(ax=axttilda, label=r'$\tilde{t}$ (const tilt)', valmin=-1, valmax=1, valinit=0)

    axtamp = plt.axes([0.2, 0.08, 0.7, 0.03])
    tamp_slider = Slider(ax=axtamp, label=r'$t_{amp}$ (var tilt)', valmin=-1, valmax=1, valinit=0)

    axp = plt.axes([0.2, 0.04, 0.7, 0.03])
    p_slider = Slider(ax=axp, label='freq', valmin=0, valmax=M-1, valinit=0, valstep=1)

    def update(val):
        c = c_slider.val

        if close_on_oneWeyl:
            pos1 = b2 + 1/3 * (-b2 + b1) - 1/3 * width * (-b2 + b1) + c * orth
            pos2 = b2 + 1/3 * (-b2 + b1) + 1/3 * width * (-b2 + b1) + c * orth
        else:
            pos1 = b2 + (1-width)/2 * (-b2 + b1) + c * orth
            pos2 = b2 + (1+width)/2 * (-b2 + b1) + c * orth

        ui_spectrum.t_amp = tamp_slider.val
        ui_spectrum.t_til = ttilda_slider.val
        ui_spectrum.p = p_slider.val

        en, k = draw_en_from_pos1pos2(pos1, pos2, ui_spectrum, k_points)
        for i, l in enumerate(lines):
            l.set_ydata(en[:, i])
            
        

    c_slider.on_changed(update)
    ttilda_slider.on_changed(update)
    tamp_slider.on_changed(update)
    p_slider.on_changed(update)

    plt.show()



if __name__ == "__main__":
    ################################################################################################
    # Select case = "draw mapping" for plotting part of Brillouin zone with path in k-space mapped
    # to smaller Brillouin zone (elongated system).
    #
    # Select case = "draw cross section" for plotting spectrum near mapped Dirac cones.
    #
    # Select case = "draw cross section zoomed in" for plotting spectrum near one of the Dirac cone.
    case = "draw cross section"
    ################################################################################################
    # Parameters
    M = 4                           # Size of unit cell in chosen direction 
    width = 0.8                     # Range of k-points while plotting spectrum near Dirac cones
    height = 3                      # Range of energies included in the plotted spectrum
    k_points = 100                  # Number of plotted k-points
    ################################################################################################
    # Run code
    if case == "draw mapping":
        draw_kpath(M)
        plot_cross_section_prev()
        plt.show()
    elif case == "draw cross section":
        plot_cross_section(M, width, height, k_points, close_on_oneWeyl=False)
    elif case == "draw cross section zoomed in":
        plot_cross_section(M, width, height, k_points, close_on_oneWeyl=True)
    ################################################################################################