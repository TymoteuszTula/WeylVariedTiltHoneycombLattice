# draw_eigenvectors.py

import numpy as np
from class_show_curser import UISpectrumTilted2
import matplotlib.pyplot as plt
from math import pi, sqrt, floor
from cmath import exp
from create_hamiltonian import HamiltonianTilt
import scipy.linalg as scla
from matplotlib.widgets import Slider
from draw_cross_section import draw_en_from_pos1pos2

def draw_en_from_pos1pos2_with_evec(pos1, pos2, ui_spec, k_points):
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
    evcs = np.zeros((k_res+1, 2 * M * N, 2 * M * N), dtype=complex)

    k1f = k1 / N
    k2f = k2 / M

    

    k = [pos1 + i/k_res * (-pos1 + pos2) for i in range(k_res+1)]
    for i in range(k_res+1):
        ham.construct_ham_inK(ui_spec.project_to_smaller_wigner_cell(k1f, k2f, k[i]))
        eigvals, eigvecs = scla.eigh(ham.H, eigvals_only=False)
        en[i, :] = eigvals
        evcs[i, :, :] = eigvecs

    return en, evcs, k

def draw_background_lattice(ax, M, a=1):

    #line_updownx = np.arange(0, (M - 1/2) * a * sqrt(3), a * sqrt(3) / 2)
    line_updownx = np.linspace(0, (M-1) * sqrt(3) * a, 2 * M - 1)
    line_upy = np.array([a/2, a] * (M-1) + [a/2])
    line_downy = np.array([-a/2, -a] * (M-1) + [-a/2])

    alpha_honeycomb=0.5

    ax.plot(line_updownx, line_upy, c='k', alpha=alpha_honeycomb)
    ax.plot(line_updownx, line_downy, c='k', alpha=alpha_honeycomb)

    for i in range(M):
        ax.plot(2 * [i * sqrt(3) * a], [-a/2, a/2], c='k', alpha=alpha_honeycomb)

    #scatter_uudd_x = np.arange(sqrt(3) * a/2, (M-1) * sqrt(3) * a, sqrt(3) * a)
    #scatter_ud_x = np.arange(0, M * sqrt(3) * a, sqrt(3) * a)
    scatter_uudd_x = np.linspace(sqrt(3) * a/2, (M - 3/2) * sqrt(3) * a, M-1)
    scatter_ud_x = np.linspace(0, (M-1) * sqrt(3) * a, M)
    scatter_uu_y = np.array([a] * (M-1))
    scatter_u_y = np.array([a/2] * M)
    scatter_d_y = np.array([-a/2] * M)
    scatter_dd_y = np.array([-a] * (M-1))

    ax.scatter(scatter_uudd_x, scatter_uu_y, c='k', alpha=alpha_honeycomb)
    ax.scatter(scatter_ud_x, scatter_u_y, c='k')
    ax.scatter(scatter_ud_x, scatter_d_y, c='k')
    ax.scatter(scatter_uudd_x, scatter_dd_y, c='k', alpha=alpha_honeycomb)

def draw_eigenvecs(ax, M, evc, k, q, a=1, pquiv=None):

    EXP = np.array([exp(1j * np.dot(k, q[i])) for i in range(2 * M)])

    X = np.linspace(0, (M-1) * sqrt(3) * a, M)
    Y = np.zeros((X.size, ))

    A1 = np.array([evc[int(i * 2)] for i in range(M)], dtype=complex)
    B1 = np.array([evc[int(i * 2 + 1)] for i in range(M)], dtype=complex)

    angleA = np.arctan(np.imag(A1)/np.real(A1))
    angleB = np.arctan(np.imag(B1)/np.real(B1))

    relAngle = angleB - angleA

    abs = np.abs(A1) + np.abs(B1)

    U = abs * np.cos(relAngle)
    V = abs * np.sin(relAngle)

     # angles='uv', scale=1/M)
    if pquiv == None:
        quiv = ax.quiver(X, Y, U, V)
        return quiv
    else:
        pquiv.set_UVC(U, V)
    

def plot_cross_section_with_eigenvectors(M, width, height, k_points):
    a = 1
    b = 4 * pi / 3 / a
    b2 = np.array([0, b])
    b1 = np.array([np.sqrt(3) / 2 * b, -b / 2])

    # Add hamiltonian
    ui_spectrum = UISpectrumTilted2()
    N = 1
    ui_spectrum.M = M
    ui_spectrum.a = a

    pos1 = b2 + 1/3 * (-b2 + b1) - 1/3 * width * (-b2 + b1)
    pos2 = b2 + 1/3 * (-b2 + b1) + 1/3 * width * (-b2 + b1)

    vec = -b2 + b1
    orth = np.array([vec[1], -vec[0]])

    en, evcs, k = draw_en_from_pos1pos2_with_evec(pos1, pos2, ui_spectrum, k_points)
    en = np.array(en)

    q = [np.array([floor(i/2) * sqrt(3) * a, -a/2 + i%2 * a]) for i in range(2 * M)]

    param = np.linspace(0, 1, len(k))

    gs_kw = dict(width_ratios=[1, 4])
    fig, axs = plt.subplots(1, 2, gridspec_kw=gs_kw)
    lines = axs[0].plot(param, en, ms=2, c='k', ls="", marker='s')
    axs[0].set_ylim(-height/2, height/2)
    axs[0].set_ylabel("Energy")
    axs[0].set_xlabel(r"$k_{parameterised}$")
    axs[0].set_title("Energy spectrum")

    # Choose initial eigenstate
    k_in = int(len(k)/2)
    en_in = en[k_in, M]
    lines_cev = axs[0].plot(param[k_in], en_in, ms=2, c='r', ls="", marker='s')

    plt.subplots_adjust(bottom=0.4)

    # Draw background honeycomb lattice
    draw_background_lattice(axs[1], M)

    # Draw eigenvectors
    quiv = draw_eigenvecs(axs[1], M, evcs[k_in, :, M], k[k_in], q, a)

    # Make four sliders

    axc = plt.axes([0.2, 0.24, 0.7, 0.03])
    c_slider = Slider(ax=axc, label='offset', valmin=-0.02, valmax=0.02, valinit=0)

    axttilda = plt.axes([0.2, 0.2, 0.7, 0.03])
    ttilda_slider = Slider(ax=axttilda, label=r'$\tilde{t}$ (const tilt)', valmin=-1, valmax=1, valinit=0)

    axtamp = plt.axes([0.2, 0.16, 0.7, 0.03])
    tamp_slider = Slider(ax=axtamp, label=r'$t_{amp}$ (var tilt)', valmin=-1, valmax=1, valinit=0)

    axp = plt.axes([0.2, 0.12, 0.7, 0.03])
    p_slider = Slider(ax=axp, label='freq', valmin=0, valmax=M-1, valinit=0, valstep=1)

    axen_lev = plt.axes([0.2, 0.08, 0.7, 0.03])
    en_lev_slider = Slider(ax=axen_lev, label='E level', valmin=-10, valmax=10, valinit=0, valstep=1)

    axen_mv = plt.axes([0.2, 0.04, 0.7, 0.03])
    en_mv_slider = Slider(ax=axen_mv, label='E move', valmin=-10, valmax=10, valinit=0, valstep=1)

    def update(val):
        c = c_slider.val

        pos1 = b2 + 1/3 * (-b2 + b1) - 1/3 * width * (-b2 + b1) + c * orth
        pos2 = b2 + 1/3 * (-b2 + b1) + 1/3 * width * (-b2 + b1) + c * orth

        ui_spectrum.t_amp = tamp_slider.val
        ui_spectrum.t_til = ttilda_slider.val
        ui_spectrum.p = p_slider.val

        en, evcs, k = draw_en_from_pos1pos2_with_evec(pos1, pos2, ui_spectrum, k_points)
        for i, l in enumerate(lines):
            l.set_ydata(en[:, i])
        lines_cev[0].set_xdata(param[k_in + en_mv_slider.val])
        lines_cev[0].set_ydata(en[k_in + en_mv_slider.val, M + en_lev_slider.val])
        draw_eigenvecs(axs[1], M, evcs[k_in + en_mv_slider.val, :, M + en_lev_slider.val],
                            k[k_in+en_mv_slider.val], q, a, quiv)
            

    c_slider.on_changed(update)
    ttilda_slider.on_changed(update)
    tamp_slider.on_changed(update)
    p_slider.on_changed(update)
    en_lev_slider.on_changed(update)
    en_mv_slider.on_changed(update)

    plt.show()

if __name__ == "__main__":

    ################################################################################################
    # Parameters
    M = 4
    width = 0.3
    height = 3
    k_points = 100
    ################################################################################################
    plot_cross_section_with_eigenvectors(M, width, height, k_points)
