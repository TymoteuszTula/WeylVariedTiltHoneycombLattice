# curses_program.py

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import curses as cs
from create_hamiltonian import HamiltonianTilt

# def main(stdscr):
#     # Clear screen
#     stdscr.clear()

#     # This raises ZeroDivisionError when i == 10.
#     for i in range(0, 10):
#         v = i-10
#         stdscr.addstr(i, 0, '10 divided by {} is {}'.format(v, 10/v))

#     stdscr.refresh()
#     stdscr.getkey()

# cs.wrapper(main)


N = 1
M = 1
t = 1
a = 1
t_bar = 0
t_amp = 0
p = 0
kpath = "GKMG"

def check_command(a_list):
    if a_list[:2] == "N=":
        N = int(a_list[2:])
    elif a_list[:2] == "M=":
        M = int(a_list[2:])
    elif a_list[:2] == "t=":
        t = float(a_list[2:])
    elif a_list[:2] == "a=":
        a = float(a_list[2:])
    elif a_list[:6] == "t_bar=":
        t_bar = float(a_list[6:])
    elif a_list[:6] == "t_amp=":
        t_amp = float(a_list[6:])
    elif a_list[:2] == "p=":
        p = int(a_list[2:])
    elif a_list[:6] == "kpath":
        kpath = str(a_list[6:])
    elif a_list == "par":
        pass
    else:
        x, _ = stdscr.getyx
        stdscr.addstr(x+1, 0, "Command not found")
    

stdscr = cs.initscr()
stdscr.addstr(0,0, "Draw energy spectrum for tilted lattice system.")
stdscr.addstr(1,0, "-----------------------------------------------")
stdscr.addstr(2,0, "Set params")
stdscr.move(3,0)

a_list = ""

while True:
    stdscr.refresh()
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
            check_command(a_list)

        a_list = ""
        continue

    a_list += chr(a)

    
    

    

cs.endwin()