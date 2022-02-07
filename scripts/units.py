"""Let L, M and T be the natural units of length, mass and time of a spherical galaxy,
such that G = 1 = L**3 / (M*T**2)
Choose M = 10*M_solar, L = 1kpc, then it follows"""

from math import pi, sqrt

if __name__ == '__main__':

    G = 1
    pc = 206265  # au
    M_solar = (4 * pi ** 2 / G)  # au**3 / yr**2
    M_0 = 10e10 * M_solar
    L_0 = 1e3 * pc
    T_0 = sqrt(L_0 ** 3 / (M_0 * G))  # yr
    V_0 = L_0 / T_0  # au / yr

    print(f'{M_0 = : g} au**3 / yr**2')
    print(f'{L_0 = : g} au')
    print(f'{T_0 = : g} yr')
    print(f'{V_0 = : g} au/yr')

