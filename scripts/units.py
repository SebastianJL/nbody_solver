"""Let L, M and T be the natural units of length, mass and time of a spherical galaxy,
such that G = 1 = L**3 / (M*T**2)
Choose M = 10*M_solar, L = 1kpc, then it follows"""

from math import pi, sqrt

if __name__ == '__main__':

    G = 1
    pc = 206265  # au
    M_solar = (4 * pi ** 2 / G)  # au**3 / yr**2
    M = 10e10 * M_solar
    L = 1e3 * pc
    T = sqrt(L ** 3 / (M * G))  # yr
    V = L / T  # au / yr

    print(f'{M = : .0f} au**3 / yr**2')
    print(f'{L = : .0f} au')
    print(f'{T = : .0f} yr')
    print(f'{V = : .0f} au/yr')

