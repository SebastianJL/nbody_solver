# N body project

This project was originally done for the course "Computational Astrophysics".
If you want the version of the project at the point in time the presentation was held
checkout the commit 954b5f6f92d57f208fceffca1a3bb8d4922d586d.

## Data format
- data.txt
  - No header
  - Data in columns\
    ID, Masses, x, y, z, Vx, Vy, Vz, softening, potential
  - Separator is tab.
- data_small_1001.txt
  - Same as data.txt but with less lines.
- data.ascii
  - No header
  - Data in single column of length 8*N, N=number of particles.
  - x[i]
    y[i]
    z[i]
    Vx[i]
    Vy[i]
    Vz[i]
    softening[i]
    potential[i]
- The positions are given relative to the galaxies center.
