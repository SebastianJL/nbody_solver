use crate::Real;
use serde::Deserialize;

// ID, Masses, x, y, z, Vx, Vy, Vz, softening, potential
#[derive(Debug, Deserialize)]
pub struct Particle {
    id: i32,
    m: Real,
    x: Real,
    y: Real,
    z: Real,
    vx: Real,
    vy: Real,
    vz: Real,
    #[serde(skip)]
    pub ax: Real,
    #[serde(skip)]
    pub ay: Real,
    #[serde(skip)]
    pub az: Real,
    eps: Real,
    pot: Real,
}

impl Particle {
    pub fn acc(&self, rhs: &Particle) -> (Real, Real, Real) {
        let (rx, ry, rz) = (rhs.x - self.x, rhs.y - self.y, rhs.z - self.z);
        let r2 = rx * rx + ry * ry + rz * rz;
        (rhs.m * rx / r2, rhs.m * ry / r2, rhs.m * rz / r2)
    }
}
