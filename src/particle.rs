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
    eps: Real,
    pot: Real,
}

impl Particle {
    fn dist2(&self, rhs: &Particle) -> Real {
        let (rx, ry, rz) = (self.x - rhs.x, self.y - rhs.y, self.z - rhs.z);
        rx * rx + ry * ry + rz * rz
    }

    pub fn force(&self, rhs: &Particle) -> Real {
        let r2 = self.dist2(rhs);
        self.m * rhs.m / r2
    }
}
