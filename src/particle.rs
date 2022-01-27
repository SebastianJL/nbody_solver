use serde::Deserialize;

use crate::Real;
use crate::vector3d::Vector3D;

// ID, Masses, x, y, z, Vx, Vy, Vz, softening, potential
#[derive(Debug, Deserialize)]
pub struct Particle {
    id: i32,
    pub m: Real,
    pub x: Real,
    pub y: Real,
    pub z: Real,
    pub vx: Real,
    pub vy: Real,
    pub vz: Real,
    #[serde(skip_deserializing)]
    pub ax: Real,
    #[serde(skip_deserializing)]
    pub ay: Real,
    #[serde(skip_deserializing)]
    pub az: Real,
    pub eps: Real,
    pot: Real,
}

impl Particle {
    /// Calculate the acceleration that particle 2 causes on particle 1.
    ///
    /// # Arguments:
    /// * `p2`: Particle 2.
    /// * `eps2`: Softening factor squared.
    ///
    pub fn acc(&self, p2: &Particle, eps2: Real) -> Vector3D {
        let (rx, ry, rz) = (p2.x - self.x, p2.y - self.y, p2.z - self.z);
        let r2 = rx * rx + ry * ry + rz * rz;
        let r2_eps = (r2 + eps2).powi(3).sqrt();
        Vector3D(p2.m * rx / r2_eps, p2.m * ry / r2_eps, p2.m * rz / r2_eps)
    }
}
