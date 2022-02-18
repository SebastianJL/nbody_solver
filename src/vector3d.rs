use crate::Real;

#[derive(Clone)]
pub struct Vector3D(pub Real, pub Real, pub Real);

impl Vector3D {
    pub fn zero() -> Self {
        Vector3D(0., 0., 0.)
    }
}
