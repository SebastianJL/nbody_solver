use crate::particle::Particle;
use crate::vector3d::Vector3D;
use crate::Real;

///
///
/// # Arguments
///
/// * `particles`: List of all particles in the system. Particle accelerations are mutated in place.
/// * `eps2`: Square of the softening parameter.
///
/// returns: ()
///
/// # Examples
///
/// ```
///
/// ```
pub fn calculate_accelerations(particles: &mut Vec<Particle>, eps2: Real) {
    let n = particles.len();
    let mut accs = vec![Vector3D(0., 0., 0.); n];

    // Calculate accelerations.
    for (i, (p1, acc)) in particles.iter().zip(accs.iter_mut()).enumerate() {
        for (j, p2) in particles.iter().enumerate() {
            if i != j {
                let a = p1.acc(p2, eps2);
                acc.0 += a.0;
                acc.1 += a.1;
                acc.2 += a.2;
            }
        }
    }

    // Write accelerations back to particles.
    for (p1, a) in particles.iter_mut().zip(accs.iter()) {
        p1.ax = a.0;
        p1.ay = a.1;
        p1.ay = a.2;
    }
}
