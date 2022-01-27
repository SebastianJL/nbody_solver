use crate::particle::Particle;
use crate::vector3d::Vector3D;

pub fn calculate_accelerations(particles: &mut Vec<Particle>) {
    let n = particles.len();
    let mut accs = vec![Vector3D(0., 0., 0.); n];

    for (i, (p1, acc)) in particles.iter().zip(accs.iter_mut()).enumerate() {
        for (j, p2) in particles.iter().enumerate() {
            if i != j {
                let a = p1.acc(p2, p1.eps);
                acc.0 += a.0;
                acc.1 += a.1;
                acc.2 += a.2;
            }
        }
    }
    for (p1, a) in particles.iter_mut().zip(accs.iter()) {
        p1.ax = a.0;
        p1.ay = a.1;
        p1.ay = a.2;
    }
}
