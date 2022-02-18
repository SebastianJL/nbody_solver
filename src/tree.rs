use crate::particle::Particle;
use crate::vector3d::Vector3D;
use crate::Real;
use itertools::izip;

pub struct OctTreeNode {
    min: Vector3D,
    max: Vector3D,
    size: Real,
    com: Vector3D,
    monopole: Real,
    children: Vec<OctTreeNode>,
}

impl OctTreeNode {
    pub fn build(particles: &Vec<&Particle>, min: Vector3D, max: Vector3D) -> Self {
        let com = center_of_mass(particles);
        let mut children = vec![];
        let midpoint = Vector3D(
            (min.0 + max.0) / 2.,
            (min.1 + max.1) / 2.,
            (min.2 + max.2) / 2.,
        );

        // Assign particles to each of the 8 octants.
        const init: Vec<&Particle> = vec![];
        let mut children_particles: [Vec<&Particle>; 8] = [init; 8];
        for p in particles {
            let x_cond = (p.x > midpoint.0) as usize;
            let y_cond = (p.y > midpoint.1) as usize;
            let z_cond = (p.z > midpoint.2) as usize;
            let particle_index = x_cond * 4 + y_cond * 2 + z_cond;
            children_particles[particle_index].push(p);
        }

        // Todo: Build children.
        let children_min = vec![
            Vector3D(min.0, min.1, min.2),
            Vector3D(min.0, min.1, midpoint.2),
            Vector3D(min.0, midpoint.1, min.2),
            Vector3D(min.0, midpoint.1, midpoint.2),
            Vector3D(midpoint.0, min.1, min.2),
            Vector3D(midpoint.0, min.1, midpoint.2),
            Vector3D(midpoint.0, midpoint.1, min.2),
            Vector3D(midpoint.0, midpoint.1, midpoint.2),
        ];
        let children_max = vec![
            Vector3D(midpoint.0, midpoint.1, midpoint.2),
            Vector3D(midpoint.0, midpoint.1, max.2),
            Vector3D(midpoint.0, max.1, midpoint.2),
            Vector3D(midpoint.0, max.1, max.2),
            Vector3D(max.0, midpoint.1, midpoint.2),
            Vector3D(max.0, midpoint.1, max.2),
            Vector3D(max.0, max.1, midpoint.2),
            Vector3D(max.0, max.1, max.2),
        ];
        for (child_particles, child_min, child_max) in
            izip!(children_particles, children_min, children_max)
        {
            if child_particles.is_empty() {
                continue;
            }

            let child = OctTreeNode::build(&child_particles, child_min, child_max);
            children.push(child);
        }

        // Build self.
        Self {
            min,
            max,
            // Todo: Implement size.
            size: 0.,
            com,
            monopole: monopole(particles),
            children,
        }
    }
}

fn center_of_mass(particles: &[&Particle]) -> Vector3D {
    Vector3D(
        particles.iter().fold(0., |sum, p| sum + p.m * p.x),
        particles.iter().fold(0., |sum, p| sum + p.m * p.y),
        particles.iter().fold(0., |sum, p| sum + p.m * p.z),
    )
}

// fn node_size(particles: &Vec<Particle>, com: Real) -> Real {
//     particles.iter().map(0., |p| 0.)
// }

fn monopole(particles: &[&Particle]) -> Real {
    particles.iter().fold(0., |sum, p| sum + p.m)
}
