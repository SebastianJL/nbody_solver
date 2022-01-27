use crate::io::write_accelerations;
use csv;
use particle::Particle;
use std::path::Path;
use std::time::Duration;
use vector3d::Vector3D;

mod io;
mod particle;
mod vector3d;

type Real = f32;

fn calculate_accelerations(particles: &mut Vec<Particle>) -> Vec<Vector3D> {
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
    accs
}

fn main() {
    let in_file = Path::new("data/data.txt");
    let out_file = Path::new("output/accelerations.dat");
    let mut particles = io::read_csv_file(in_file).expect("Error reading file.");
    let t0 = std::time::Instant::now();
    let accs = calculate_accelerations(&mut particles);
    let dt1 = t0.elapsed();
    write_accelerations(out_file, &accs).expect("Error writing file.");
    let dt2 = t0.elapsed() - dt1;

    println!("{:?}", dt1);
    println!("{:?}", dt2);
}
