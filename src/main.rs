use std::path::Path;

mod io;
mod math;
mod particle;
mod vector3d;

type Real = f32;

fn main() {
    let in_file = Path::new("data/data.txt");
    let out_file = Path::new("output/accelerations.dat");
    let mut particles = io::read_csv_file(in_file).expect("Error reading file.");
    let t0 = std::time::Instant::now();
    let softening: Real = 0.;
    let eps2 = softening.powi(2);
    math::calculate_accelerations(&mut particles, eps2);
    let dt1 = t0.elapsed();
    io::write_particles(out_file, &particles).expect("Error writing file.");
    let dt2 = t0.elapsed() - dt1;

    println!("{:?}", dt1);
    println!("{:?}", dt2);
}
