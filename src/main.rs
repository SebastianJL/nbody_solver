use std::path::Path;

mod io;
mod math;
mod particle;
mod vector3d;

type Real = f32;

fn main() {
    let eps: Real = 0.5; // Softening
    let eps2 = eps * eps;

    // Read data.
    let in_file = Path::new("data/data.txt");
    let t0 = std::time::Instant::now();
    let mut particles = io::read_csv_file(in_file).expect("Error reading file.");
    let dt0 = t0.elapsed();

    // Calculate forces.
    let t1 = std::time::Instant::now();
    math::calculate_accelerations(&mut particles, eps2);
    let dt1 = t0.elapsed();

    // Write data.
    let out_file = format!(
        "./output/accelerations_n={}_eps={}_.dat",
        particles.len(),
        eps
    );
    let t2 = std::time::Instant::now();
    io::write_particles(out_file, &particles).expect("Error writing file.");
    let dt2 = t1.elapsed();

    // Print timings.
    println!("{:?}", dt0);
    println!("{:?}", dt1);
    println!("{:?}", dt2);
}
