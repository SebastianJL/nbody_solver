use std::path::Path;

mod io;
mod math;
mod particle;
mod vector3d;

type Real = f32;

fn calculate_accelerations(eps: Real) {
    let eps2 = eps * eps;

    println!("Running for eps={:?}...", eps);
    // Read data.
    let in_file = Path::new("data/data_small_1001.txt");
    let t0 = std::time::Instant::now();
    let mut particles = io::read_csv_file(in_file).expect("Error reading file.");
    let dt0 = t0.elapsed();

    // Calculate forces.
    let t1 = std::time::Instant::now();
    math::calculate_accelerations(&mut particles, eps2);
    let dt1 = t1.elapsed();

    // Write data.
    let out_file = format!(
        "./output/accelerations_n={}_eps={}_.dat",
        particles.len(),
        eps
    );
    let t2 = std::time::Instant::now();
    io::write_particles(out_file, &particles).expect("Error writing file.");
    let dt2 = t2.elapsed();

    // Print timings.
    println!("reading {:?}", dt0);
    println!("acceleration calculation {:?}", dt1);
    println!("writing {:?}", dt2);
}

fn main() {
    let r_mean: Real = 0.02710602183160793; // Mean interparticle separation.
    let eps_values = [0., 0.1 * r_mean, 0.5 * r_mean, r_mean];
    for eps in eps_values {
        calculate_accelerations(eps);
    }
}
