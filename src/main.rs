#[macro_use]
extern crate log;

use std::fs::OpenOptions;
use std::path::Path;

mod io;
mod math;
mod particle;
mod vector3d;

type Real = f32;

fn setup_logging() {
    use simplelog::*;
    CombinedLogger::init(vec![
        TermLogger::new(
            LevelFilter::Info,
            Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        WriteLogger::new(
            LevelFilter::Info,
            Config::default(),
            OpenOptions::new()
                .append(true)
                .create(true)
                .open("run.log")
                .unwrap(),
        ),
    ])
    .unwrap();
}

fn main() {
    setup_logging();

    let r_mean: Real = 0.02710602183160793; // Mean inter-particle separation for 50'010. See fit_mass_distribution.py
    let eps = 0.1 * r_mean;
    let eps2 = eps * eps;

    info!("Running for eps={:?}...", eps);

    // Read data.
    let in_file = Path::new("data/data.txt");
    let t0 = std::time::Instant::now();
    let mut particles = io::read_csv_file(in_file).expect("Error reading file.");
    let dt0 = t0.elapsed();
    info!("reading {:?}", dt0);

    // Calculate forces.
    // Write data.
    let t1 = std::time::Instant::now();
    math::calculate_accelerations_direct(&mut particles, eps2);
    info!("acceleration calculation {:?}", dt1);

    let out_file = format!(
        "./output/acc_direct_rs_n={}_eps={}_.dat",
        particles.len(),
        eps
    );
    let t2 = std::time::Instant::now();

    let dt1 = t1.elapsed();
    io::write_particles(out_file, &particles).expect("Error writing file.");
    let dt2 = t2.elapsed();
    info!("writing {:?}", dt2);
}
