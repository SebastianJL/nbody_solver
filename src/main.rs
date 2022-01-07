use csv;
use serde::Deserialize;
use std::path::Path;

type Real = f64;
use std::f64 as real;

// ID, Masses, x, y, z, Vx, Vy, Vz, softening, potential
#[derive(Debug, Deserialize)]
struct Particle {
    id: String,
    m: String,
    x: Real,
    y: Real,
    z: Real,
    vx: Real,
    vy: Real,
    vz: Real,
    eps: Real,
    pot: Real,
}

fn read_csv_file(p: &Path) -> csv::Result<Vec<Particle>> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_path(p)?;
    let mut particles: Vec<Particle> = vec![];
    for result in reader.deserialize() {
        let p: Particle = result?;
        particles.push(p)
    }
    Ok(particles)
}

fn main() {
    let path = Path::new("data/data.txt");
    let particles = read_csv_file(path).expect("Error reading file.");
}
