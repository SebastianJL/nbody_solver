use crate::Particle;
use std::path::Path;

pub fn read_csv_file(p: &Path) -> csv::Result<Vec<Particle>> {
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
