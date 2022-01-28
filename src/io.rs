use std::fs::File;
use std::path::Path;

use byteorder::{LittleEndian, WriteBytesExt};

use crate::particle::Particle;

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

pub fn write_particles<P: AsRef<Path>>(path: P, particles: &Vec<Particle>) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    for p in particles.iter() {
        file.write_f64::<LittleEndian>(p.m as f64)?;
        file.write_f64::<LittleEndian>(p.x as f64)?;
        file.write_f64::<LittleEndian>(p.y as f64)?;
        file.write_f64::<LittleEndian>(p.z as f64)?;
        file.write_f64::<LittleEndian>(p.ax as f64)?;
        file.write_f64::<LittleEndian>(p.ay as f64)?;
        file.write_f64::<LittleEndian>(p.az as f64)?;
    }
    Ok(())
}
