use crate::{Particle, Vector3D};
use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
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

pub fn write_accelerations(path: &Path, vecs: &Vec<Vector3D>) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    for a in vecs.iter() {
        file.write_f64::<LittleEndian>(a.0 as f64)?;
        file.write_f64::<LittleEndian>(a.1 as f64)?;
        file.write_f64::<LittleEndian>(a.2 as f64)?;
    }
    Ok(())
}
