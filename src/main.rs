use csv;
use csv::Writer;
use particle::Particle;
use std::path::Path;

mod io;
mod particle;

type Real = f32;

fn calculate_forces(particles: &Vec<Particle>) -> () {
    let N = particles.len();

    for (i, p1) in particles.iter().enumerate() {
        let mut forces: Vec<Real> = vec![];
        for (j, p2) in particles.iter().enumerate() {
            if i < j {
                let f = p1.force(p2);
                forces.push(f);
                // println!("({:?}, {:?}): {:?}", i, j, p1.force(p2));
            }
        }
    }
}

fn main() {
    let path = Path::new("data/data.txt");
    let particles = io::read_csv_file(path).expect("Error reading file.");
    println! {"{:?}", particles[0]};
    // calculate_forces(&particles);
}
