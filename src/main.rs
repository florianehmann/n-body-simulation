use macroquad::prelude::*;
use n_body_simulation::sim::{particle::Particle, universe::Universe};
use nalgebra::{SVector, vector};

#[macroquad::main("N-Body")]
async fn main() {
    let (w, h) = (screen_width(), screen_height());
    let r = 5.0;

    let mut universe = Universe::new(vec![
        Particle::new(vector![0.25 * w, 0.5 * h], vector![0.0, 3.0]),
        Particle::new(vector![0.75 * w, 0.5 * h], vector![0.0, -1.0]),
        Particle::new(vector![0.5 * w, 0.75 * h], vector![0.0, 0.5]),
    ]);

    loop {
        clear_background(BLACK);

        // draw dots
        for particle in &universe.particles {
            draw_circle(particle.pos.x, particle.pos.y, r, WHITE);
        }

        universe.step();

        next_frame().await
    }
}
