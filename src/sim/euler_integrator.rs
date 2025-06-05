use crate::sim::{Integrator, Universe};

/// Integrates forces using Euler's method with a fixed time step.
pub struct EulerIntegrator {
    pub dt: f32,
}

impl Integrator for EulerIntegrator {
    fn step(&mut self, universe: &mut Universe) {
        for particle in &mut universe.particles {
            particle.vel += self.dt * particle.force;
            particle.pos += self.dt * particle.vel;
        }
    }
}
