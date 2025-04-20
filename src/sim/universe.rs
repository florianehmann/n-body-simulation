use crate::sim::particle::Particle;

pub struct Universe<const D: usize> {
    pub particles: Vec<Particle<D>>,
}

impl<const D: usize> Universe<D> {
    pub fn new(particles: Vec<Particle<D>>) -> Self {
        Self { particles }
    }

    pub fn step(&mut self) {
        // reset forces
        for particle in &mut self.particles {
            particle.force *= 0.0;
        }

        // accumulate forces
        for i in 0..self.particles.len() {
            for j in i + 1..self.particles.len() {
                let (left, right) = self.particles.split_at_mut(j);
                let particle1 = &mut left[i];
                let particle2 = &mut right[0];

                let r_vec = particle1.vector_to(particle2);
                let r = r_vec.lp_norm(2);
                let f12 = 1000.0 / r.powf(2.0);
                let f12_vec = f12 * r_vec / r;

                particle1.force += f12_vec;
                particle2.force -= f12_vec;
            }
        }

        // update velocities and positions
        for particle in &mut self.particles {
            particle.vel += particle.force;
            particle.pos += particle.vel;
        }
    }
}
