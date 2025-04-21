use nalgebra::SVector;
use rand::rng;
use rand_distr::{Distribution, Normal};

use crate::sim::particle::Particle;

pub struct Universe<const D: usize> {
    pub particles: Vec<Particle<D>>,
}

impl<const D: usize> Universe<D> {
    pub fn new(particles: Vec<Particle<D>>) -> Self {
        Self { particles }
    }

    pub fn gaussian_nebula(n: usize, mu: SVector<f32, D>, sigma: SVector<f32, D>) -> Self {
        let mut rng = rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut sample = SVector::<f32, D>::zeros();
        let particles = (0..n)
            .map(|_| {
                for i in 0..D {
                    sample[i] = normal.sample(&mut rng);
                }
                mu + sigma.component_mul(&sample)
            })
            .map(|pos| Particle::new(pos, None))
            .collect();

        Self::new(particles)
    }

    pub fn center_of_mass(&self) -> SVector<f32, D> {
        self.particles.iter().map(|p| p.pos).sum()
    }

    pub fn zero_center_of_mass(mut self) -> Self {
        let n = self.particles.len();
        if n == 0 {
            return self;
        }

        let com = self.center_of_mass();
        self.particles
            .iter_mut()
            .for_each(|p| p.pos -= com / (n as f32));
        self
    }

    pub fn total_velocity(&self) -> SVector<f32, D> {
        self.particles.iter().map(|p| p.vel).sum()
    }

    pub fn zero_total_velocity(mut self) -> Self {
        let n = self.particles.len();
        if n == 0 {
            return self;
        }

        let total_vel = self.total_velocity();
        self.particles
            .iter_mut()
            .for_each(|p| p.vel -= total_vel / (n as f32));
        self
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
                let f12 = 100.0 / (r.powf(2.0) + (10.0 as f32).powf(2.0));
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

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::vector;

    use super::*;

    #[test]
    fn test_center_of_mass() {
        let universe = Universe {
            particles: vec![
                Particle::new(vector![-1.0, -2.0], None),
                Particle::new(vector![2.0, 4.0], None),
            ],
        };
        let com = universe.center_of_mass();
        assert_relative_eq!(com, vector![1.0, 2.0]);
    }

    #[test]
    fn test_zero_center_of_mass() {
        let universe = Universe {
            particles: vec![
                Particle::new(vector![-1.0, -2.0], None),
                Particle::new(vector![2.0, 4.0], None),
            ],
        };
        let zeroed_universe = universe.zero_center_of_mass();
        let com = zeroed_universe.center_of_mass();
        assert_relative_eq!(com, vector![0.0, 0.0]);
    }

    #[test]
    fn test_gaussian_nebula_particle_count() {
        let mu = vector![0.0, 0.0];
        let sigma = vector![1.0, 1.0];
        let n = 100;
        let universe = Universe::gaussian_nebula(n, mu, sigma);
        assert_eq!(universe.particles.len(), n);
    }
}
