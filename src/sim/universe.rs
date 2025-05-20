use nalgebra::{SVector, vector};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use crate::sim::particle::Particle;

#[derive(Clone)]
pub struct Universe<const D: usize> {
    pub particles: Vec<Particle<D>>,
}

impl<const D: usize> Universe<D> {
    #[must_use]
    pub const fn new(particles: Vec<Particle<D>>) -> Self {
        Self { particles }
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn gaussian_nebula(
        n: usize,
        mu: SVector<f32, D>,
        sigma: SVector<f32, D>,
        seed: Option<u64>,
    ) -> Self {
        let mut rng = seed.map_or_else(rand::rngs::StdRng::from_os_rng, |seed| {
            let mut seed_array = [0u8; 32];
            seed_array[..8].copy_from_slice(&seed.to_le_bytes());
            rand::rngs::StdRng::from_seed(seed_array)
        });

        let normal = Normal::new(0.0, 1.0).expect("Sigma is hard-coded to be finite");
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

    #[must_use]
    pub fn center_of_mass(&self) -> SVector<f32, D> {
        self.particles.iter().map(|p| p.pos).sum()
    }

    #[must_use]
    pub fn zero_center_of_mass(mut self) -> Self {
        let n = self.particles.len();
        if n == 0 {
            return self;
        }

        let com = self.center_of_mass();
        #[allow(clippy::cast_precision_loss)]
        self.particles
            .iter_mut()
            .for_each(|p| p.pos -= com / (n as f32));
        self
    }

    #[must_use]
    pub fn total_velocity(&self) -> SVector<f32, D> {
        self.particles.iter().map(|p| p.vel).sum()
    }

    #[must_use]
    pub fn zero_total_velocity(mut self) -> Self {
        let n = self.particles.len();
        if n == 0 {
            return self;
        }

        let total_vel = self.total_velocity();
        #[allow(clippy::cast_precision_loss)]
        self.particles
            .iter_mut()
            .for_each(|p| p.vel -= total_vel / (n as f32));
        self
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn set_random_velocity(
        mut self,
        mu: SVector<f32, D>,
        sigma: SVector<f32, D>,
        seed: Option<u64>,
    ) -> Self {
        let mut rng = seed.map_or_else(rand::rngs::StdRng::from_os_rng, |seed| {
            let mut seed_array = [0u8; 32];
            seed_array[..8].copy_from_slice(&seed.to_le_bytes());
            rand::rngs::StdRng::from_seed(seed_array)
        });

        let normal = Normal::new(0.0, 1.0).expect("Sigma is hard-coded to be finite");
        let mut sample = SVector::<f32, D>::zeros();
        self.particles.iter_mut().for_each(|p| {
            for i in 0..D {
                sample[i] = normal.sample(&mut rng);
            }
            let dv = mu + sigma.component_mul(&sample);
            p.vel += dv;
        });

        self
    }

    fn angular_momentum_particle_xy(particle: &Particle<D>, com: SVector<f32, D>) -> f32 {
        let pos = particle.pos - com;
        #[allow(clippy::suboptimal_flops)]
        {
            pos[0] * particle.vel[1] - pos[1] * particle.vel[0]
        }
    }

    /// Returns the total angular momentum of the universe in the xy plane.
    ///
    /// # Returns
    ///
    /// Total angular momentum in the xy plane of the universe.
    #[must_use]
    pub fn total_angular_momentum_xy(&self) -> f32 {
        let com = self.center_of_mass();
        self.particles
            .iter()
            .map(|p| Self::angular_momentum_particle_xy(p, com))
            .sum::<f32>()
    }

    #[must_use]
    pub fn set_angular_momentum_xy_equally(mut self, angular_momentum: f32) -> Self {
        let n = self.particles.len();
        if n == 0 {
            return self;
        }

        let com = self.center_of_mass();
        #[allow(clippy::cast_precision_loss)]
        let target_l_avg = angular_momentum / (n as f32);
        self.particles.iter_mut().for_each(|p| {
            let current_l = Self::angular_momentum_particle_xy(p, com);
            let delta_l = target_l_avg - current_l;
            #[allow(clippy::suboptimal_flops)]
            let r = p.pos[0].powi(2) + p.pos[1].powi(2);
            let xy = p.pos[0] * p.pos[1];
            #[allow(clippy::suboptimal_flops)]
            {
                p.vel[0] = (-delta_l * p.pos[1] - p.vel[0] * xy + p.vel[1] * p.pos[1].powi(2)) / r;
                p.vel[1] = (delta_l * p.pos[0] - p.vel[1] * xy + p.vel[0] * p.pos[0].powi(2)) / r;
            }
        });

        self
    }

    /// Sets a uniform angular velocity in the xy plane for all particles.
    ///
    /// # Arguments
    ///
    /// * `period` - The desired rotation period in simulation time.
    ///
    /// # Returns
    ///
    /// The updated system with modified particle velocities.
    ///
    /// # Note
    ///
    /// This method does not account for center-of-mass offset or pre-existing
    /// angular momentum. It assumes the origin is the center of rotation and
    /// adds the rotational component to the current velocities.
    #[must_use]
    pub fn set_rotation_period(mut self, period: f32) -> Self {
        let target_angular_velocity = 2.0 * std::f32::consts::PI / period;
        self.particles.iter_mut().for_each(|p| {
            let vel = vector![-p.pos[1], p.pos[0]] * target_angular_velocity;
            p.vel[0] += vel[0];
            p.vel[1] += vel[1];
        });

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

                let r_vec = particle2.pos - particle1.pos;

                let r_squared = r_vec.norm_squared();
                let softened = r_squared + 0.01; // (r^2 + epsilon^2)
                let inv_r = 1.0 / softened.sqrt();

                let f12 = 1.0e-4 * inv_r * inv_r;
                let f12_vec = f12 * r_vec * inv_r;

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
        let universe = Universe::gaussian_nebula(n, mu, sigma, None);
        assert_eq!(universe.particles.len(), n);
    }

    #[test]
    fn test_set_angular_momentum_xy() {
        let mu = vector![0.0, 0.0];
        let sigma = vector![1.0, 1.0];
        let n = 100;
        let target_angular_momentum = 50.0;

        let universe = Universe::gaussian_nebula(n, mu, sigma, Some(254))
            .zero_center_of_mass()
            .set_angular_momentum_xy_equally(target_angular_momentum);

        let actual_angular_momentum = universe.total_angular_momentum_xy();
        assert_relative_eq!(
            actual_angular_momentum,
            target_angular_momentum,
            epsilon = 1.0
        );
    }
}
