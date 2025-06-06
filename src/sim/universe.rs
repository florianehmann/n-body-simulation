//! Universe abstraction for n-body simulations.
//!
//! # Features
//! - Construction of universes from custom or randomly generated particle distributions
//! - Utilities for centering, velocity normalization, and angular momentum control
//! - Direct O(N²) and Barnes-Hut O(N log N) simulation steps
//! - Methods for computing physical properties such as center of mass and angular momentum
//!
//! # Example
//! ```rust
//! use n_body_simulation::sim::Universe;
//! use nalgebra::vector;
//!
//! // Create a universe with 100 particles in a Gaussian nebula
//! let universe = Universe::gaussian_nebula(
//!     100,
//!     vector![0.0, 0.0, 0.0],
//!     vector![1.0, 1.0, 1.0], None
//! );
//! ```
//!

use nalgebra::{SVector, vector};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Represents a single particle in the n-body simulation.
///
/// Each particle has a position, velocity, and force vector in 3D space.
#[derive(Clone, Debug)]
pub struct Particle {
    /// Position vector (x, y, z) of the particle.
    pub pos: SVector<f32, 3>,
    /// Velocity vector (vx, vy, vz) of the particle.
    pub vel: SVector<f32, 3>,
    /// Accumulated force vector (fx, fy, fz) acting on the particle.
    pub force: SVector<f32, 3>,
}

impl Particle {
    /// Creates a new `Particle` with the given position and optional velocity.
    ///
    /// If `vel` is `None`, the velocity is initialized to zero.
    ///
    /// # Parameters
    /// - `pos`: The initial position of the particle.
    /// - `vel`: Optional initial velocity of the particle.
    ///
    /// # Returns
    /// A new `Particle` instance.
    pub fn new(pos: SVector<f32, 3>, vel: Option<SVector<f32, 3>>) -> Self {
        Self {
            pos,
            vel: vel.unwrap_or_else(SVector::<f32, 3>::zeros),
            ..Default::default()
        }
    }

    /// Computes the vector from this particle to another particle.
    ///
    /// # Parameters
    /// - `other`: The other particle.
    ///
    /// # Returns
    /// The vector pointing from `self` to `other`.
    #[must_use]
    pub fn vector_to(&self, other: &Self) -> SVector<f32, 3> {
        other.pos - self.pos
    }
}

impl Default for Particle {
    /// Returns a particle at the origin with zero velocity and zero force.
    fn default() -> Self {
        Self {
            pos: SVector::<f32, 3>::zeros(),
            vel: SVector::<f32, 3>::zeros(),
            force: SVector::<f32, 3>::zeros(),
        }
    }
}

/// Manages a collection of particles and provides methods for initializing and analyzing an n-body
/// system.
#[derive(Clone)]
pub struct Universe {
    pub particles: Vec<Particle>,
}

impl Universe {
    /// Creates a new `Universe` from a vector of particles.
    ///
    /// # Parameters
    /// - `particles`: The particles to include in the universe.
    ///
    /// # Returns
    /// A new `Universe` containing the given particles.
    #[must_use]
    pub const fn new(particles: Vec<Particle>) -> Self {
        Self { particles }
    }

    /// Generates a universe of `n` particles distributed in 3D space according to a Gaussian
    /// (normal) distribution.
    ///
    /// Each particle's position is sampled independently for each axis using the provided mean `mu`
    /// and standard deviation `sigma`. Optionally, a random seed can be provided for
    /// reproducibility.
    ///
    /// # Parameters
    /// - `n`: Number of particles to generate.
    /// - `mu`: Mean position for the Gaussian distribution (per axis).
    /// - `sigma`: Standard deviation for the Gaussian distribution (per axis).
    /// - `seed`: Optional random seed for reproducibility.
    ///
    /// # Returns
    /// A new `Universe` containing the generated particles.
    ///
    /// # Panics
    /// Panics if the random distribution can't be generated. This is prevented by the function
    /// logic and should not happen.
    #[must_use]
    pub fn gaussian_nebula(
        n: usize,
        mu: SVector<f32, 3>,
        sigma: SVector<f32, 3>,
        seed: Option<u64>,
    ) -> Self {
        let mut rng = seed.map_or_else(rand::rngs::StdRng::from_os_rng, |seed| {
            let mut seed_array = [0u8; 32];
            seed_array[..8].copy_from_slice(&seed.to_le_bytes());
            rand::rngs::StdRng::from_seed(seed_array)
        });

        let normal = Normal::new(0.0, 1.0).expect("Sigma is hard-coded to be finite");
        let mut sample = SVector::<f32, 3>::zeros();

        let particles = (0..n)
            .map(|_| {
                for i in 0..3 {
                    sample[i] = normal.sample(&mut rng);
                }
                mu + sigma.component_mul(&sample)
            })
            .map(|pos| Particle::new(pos, None))
            .collect();

        Self::new(particles)
    }

    /// Computes the center of mass of all particles in the universe.
    ///
    /// # Returns
    /// The average position of all particles as a 3D vector.
    #[must_use]
    pub fn center_of_mass(&self) -> SVector<f32, 3> {
        #[allow(clippy::cast_precision_loss)]
        let n = self.particles.len() as f32;
        let position_sum: SVector<f32, 3> = self.particles.iter().map(|p| p.pos).sum();
        position_sum / n
    }

    /// Returns a new universe with the center of mass shifted to the origin.
    ///
    /// This subtracts the center of mass from every particle's position.
    ///
    /// # Returns
    /// A new `Universe` with zeroed center of mass.
    #[must_use]
    pub fn zero_center_of_mass(mut self) -> Self {
        let n = self.particles.len();
        if n == 0 {
            return self;
        }

        let com = self.center_of_mass();
        self.particles.iter_mut().for_each(|p| p.pos -= com);
        self
    }

    /// Computes the total velocity (vector sum) of all particles in the universe.
    ///
    /// # Returns
    /// The sum of all particle velocities as a 3D vector.
    #[must_use]
    pub fn total_velocity(&self) -> SVector<f32, 3> {
        self.particles.iter().map(|p| p.vel).sum()
    }

    /// Returns a new universe with the total velocity set to zero.
    ///
    /// This subtracts the average velocity from every particle.
    ///
    /// # Returns
    /// A new `Universe` with zero total velocity.
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

    /// Adds a random velocity to each particle, sampled from a Gaussian distribution.
    ///
    /// Each velocity component is sampled independently using the provided mean `mu` and standard
    /// deviation `sigma`. Optionally, a random seed can be provided for reproducibility.
    ///
    /// # Parameters
    /// - `mu`: Mean velocity for the Gaussian distribution (per axis).
    /// - `sigma`: Standard deviation for the Gaussian distribution (per axis).
    /// - `seed`: Optional random seed for reproducibility.
    ///
    /// # Returns
    /// The updated `Universe` with modified particle velocities.
    ///
    /// # Panics
    /// Panics if the random distribution can't be generated. This is prevented by the function
    /// logic and should not happen.
    #[must_use]
    pub fn set_random_velocity(
        mut self,
        mu: SVector<f32, 3>,
        sigma: SVector<f32, 3>,
        seed: Option<u64>,
    ) -> Self {
        let mut rng = seed.map_or_else(rand::rngs::StdRng::from_os_rng, |seed| {
            let mut seed_array = [0u8; 32];
            seed_array[..8].copy_from_slice(&seed.to_le_bytes());
            rand::rngs::StdRng::from_seed(seed_array)
        });

        let normal = Normal::new(0.0, 1.0).expect("Sigma is hard-coded to be finite");
        let mut sample = SVector::<f32, 3>::zeros();
        self.particles.iter_mut().for_each(|p| {
            for i in 0..3 {
                sample[i] = normal.sample(&mut rng);
            }
            let dv = mu + sigma.component_mul(&sample);
            p.vel += dv;
        });

        self
    }

    /// Computes the angular momentum of a single particle in the xy plane, relative to a given
    /// center of mass.
    ///
    /// # Parameters
    /// - `particle`: The particle whose angular momentum is computed.
    /// - `com`: The center of mass to use as the origin.
    ///
    /// # Returns
    /// The angular momentum of the particle in the xy plane.
    fn angular_momentum_particle_xy(particle: &Particle, com: SVector<f32, 3>) -> f32 {
        let pos = particle.pos - com;
        #[allow(clippy::suboptimal_flops)]
        {
            pos[0] * particle.vel[1] - pos[1] * particle.vel[0]
        }
    }

    /// Returns the total angular momentum of the universe in the xy plane.
    ///
    /// # Returns
    /// Total angular momentum in the xy plane of the universe.
    #[must_use]
    pub fn total_angular_momentum_xy(&self) -> f32 {
        let com = self.center_of_mass();
        self.particles
            .iter()
            .map(|p| Self::angular_momentum_particle_xy(p, com))
            .sum::<f32>()
    }

    /// Sets a uniform angular velocity in the xy plane for all particles.
    ///
    /// # Arguments
    /// * `period` - The desired rotation period in simulation time.
    ///
    /// # Returns
    /// The updated system with modified particle velocities.
    ///
    /// # Note
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
                Particle::new(vector![-1.0, -2.0, 0.0], None),
                Particle::new(vector![2.0, 4.0, 0.0], None),
            ],
        };
        let com = universe.center_of_mass();
        assert_relative_eq!(com, vector![0.5, 1.0, 0.0]);
    }

    #[test]
    fn test_zero_center_of_mass() {
        let universe = Universe {
            particles: vec![
                Particle::new(vector![-1.0, -2.0, 0.0], None),
                Particle::new(vector![2.0, 4.0, 0.0], None),
            ],
        };
        let zeroed_universe = universe.zero_center_of_mass();
        let com = zeroed_universe.center_of_mass();
        assert_relative_eq!(com, vector![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_gaussian_nebula_particle_count() {
        let mu = vector![0.0, 0.0, 0.0];
        let sigma = vector![1.0, 1.0, 0.0];
        let n = 100;
        let universe = Universe::gaussian_nebula(n, mu, sigma, None);
        assert_eq!(universe.particles.len(), n);
    }
}
