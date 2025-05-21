use nalgebra::{SVector, vector};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

#[derive(Clone)]
pub struct Universe {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    pub vx: Vec<f32>,
    pub vy: Vec<f32>,
    pub vz: Vec<f32>,
    pub fx: Vec<f32>,
    pub fy: Vec<f32>,
    pub fz: Vec<f32>,
}

impl Universe {
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
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

        let vx = vec![0.0; n];
        let vy = vec![0.0; n];
        let vz = vec![0.0; n];
        let fx = vec![0.0; n];
        let fy = vec![0.0; n];
        let fz = vec![0.0; n];
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        for _ in 0..n {
            for i in 0..3 {
                sample[i] = normal.sample(&mut rng);
            }
            let pos = mu + sigma.component_mul(&sample);
            x.push(pos[0]);
            y.push(pos[1]);
            z.push(pos[2]);
        }

        Self {
            x,
            y,
            z,
            vx,
            vy,
            vz,
            fx,
            fy,
            fz,
        }
    }

    #[must_use]
    pub fn center_of_mass(&self) -> SVector<f32, 3> {
        #[allow(clippy::cast_precision_loss)]
        let n = self.x.len() as f32;
        SVector::<f32, 3>::new(
            self.x.iter().sum::<f32>() / n,
            self.y.iter().sum::<f32>() / n,
            self.z.iter().sum::<f32>() / n,
        )
    }

    #[must_use]
    pub fn zero_center_of_mass(mut self) -> Self {
        let com = self.center_of_mass();
        for i in 0..self.x.len() {
            self.x[i] -= com[0];
            self.y[i] -= com[1];
            self.z[i] -= com[2];
        }
        self
    }

    #[must_use]
    pub fn total_velocity(&self) -> SVector<f32, 3> {
        #[allow(clippy::cast_precision_loss)]
        let n = self.x.len() as f32;
        SVector::<f32, 3>::new(
            self.vx.iter().sum::<f32>() / n,
            self.vy.iter().sum::<f32>() / n,
            self.vz.iter().sum::<f32>() / n,
        )
    }

    #[must_use]
    pub fn zero_total_velocity(mut self) -> Self {
        let total_vel = self.total_velocity();
        for i in 0..self.x.len() {
            self.vx[i] -= total_vel[0];
            self.vy[i] -= total_vel[1];
            self.vz[i] -= total_vel[2];
        }
        self
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
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
        for i in 0..self.x.len() {
            for j in 0..3 {
                sample[j] = normal.sample(&mut rng);
            }
            let dv = mu + sigma.component_mul(&sample);
            self.vx[i] += dv[0];
            self.vy[i] += dv[1];
            self.vz[i] += dv[2];
        }
        self
    }

    #[allow(clippy::suboptimal_flops)]
    fn angular_momentum_particle_xy(&self, particle_index: usize, com: SVector<f32, 3>) -> f32 {
        let particle_pos = vector![self.x[particle_index], self.y[particle_index], 0.0];
        let particle_vel = vector![self.vx[particle_index], self.vy[particle_index], 0.0];
        let pos = particle_pos - com;
        pos[0] * particle_vel[1] - pos[1] * particle_vel[0]
    }

    /// Returns the total angular momentum of the universe in the xy plane.
    ///
    /// # Returns
    ///
    /// Total angular momentum in the xy plane of the universe.
    #[must_use]
    pub fn total_angular_momentum_xy(&self) -> f32 {
        let com = self.center_of_mass();
        let mut total_angular_momentum = 0.0;
        for i in 0..self.x.len() {
            total_angular_momentum += self.angular_momentum_particle_xy(i, com);
        }
        total_angular_momentum
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
        for i in 0..self.x.len() {
            self.vx[i] += -self.y[i] * target_angular_velocity;
            self.vy[i] += self.x[i] * target_angular_velocity;
        }
        self
    }

    #[allow(clippy::similar_names)]
    #[allow(clippy::suboptimal_flops)]
    pub fn step(&mut self) {
        // reset forces
        for i in 0..self.x.len() {
            self.fx[i] = 0.0;
            self.fy[i] = 0.0;
            self.fz[i] = 0.0;
        }

        // accumulate forces
        for i in 0..self.x.len() {
            for j in i + 1..self.x.len() {
                let dx = self.x[j] - self.x[i];
                let dy = self.y[j] - self.y[i];
                let dz = self.z[j] - self.z[i];

                let r_squared = dx * dx + dy * dy + dz * dz;
                let softened = r_squared + 0.01; // (r^2 + epsilon^2)
                let inv_r = 1.0 / softened.sqrt();

                let f12 = 1.0e-4 * inv_r.powi(3);
                let fx = f12 * dx;
                let fy = f12 * dy;
                let fz = f12 * dz;

                self.fx[i] += fx;
                self.fy[i] += fy;
                self.fz[i] += fz;

                self.fx[j] -= fx;
                self.fy[j] -= fy;
                self.fz[j] -= fz;
            }
        }

        // update velocities and positions
        for i in 0..self.x.len() {
            self.vx[i] += self.fx[i];
            self.vy[i] += self.fy[i];
            self.vz[i] += self.fz[i];

            self.x[i] += self.vx[i];
            self.y[i] += self.vy[i];
            self.z[i] += self.vz[i];
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
            x: vec![-1.0, 2.0],
            y: vec![-2.0, 4.0],
            z: vec![0.0, 0.0],
            vx: vec![0.0; 2],
            vy: vec![0.0; 2],
            vz: vec![0.0; 2],
            fx: vec![0.0; 2],
            fy: vec![0.0; 2],
            fz: vec![0.0; 2],
        };
        let com = universe.center_of_mass();
        assert_relative_eq!(com, vector![0.5, 1.0, 0.0]);
    }

    #[test]
    fn test_zero_center_of_mass() {
        let universe = Universe {
            x: vec![-1.0, 2.0],
            y: vec![-2.0, 4.0],
            z: vec![0.0, 0.0],
            vx: vec![0.0; 2],
            vy: vec![0.0; 2],
            vz: vec![0.0; 2],
            fx: vec![0.0; 2],
            fy: vec![0.0; 2],
            fz: vec![0.0; 2],
        };
        let zeroed_universe = universe.zero_center_of_mass();
        let com = zeroed_universe.center_of_mass();
        assert_relative_eq!(com, vector![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_gaussian_nebula_particle_count() {
        let mu = vector![0.0, 0.0, 0.0];
        let sigma = vector![1.0, 1.0, 1.0];
        let n = 100;
        let universe = Universe::gaussian_nebula(n, mu, sigma, None);
        assert_eq!(universe.x.len(), n);
    }
}
