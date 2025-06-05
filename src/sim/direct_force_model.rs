use crate::sim::{ForceModel, Universe};

/// Calculates particle forces through direct pairwise force computation (O(N²)).
///
/// This method resets all forces, computes gravitational forces between all unique pairs of
/// particles, and then updates velocities and positions accordingly.
pub struct DirectForceModel {}

impl ForceModel for DirectForceModel {
    fn compute_forces(&mut self, universe: &mut Universe) {
        // reset forces
        for particle in &mut universe.particles {
            particle.force *= 0.0;
        }

        // accumulate forces
        for i in 0..universe.particles.len() {
            for j in i + 1..universe.particles.len() {
                let (left, right) = universe.particles.split_at_mut(j);
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
    }
}
