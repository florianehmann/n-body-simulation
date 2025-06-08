use nalgebra::SVector;
use rayon::prelude::*;

use crate::sim::{
    ForceModel, Universe,
    barnes_hut::{Octree, SubtreeAggregate},
};

/// Calculates particle forces through the Barnes-Hut approximation (O(N log N)).
///
/// This method resets all forces and computes gravitational forces on all particles.
pub struct BarnesHutForceModel {
    tree: Octree,
}

impl BarnesHutForceModel {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tree: Octree::new(SVector::zeros(), 1.0),
        }
    }
}

impl Default for BarnesHutForceModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ForceModel for BarnesHutForceModel {
    fn compute_forces(&mut self, universe: &mut Universe) {
        // reset forces
        for particle in &mut universe.particles {
            particle.force *= 0.0;
        }

        // determine Barnes-Hut octree
        self.tree = Octree::from_particles(&universe.particles);

        // determine approximate forces
        universe.particles.par_iter_mut().for_each(|particle| {
            let mut f = |agg: &SubtreeAggregate| {
                let r_vec = agg.center_of_mass - particle.pos;

                let r_squared = r_vec.norm_squared();
                let softened = r_squared + 0.01; // (r^2 + epsilon^2)
                let inv_r = 1.0 / softened.sqrt();

                let f12 = 1.0e-4 * agg.total_mass * inv_r * inv_r;
                let f12_vec = f12 * r_vec * inv_r;

                particle.force += f12_vec;
            };
            self.tree
                .for_each_relevant_aggregate(particle.pos, 1.0, &mut f);
        });
    }
}
