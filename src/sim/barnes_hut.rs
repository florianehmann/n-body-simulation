//! Barnes-Hut Octree implementation for efficient n-body force computation.
//!
//! This module provides the [`OctreeNode`] structure and related algorithms for spatial
//! partitioning and hierarchical force approximation in gravitational n-body simulations.
//! The Barnes-Hut algorithm reduces the computational complexity of force calculations
//! from O(N²) to O(N log N) by grouping distant particles and approximating their collective
//! influence using multipole expansions.
//!
//! # Features
//! - Construction of an octree from a set of particles
//! - Efficient insertion and subdivision logic
//! - Recursive aggregation of mass and center of mass for each subtree
//! - Fast force evaluation using the multipole acceptance criterion (MAC)
//! - Debugging utilities for tree inspection
//!
//! # Usage
//! This module is intended to be used as part of the simulation core. The octree can be
//! constructed from a list of particles and then used to efficiently compute gravitational
//! forces using the Barnes-Hut approximation.
//!
//! # Example
//! ```rust
//! use n_body_simulation::sim::barnes_hut::OctreeNode;
//! use n_body_simulation::sim::particle::Particle;
//! use nalgebra::vector;
//!
//! let particles = vec![Particle::new(vector![0.0, 0.0, 0.0], None)];
//! let tree = OctreeNode::from_particles(&particles);
//! ```

use nalgebra::{SVector, vector};

use super::particle::Particle;

pub struct SubtreeAggregate {
    pub center_of_mass: SVector<f32, 3>,
    pub total_mass: f32,
}

pub struct OctreeNode {
    center: SVector<f32, 3>,
    children: Option<[Box<OctreeNode>; 8]>,
    half_width: f32,
    particle_index: Option<usize>,
    aggregate: Option<SubtreeAggregate>,
}

/// Represents a node in an octree structure.
///
/// Each `OctreeNode` can either be a leaf node containing a single particle or an internal node
/// with up to eight children, each representing a subregion of space. The node stores its spatial
/// center, half-width (defining its cubic bounds), and optionally the index of a particle if it is
/// leaf. Internal nodes aggregate information about their subtree, such as the total mass and
/// center of mass, to enable efficient force calculations.
///
/// # Usage
/// This struct is intended for use in spatial partitioning and efficient force computation in
/// n-body simulations using the Barnes-Hut approximation.
impl OctreeNode {
    /// Creates a new `OctreeNode` with the specified center and half-width.
    ///
    /// # Parameters
    /// - `center`: The center position of the node in 3D space.
    /// - `half_width`: Half the width of the node's bounding cube.
    ///
    /// # Returns
    /// A new `OctreeNode` with no children, no particle, and no aggregate data.
    #[must_use]
    pub const fn new(center: SVector<f32, 3>, half_width: f32) -> Self {
        Self {
            center,
            children: None,
            half_width,
            particle_index: None,
            aggregate: None,
        }
    }

    /// Inserts a particle into the octree node.
    ///
    /// This method places the particle with the given `particle_index` into the octree,
    /// subdividing the node if necessary. If the node is a leaf and empty, the particle
    /// is stored directly. If the node already contains a particle and has no children,
    /// it is subdivided, and both the existing and new particles are reinserted into the
    /// appropriate child nodes. If the node has children, the particle is recursively
    /// inserted into the correct child node based on its position.
    ///
    /// # Parameters
    /// - `particle_index`: The index of the particle to insert.
    /// - `particles`: A slice containing all particles, used to access the position of the
    ///   particle.
    ///
    /// # Panics
    /// Panics if the node is expected to contain a particle but does not. This should not occur
    /// under normal operation, as the logic ensures a particle is present before subdivision.
    pub fn insert(&mut self, particle_index: usize, particles: &[Particle]) {
        if self.particle_index.is_none() && self.children.is_none() {
            self.particle_index = Some(particle_index);
            return;
        }

        // turn leaf into inner node and reinsert pre-existing leaf particle
        if self.children.is_none() {
            let existing_index = self
                .particle_index
                .take()
                .expect("Ensured by early return above");
            self.subdivide();
            self.insert(existing_index, particles);
        }

        let child_index = self.get_child_index(particles[particle_index].pos);
        if let Some(children) = &mut self.children {
            children[child_index].insert(particle_index, particles);
        }
    }

    /// Inserts all particles from the given slice into the octree.
    ///
    /// This method iterates over the provided slice of [`Particle`] instances and inserts
    /// each particle into the octree by calling [`Self::insert`] for each index. This is
    /// typically used to populate the octree with all particles at construction time.
    ///
    /// # Parameters
    /// - `particles`: A slice containing all particles to be inserted into the octree.
    pub fn insert_particles(&mut self, particles: &[Particle]) {
        particles.iter().enumerate().for_each(|(idx, _)| {
            self.insert(idx, particles);
        });
    }

    /// Constructs an `OctreeNode` from a slice of particles.
    ///
    /// This method determines the axis-aligned bounding box that contains all particles,
    /// then creates a root node centered in this box with a half-width large enough to
    /// encompass all particles. It inserts all particles into the tree and computes
    /// aggregate properties (center of mass and total mass) for each subtree.
    ///
    /// # Parameters
    /// - `particles`: A slice of [`Particle`]s to be inserted into the octree.
    ///
    /// # Returns
    /// An `OctreeNode` representing the root of the constructed octree.
    #[must_use]
    pub fn from_particles(particles: &[Particle]) -> Self {
        let (mut min, mut max) = (particles[0].pos, particles[0].pos);
        for p in particles.iter().skip(1) {
            for i in 0..3 {
                if p.pos[i] < min[i] {
                    min[i] = p.pos[i];
                }
                if p.pos[i] > max[i] {
                    max[i] = p.pos[i];
                }
            }
        }
        let center = (min + max) / 2.0;
        let half_width = ((max - min).amax()) / 2.0 + 1e-5;
        let mut tree = Self::new(center, half_width);
        tree.insert_particles(particles);
        tree.compute_aggregates(particles);
        tree
    }

    /// Subdivides the current octree node into eight child nodes.
    ///
    /// This method splits the cubic region represented by the current node into eight
    /// equally sized subcubes (octants), each represented by a new child node. The children
    /// are positioned such that together they fully cover the original node's volume.
    /// Each child node is centered at the appropriate offset from the parent node's center,
    /// and has half the parent's half-width.
    ///
    /// After calling this method, the `children` field will contain an array of eight
    /// boxed `OctreeNode` instances, each representing one octant of the parent node's space.
    pub fn subdivide(&mut self) {
        let half_width = self.half_width / 2.0;
        let mut child_idx = 0;
        let children = std::array::from_fn(|_| {
            let i_x = (child_idx >> 2) & 1;
            let i_y = (child_idx >> 1) & 1;
            let i_z = child_idx & 1;
            let offset = vector![
                if i_x == 0 { -half_width } else { half_width },
                if i_y == 0 { -half_width } else { half_width },
                if i_z == 0 { -half_width } else { half_width },
            ];
            child_idx += 1;
            Box::new(Self::new(self.center + offset, half_width))
        });

        self.children = Some(children);
    }

    /// Returns the index of the child octant that contains the given position.
    ///
    /// This method determines which of the eight child nodes (octants) a given position
    /// belongs to, based on its coordinates relative to the center of this node.
    ///
    /// # Parameters
    /// - `position`: The 3D position to locate within the octree node.
    ///
    /// # Returns
    /// An integer in the range 0..8 indicating the child index (octant).
    #[must_use]
    pub fn get_child_index(&self, position: SVector<f32, 3>) -> usize {
        let mut index = 0;
        if position[0] >= self.center[0] {
            index |= 4; // 100
        }
        if position[1] >= self.center[1] {
            index |= 2; // 010
        }
        if position[2] >= self.center[2] {
            index |= 1; // 001
        }
        index
    }

    /// Recursively applies a function to this node and all descendant nodes.
    ///
    /// This method traverses the octree in depth-first order, calling the provided
    /// function `func` on each node.
    ///
    /// # Parameters
    /// - `func`: A mutable reference to a function or closure to apply to each node.
    pub fn for_each_dyn(&self, func: &mut dyn FnMut(&Self)) {
        func(self);
        if let Some(children) = &self.children {
            for child in children {
                child.for_each_dyn(func);
            }
        }
    }

    /// Computes and stores the aggregate properties (center of mass and total mass) for this node.
    ///
    /// This method recursively computes the total mass and center of mass for each subtree,
    /// storing the result in the `aggregate` field. For leaf nodes, the aggregate is based
    /// on the contained particle. For internal nodes, the aggregate is computed from the
    /// aggregates of all children.
    ///
    /// # Parameters
    /// - `particles`: A slice of all particles, used to access positions and masses.
    ///
    /// # Panics
    /// Panics if a child node does not have its aggregate computed.
    pub fn compute_aggregates(&mut self, particles: &[Particle]) {
        if let Some(children) = &mut self.children {
            let mut center_of_mass_term = SVector::<f32, 3>::zeros();
            let mut total_mass: f32 = 0.0;
            for child in children {
                child.compute_aggregates(particles);

                let child_aggregate = child
                    .aggregate
                    .as_ref()
                    .expect("Child has just been aggregated");
                center_of_mass_term += child_aggregate.center_of_mass * child_aggregate.total_mass;
                total_mass += child_aggregate.total_mass;
            }

            self.aggregate = Some(SubtreeAggregate {
                center_of_mass: if total_mass > 0.0 {
                    center_of_mass_term / total_mass
                } else {
                    self.center
                },
                total_mass,
            });
        } else if let Some(particle_index) = self.particle_index {
            self.aggregate = Some(SubtreeAggregate {
                center_of_mass: particles[particle_index].pos,
                total_mass: 1.0,
            });
        } else {
            self.aggregate = Some(SubtreeAggregate {
                center_of_mass: self.center,
                total_mass: 0.0,
            });
        }
    }

    /// Applies a function to all relevant subtree aggregates for a given position using the
    /// Barnes-Hut criterion.
    ///
    /// This method traverses the octree and, for each node, decides whether to treat the node as a
    /// single aggregate (if the opening angle criterion `theta_mac` is satisfied) or to recurse
    /// into its children. The provided function `func` is called for each relevant aggregate.
    ///
    /// # Parameters
    /// - `position`: The position from which the opening angle is evaluated (e.g., the position of
    ///   the particle being updated).
    /// - `theta_mac`: The maximum allowed opening angle for the multipole acceptance criterion
    ///   (MAC).
    /// - `func`: A mutable reference to a function or closure to apply to each relevant aggregate.
    ///
    /// # Panics
    /// Panics if the tree has not been aggregated by previously calling
    /// [`Self::compute_aggregates`].
    /// This happens automatically if the tree is constructed by [`Self::from_particles`].
    pub fn for_each_relevant_aggregate<F>(
        &self,
        position: SVector<f32, 3>,
        theta_mac: f32,
        func: &mut F,
    ) where
        F: FnMut(&SubtreeAggregate),
    {
        let r = (position - self.center).norm();
        let theta = 2.0 * self.half_width / r;
        if theta < theta_mac {
            func(
                self.aggregate
                    .as_ref()
                    .expect("Tree needs to be aggregated"),
            );
            return;
        }

        if let Some(children) = &self.children {
            for child in children {
                child.for_each_relevant_aggregate(position, theta_mac, func);
            }
        }
    }

    /// Recursively prints the structure of the octree for debugging purposes.
    ///
    /// This method prints information about the current node, including whether it holds a
    /// particle, whether it has children, and its spatial bounds. It then recursively prints all
    /// child nodes, increasing the indentation for each level of depth. The method skips over empty
    /// leaf nodes.
    ///
    /// # Parameters
    /// - `depth`: The current depth in the tree, used to control indentation.
    pub fn print(&self, depth: usize) {
        let indent = " ".repeat(4 * depth);

        if self.particle_index.is_none() && self.children.is_none() {
            return;
        }

        println!(
            "{}Node: holds particle {}, children {}",
            indent,
            self.particle_index.is_some(),
            self.children.is_some()
        );

        let coord_min = self.center.map(|x| x - self.half_width);
        let coord_max = self.center.map(|x| x + self.half_width);
        println!("{indent}Bounds:");
        println!("{indent} x: {} to {}", coord_min[0], coord_max[0]);
        println!("{indent} y: {} to {}", coord_min[1], coord_max[1]);
        println!("{indent} z: {} to {}", coord_min[2], coord_max[2]);
        if let Some(children) = &self.children {
            for child in children {
                child.print(depth + 1);
            }
        } else {
            println!("{indent}Leaf node");
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use rand::{Rng, SeedableRng};

    use crate::sim::universe::Universe;

    use super::*;

    #[test]
    fn test_init() {
        let _ = OctreeNode::new(SVector::zeros(), 1.0);
    }

    #[test]
    fn test_subdivide() {
        let mut node = OctreeNode::new(SVector::zeros(), 1.0);
        node.subdivide();
        assert!(node.children.is_some());
        assert_eq!(node.children.as_ref().unwrap().len(), 8);
    }

    #[test]
    fn test_insert() {
        let mut node = OctreeNode::new(SVector::zeros(), 1.0);
        let particles = vec![
            Particle::new(vector![0.1, 0.1, 0.1], None),
            Particle::new(vector![-0.1, -0.1, -0.1], None),
        ];

        node.insert(0, &particles);
        assert!(node.particle_index.is_some());

        node.insert(1, &particles);
        assert!(node.children.is_some());
        assert_eq!(node.children.as_ref().unwrap().len(), 8);
    }

    fn create_test_tree(particle_count: usize) -> (OctreeNode, Vec<Particle>) {
        let mut node = OctreeNode::new(SVector::zeros(), 1.0);
        let mut seed_array = [0u8; 32];
        seed_array[..8].copy_from_slice(&245_i64.to_le_bytes());
        let mut rng = rand::rngs::StdRng::from_seed(seed_array);
        let particles = (0..particle_count)
            .map(|_| {
                let pos = vector![
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0)
                ];
                Particle::new(pos, None)
            })
            .collect::<Vec<_>>();

        node.insert_particles(&particles);

        (node, particles)
    }

    #[test]
    fn test_insert_particles() {
        let (tree, particles) = create_test_tree(10);

        let mut particles_in_tree: usize = 0;
        tree.for_each_dyn(&mut |n| {
            if n.particle_index.is_some() {
                particles_in_tree += 1;
            }
        });

        assert_eq!(particles_in_tree, particles.len());
    }

    #[test]
    fn test_aggregate_preserve_total_mass() {
        let (mut tree, particles) = create_test_tree(10);
        let universe = Universe { particles };
        tree.compute_aggregates(&universe.particles);
        let tree_aggregate = tree.aggregate.expect("Aggregated tree");

        #[allow(clippy::cast_precision_loss)]
        let n = universe.particles.len() as f32;
        assert_relative_eq!(tree_aggregate.total_mass, n);
    }

    #[test]
    fn test_aggregate_preserve_center_of_mass() {
        let (mut tree, particles) = create_test_tree(10);
        let universe = Universe { particles };
        tree.compute_aggregates(&universe.particles);
        let tree_aggregate = tree.aggregate.expect("Aggregated tree");
        assert_relative_eq!(tree_aggregate.center_of_mass, universe.center_of_mass());
    }
}
