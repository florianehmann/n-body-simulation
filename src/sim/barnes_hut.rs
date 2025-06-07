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
//! use n_body_simulation::sim::Particle;
//! use nalgebra::vector;
//!
//! let particles = vec![Particle::new(vector![0.0, 0.0, 0.0], None)];
//! let tree = OctreeNode::from_particles(&particles);
//! ```

use nalgebra::{SVector, vector};

use super::Particle;

#[derive(Clone)]
pub struct SubtreeAggregate {
    pub center_of_mass: SVector<f32, 3>,
    pub total_mass: f32,
}

/// Represents a node in an octree structure.
///
/// Each `OctreeNode` can either be a leaf node containing a single particle or an internal node
/// with up to eight children, each representing a subregion of space. The node stores its spatial
/// center, half-width (defining its cubic bounds), and optionally the index of a particle if it is
/// leaf. Internal nodes aggregate information about their subtree, such as the total mass and
/// center of mass, to enable efficient force calculations.
#[derive(Clone)]
pub struct OctreeNode {
    center: SVector<f32, 3>,
    children: Option<[usize; 8]>,
    half_width: f32,
    particle_index: Option<usize>,
    aggregate: Option<SubtreeAggregate>,
}

/// Represents an entire octree.
///
/// The nodes of the tree are stored in a reusable arena and the root node is at index 0.
pub struct Octree {
    nodes: Vec<OctreeNode>,
}

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
}

impl Octree {
    /// Creates a new octree by instantiating the node arena and creating a root node.
    ///
    /// # Parameters
    /// - `center`: Center of the volume covered by the octree.
    /// - `half_width`: Half the width of the node's bounding cube.
    #[must_use]
    pub fn new(center: SVector<f32, 3>, half_width: f32) -> Self {
        Self {
            nodes: vec![OctreeNode::new(center, half_width)],
        }
    }

    /// Subdivides an octree node into eight child nodes.
    ///
    /// This method splits the cubic region represented by the current node into eight
    /// equally sized subcubes (octants), each represented by a new child node. The children
    /// are positioned such that together they fully cover the original node's volume.
    /// Each child node is centered at the appropriate offset from the parent node's center,
    /// and has half the parent's half-width.
    pub fn subdivide_node(&mut self, node_idx: usize) {
        let node_center = self.nodes[node_idx].center;
        let half_width = self.nodes[node_idx].half_width / 2.0;

        let mut child_indices = [0usize; 8];
        #[allow(clippy::needless_range_loop)]
        for i in 0..8 {
            let i_x = (i >> 2) & 1;
            let i_y = (i >> 1) & 1;
            let i_z = i & 1;
            let offset = vector![
                if i_x == 0 { -half_width } else { half_width },
                if i_y == 0 { -half_width } else { half_width },
                if i_z == 0 { -half_width } else { half_width },
            ];
            let child = OctreeNode::new(node_center + offset, half_width);
            self.nodes.push(child);
            child_indices[i] = self.nodes.len() - 1;
        }

        self.nodes[node_idx].children = Some(child_indices);
    }

    /// Inserts a particle into the specified octree node.
    ///
    /// This method places the particle with the given `particle_index` into the octree,
    /// subdividing the node if necessary. If the node is a leaf and empty, the particle
    /// is stored directly. If the node already contains a particle and has no children,
    /// it is subdivided, and both the existing and new particles are reinserted into the
    /// appropriate child nodes. If the node has children, the particle is recursively
    /// inserted into the correct child node based on its position.
    ///
    /// # Parameters
    /// - `node_index`: The index of the node at which to insert.
    /// - `particle_index`: The index of the particle to insert.
    /// - `particles`: A slice containing all particles, used to access the position of the
    ///   particle.
    ///
    /// # Panics
    /// Panics if the node is expected to contain a particle but does not. This should not occur
    /// under normal operation, as the logic ensures a particle is present before subdivision.
    pub fn insert(&mut self, node_index: usize, particle_index: usize, particles: &[Particle]) {
        // block limits scope of mutable borrow to node before calling insert recursively
        let need_subdivide = {
            let node = &mut self.nodes[node_index];
            if node.particle_index.is_none() && node.children.is_none() {
                node.particle_index = Some(particle_index);
                return;
            }
            node.children.is_none()
        };

        // turn leaf into inner node and reinsert pre-existing leaf particle
        if need_subdivide {
            let existing_index = self.nodes[node_index]
                .particle_index
                .take()
                .expect("Ensured by early return above");
            self.subdivide_node(node_index);
            self.insert(node_index, existing_index, particles);
        }

        let child_index = {
            let node = &self.nodes[node_index];
            node.get_child_index(particles[particle_index].pos)
        };
        if let Some(children) = &self.nodes[node_index].children {
            self.insert(children[child_index], particle_index, particles);
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
            self.insert(0, idx, particles);
        });
    }

    /// Constructs an octree from a slice of particles.
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
    /// An [`Octree`] that spatially indexes the `particles`.
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
        tree.compute_aggregates(0, particles);
        tree
    }

    /// Computes and stores the aggregate properties (center of mass and total mass) for a given
    /// node.
    ///
    /// This method recursively computes the total mass and center of mass for each subtree,
    /// storing the result in the `aggregate` field. For leaf nodes, the aggregate is based
    /// on the contained particle. For internal nodes, the aggregate is computed from the
    /// aggregates of all children.
    ///
    /// # Parameters
    /// - `node_index`: Index of the node within the arena for which to aggregate subtree.
    /// - `particles`: A slice of all particles, used to access positions and masses.
    ///
    /// # Panics
    /// Panics if a child node does not have its aggregate computed.
    pub fn compute_aggregates(&mut self, node_index: usize, particles: &[Particle]) {
        if let Some(child_indices) = self.nodes[node_index].children {
            let mut center_of_mass_term = SVector::<f32, 3>::zeros();
            let mut total_mass: f32 = 0.0;
            for child_index in child_indices {
                self.compute_aggregates(child_index, particles);
                let child_aggregate = self.nodes[child_index]
                    .aggregate
                    .as_ref()
                    .expect("Child has just been aggregated");
                center_of_mass_term += child_aggregate.center_of_mass * child_aggregate.total_mass;
                total_mass += child_aggregate.total_mass;
            }

            self.nodes[node_index].aggregate = Some(SubtreeAggregate {
                center_of_mass: if total_mass > 0.0 {
                    center_of_mass_term / total_mass
                } else {
                    self.nodes[node_index].center
                },
                total_mass,
            });
        } else if let Some(particle_index) = self.nodes[node_index].particle_index {
            self.nodes[node_index].aggregate = Some(SubtreeAggregate {
                center_of_mass: particles[particle_index].pos,
                total_mass: 1.0,
            });
        } else {
            self.nodes[node_index].aggregate = Some(SubtreeAggregate {
                center_of_mass: self.nodes[node_index].center,
                total_mass: 0.0,
            });
        }
    }

    /// Traverse the octree in depth-first order, calling the provided function `func` on each node.
    ///
    /// # Parameters
    /// - `func`: A mutable reference to a function or closure to apply to each node.
    pub fn for_each_dyn(&self, func: &mut dyn FnMut(&OctreeNode)) {
        self.for_each_dyn_internal(0, func);
    }

    /// Internal recursive helper for `for_each_dyn`.
    /// Traverses the octree from the given node index, applying the function to each node.
    fn for_each_dyn_internal(&self, node_index: usize, func: &mut dyn FnMut(&OctreeNode)) {
        func(&self.nodes[node_index]);
        if let Some(child_indices) = self.nodes[node_index].children {
            for child_index in child_indices {
                self.for_each_dyn_internal(child_index, func);
            }
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
        self.for_each_relevant_aggregate_internal(0, position, theta_mac, func);
    }

    /// Recursion logic helper for [`Self::for_each_relevant_aggregate`].
    /// Allows for specifying node.
    fn for_each_relevant_aggregate_internal<F>(
        &self,
        node_index: usize,
        position: SVector<f32, 3>,
        theta_mac: f32,
        func: &mut F,
    ) where
        F: FnMut(&SubtreeAggregate),
    {
        let r = (position - self.nodes[node_index].center).norm();
        let theta = 2.0 * self.nodes[node_index].half_width / r;
        if theta < theta_mac {
            func(
                self.nodes[node_index]
                    .aggregate
                    .as_ref()
                    .expect("Tree needs to be aggregated"),
            );
            return;
        }

        if let Some(child_indices) = self.nodes[node_index].children {
            for child_index in child_indices {
                self.for_each_relevant_aggregate_internal(child_index, position, theta_mac, func);
            }
        }
    }

    /// Recursively prints the structure of the octree for debugging purposes.
    ///
    /// This method prints information about the current node, including whether it holds a
    /// particle, whether it has children, and its spatial bounds. It then recursively prints all
    /// child nodes, increasing the indentation for each level of depth. The method skips over empty
    /// leaf nodes.
    pub fn print(&self) {
        self.print_internal(0, 0);
    }

    /// Recursion logic helper for [`Self::print`].
    /// Allows for specifying node and print depth.
    fn print_internal(&self, node_index: usize, depth: usize) {
        let indent = " ".repeat(4 * depth);

        if self.nodes[node_index].particle_index.is_none()
            && self.nodes[node_index].children.is_none()
        {
            return;
        }

        println!(
            "{}Node: holds particle {}, children {}",
            indent,
            self.nodes[node_index].particle_index.is_some(),
            self.nodes[node_index].children.is_some()
        );

        let coord_min = self.nodes[node_index]
            .center
            .map(|x| x - self.nodes[node_index].half_width);
        let coord_max = self.nodes[node_index]
            .center
            .map(|x| x + self.nodes[node_index].half_width);
        println!("{indent}Bounds:");
        println!("{indent} x: {} to {}", coord_min[0], coord_max[0]);
        println!("{indent} y: {} to {}", coord_min[1], coord_max[1]);
        println!("{indent} z: {} to {}", coord_min[2], coord_max[2]);
        if let Some(child_indices) = self.nodes[node_index].children {
            for child_index in child_indices {
                self.print_internal(child_index, depth + 1);
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
        let mut tree = Octree::new(SVector::zeros(), 1.0);
        tree.subdivide_node(0);
        assert!(tree.nodes[0].children.is_some());
        assert_eq!(tree.nodes[0].children.as_ref().unwrap().len(), 8);
    }

    #[test]
    fn test_insert() {
        let mut tree = Octree::new(SVector::zeros(), 1.0);
        let particles = vec![
            Particle::new(vector![0.1, 0.1, 0.1], None),
            Particle::new(vector![-0.1, -0.1, -0.1], None),
        ];

        tree.insert(0, 0, &particles);
        assert!(tree.nodes[0].particle_index.is_some());

        tree.insert(0, 1, &particles);
        assert!(tree.nodes[0].children.is_some());
        assert_eq!(tree.nodes[0].children.as_ref().unwrap().len(), 8);
    }

    fn create_test_tree(particle_count: usize) -> (Octree, Vec<Particle>) {
        let mut tree = Octree::new(SVector::zeros(), 1.0);
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

        tree.insert_particles(&particles);

        (tree, particles)
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
        tree.compute_aggregates(0, &universe.particles);
        let tree_aggregate = tree.nodes[0].clone().aggregate.expect("Aggregated tree");

        #[allow(clippy::cast_precision_loss)]
        let n = universe.particles.len() as f32;
        assert_relative_eq!(tree_aggregate.total_mass, n);
    }

    #[test]
    fn test_aggregate_preserve_center_of_mass() {
        let (mut tree, particles) = create_test_tree(10);
        let universe = Universe { particles };
        tree.compute_aggregates(0, &universe.particles);
        let tree_aggregate = tree.nodes[0].clone().aggregate.expect("Aggregated tree");
        assert_relative_eq!(tree_aggregate.center_of_mass, universe.center_of_mass());
    }
}
