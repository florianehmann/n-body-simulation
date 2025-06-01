use nalgebra::{SVector, vector};

use super::particle::Particle;

struct SubtreeAggregate {
    center_of_mass: SVector<f32, 3>,
    total_mass: f32,
}

struct OctreeNode {
    center: SVector<f32, 3>,
    children: Option<[Box<OctreeNode>; 8]>,
    half_width: f32,
    particle_index: Option<usize>,
    aggregate: Option<SubtreeAggregate>,
}

impl OctreeNode {
    pub const fn new(center: SVector<f32, 3>, half_width: f32) -> Self {
        Self {
            center,
            children: None,
            half_width,
            particle_index: None,
            aggregate: None,
        }
    }

    pub fn insert(&mut self, particle_index: usize, particles: &Vec<Particle>) {
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

    pub fn insert_particles(&mut self, particles: &Vec<Particle>) {
        particles.iter().enumerate().for_each(|(idx, _)| {
            self.insert(idx, particles);
        });
    }

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

    pub fn for_each_dyn(&self, func: &mut dyn FnMut(&Self)) {
        func(self);
        if let Some(children) = &self.children {
            for child in children {
                child.for_each_dyn(func);
            }
        }
    }

    pub fn compute_aggregates(&mut self, particles: &Vec<Particle>) {
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

    #[allow(dead_code)]
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
    use rand::{Rng, SeedableRng};

    use crate::sim::universe::Universe;

    use super::*;

    #[test]
    fn test_init() {
        OctreeNode::new(SVector::zeros(), 1.0);
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
        assert!(tree_aggregate.total_mass - n < 1e-5);
    }

    #[test]
    fn test_aggregate_preserve_center_of_mass() {
        let (mut tree, particles) = create_test_tree(10);
        let universe = Universe { particles };
        tree.compute_aggregates(&universe.particles);
        let tree_aggregate = tree.aggregate.expect("Aggregated tree");

        let diff = tree_aggregate.center_of_mass - universe.center_of_mass();
        for i in 0..3 {
            assert!(diff[i].abs() < 1e-5, "Component {} differs: {}", i, diff[i]);
        }
    }
}
