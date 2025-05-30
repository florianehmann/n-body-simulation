use nalgebra::{SVector, vector};

use super::particle::Particle;

struct OctreeNode {
    center: SVector<f32, 3>,
    children: Option<[Box<OctreeNode>; 8]>,
    half_width: f32,
    particle_index: Option<usize>,
}

impl OctreeNode {
    pub fn new(center: SVector<f32, 3>, half_width: f32) -> Self {
        Self {
            center: center,
            children: None,
            half_width: half_width,
            particle_index: None,
        }
    }

    pub fn insert(&mut self, particle_index: usize, particles: &Vec<Particle>) {
        if self.particle_index.is_none() {
            self.particle_index = Some(particle_index);
            return;
        }

        if self.children.is_none() {
            self.subdivide();
            if let Some(index) = self.particle_index.take() {
                self.insert(index, particles);
            }
        }

        let child_index = self.get_child_index(particles[particle_index].pos);
        if let Some(children) = &mut self.children {
            children[child_index].insert(particle_index, particles);
        }
    }

    pub fn subdivide(&mut self) {
        let half_width = self.half_width / 2.0;
        let mut idx = 0;
        let children = std::array::from_fn(|_| {
            let i_x = (idx >> 2) & 1;
            let i_y = (idx >> 1) & 1;
            let i_z = idx & 1;
            let offset = vector![
                if i_x == 0 { -half_width } else { half_width },
                if i_y == 0 { -half_width } else { half_width },
                if i_z == 0 { -half_width } else { half_width },
            ];
            idx += 1;
            Box::new(OctreeNode::new(self.center + offset, half_width))
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
}

#[cfg(test)]
mod tests {
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
}
