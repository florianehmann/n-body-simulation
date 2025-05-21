use nalgebra::SVector;

use super::particle::Particle;

struct OctreeNode {
    center: SVector<f32, 3>,
    children: Option<[Box<OctreeNode>; 8]>,
    half_width: f32,
    particle_index: Option<usize>,
}

impl OctreeNode {
    pub fn new() -> Self {
        Self {
            center: SVector::zeros(),
            children: None,
            half_width: 1.0,
            particle_index: None,
        }
    }

    pub fn insert(&mut self, particle_index: usize, particles: &Vec<Particle>) {
        if self.children.is_none() {
            self.particle_index = Some(particle_index);
            return;
        }

        if self.children.is_none() {
            self.subdivide();
            if let Some(index) = self.particle_index.take() {
                self.insert(index, particles);
            }
        }

        let child_index = self.get_child_index(position);
        if let Some(children) = &mut self.children {
            children[child_index].insert(index, position);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        OctreeNode::new();
    }
}
