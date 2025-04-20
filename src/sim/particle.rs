use nalgebra::SVector;

pub struct Particle<const D: usize> {
    pub pos: SVector<f32, D>,
    pub vel: SVector<f32, D>,
    pub force: SVector<f32, D>,
}

impl<const D: usize> Particle<D> {
    pub fn new(pos: SVector<f32, D>, vel: SVector<f32, D>) -> Self {
        Self { pos, vel, force: SVector::<f32, D>::zeros() }
    }

    pub fn vector_to(&self, other: &Self) -> SVector<f32, D> {
        other.pos - self.pos
    }
}
