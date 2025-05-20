use nalgebra::SVector;

#[derive(Clone)]
pub struct Particle<const D: usize> {
    pub pos: SVector<f32, D>,
    pub vel: SVector<f32, D>,
    pub force: SVector<f32, D>,
}

impl<const D: usize> Particle<D> {
    pub fn new(pos: SVector<f32, D>, vel: Option<SVector<f32, D>>) -> Self {
        Self {
            pos,
            vel: vel.unwrap_or_else(SVector::<f32, D>::zeros),
            ..Default::default()
        }
    }

    #[must_use]
    pub fn vector_to(&self, other: &Self) -> SVector<f32, D> {
        other.pos - self.pos
    }
}

impl<const D: usize> Default for Particle<D> {
    fn default() -> Self {
        Self {
            pos: SVector::<f32, D>::zeros(),
            vel: SVector::<f32, D>::zeros(),
            force: SVector::<f32, D>::zeros(),
        }
    }
}
