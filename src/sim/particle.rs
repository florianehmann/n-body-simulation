use nalgebra::SVector;

#[derive(Clone, Debug)]
pub struct Particle {
    pub pos: SVector<f32, 3>,
    pub vel: SVector<f32, 3>,
    pub force: SVector<f32, 3>,
}

impl Particle {
    pub fn new(pos: SVector<f32, 3>, vel: Option<SVector<f32, 3>>) -> Self {
        Self {
            pos,
            vel: vel.unwrap_or_else(SVector::<f32, 3>::zeros),
            ..Default::default()
        }
    }

    #[must_use]
    pub fn vector_to(&self, other: &Self) -> SVector<f32, 3> {
        other.pos - self.pos
    }
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            pos: SVector::<f32, 3>::zeros(),
            vel: SVector::<f32, 3>::zeros(),
            force: SVector::<f32, 3>::zeros(),
        }
    }
}
