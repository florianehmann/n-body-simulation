//! Particle type for n-body simulations.
//!
//! This module defines the [`Particle`] struct, which represents a single body in the simulation
//! with position, velocity, and force vectors in three-dimensional space. It also provides
//! constructors and utility methods for particle manipulation.

use nalgebra::SVector;

/// Represents a single particle in the n-body simulation.
///
/// Each particle has a position, velocity, and force vector in 3D space.
#[derive(Clone, Debug)]
pub struct Particle {
    /// Position vector (x, y, z) of the particle.
    pub pos: SVector<f32, 3>,
    /// Velocity vector (vx, vy, vz) of the particle.
    pub vel: SVector<f32, 3>,
    /// Accumulated force vector (fx, fy, fz) acting on the particle.
    pub force: SVector<f32, 3>,
}

impl Particle {
    /// Creates a new `Particle` with the given position and optional velocity.
    ///
    /// If `vel` is `None`, the velocity is initialized to zero.
    ///
    /// # Parameters
    /// - `pos`: The initial position of the particle.
    /// - `vel`: Optional initial velocity of the particle.
    ///
    /// # Returns
    /// A new `Particle` instance.
    pub fn new(pos: SVector<f32, 3>, vel: Option<SVector<f32, 3>>) -> Self {
        Self {
            pos,
            vel: vel.unwrap_or_else(SVector::<f32, 3>::zeros),
            ..Default::default()
        }
    }

    /// Computes the vector from this particle to another particle.
    ///
    /// # Parameters
    /// - `other`: The other particle.
    ///
    /// # Returns
    /// The vector pointing from `self` to `other`.
    #[must_use]
    pub fn vector_to(&self, other: &Self) -> SVector<f32, 3> {
        other.pos - self.pos
    }
}

impl Default for Particle {
    /// Returns a particle at the origin with zero velocity and zero force.
    fn default() -> Self {
        Self {
            pos: SVector::<f32, 3>::zeros(),
            vel: SVector::<f32, 3>::zeros(),
            force: SVector::<f32, 3>::zeros(),
        }
    }
}
