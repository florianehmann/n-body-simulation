use super::Universe;

/// Trait defining the interface for dynamics integrators.
///
/// Integrators apply the forces acting on the particles to evolve their position and velocity
/// through time.
pub trait Integrator {
    /// Advance the simulation by one time step.
    fn step(&mut self, universe: &mut Universe);
}
