use super::Universe;

/// Trait defining the interface for force models.
///
/// Force models determine and update the forces on every particle in a universe based on their
/// positions.
pub trait ForceModel {
    /// Compute the current forces on every particle in the universe and store them in the
    /// particles' force accumulators.
    fn compute_forces(&mut self, universe: &mut Universe);
}
