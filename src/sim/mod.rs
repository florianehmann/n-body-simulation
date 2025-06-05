//! Simulation module for n-body dynamics.
//!
//! This module contains the core components for simulating gravitational n-body systems,
//! including particle definitions, the Barnes-Hut octree for efficient force computation,
//! and the universe abstraction for managing collections of particles and advancing the simulation.

pub mod barnes_hut;
pub mod direct_force_model;
pub mod euler_integrator;
pub mod force_model;
pub mod integrator;
pub mod universe;

pub use force_model::ForceModel;
pub use integrator::Integrator;
pub use universe::{Particle, Universe};
