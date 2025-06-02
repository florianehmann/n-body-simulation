//! Simulation module for n-body dynamics.
//!
//! This module contains the core components for simulating gravitational n-body systems,
//! including particle definitions, the Barnes-Hut octree for efficient force computation,
//! and the universe abstraction for managing collections of particles and advancing the simulation.

pub mod barnes_hut;
pub mod particle;
pub mod universe;
