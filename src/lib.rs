//! # n-body-simulation
//!
//! A Rust library for simulating gravitational n-body systems in three dimensions.
//!
//! This crate provides efficient algorithms and data structures for simulating the dynamics
//! of many-body systems under Newtonian gravity. It includes a direct O(N²) solver as well as
//! an efficient Barnes-Hut octree implementation for O(N log N) force computation.
//!
//! ## Features
//! - Particle and universe abstractions
//! - Barnes-Hut octree for fast force calculation
//! - Utilities for initializing and evolving particle systems

#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![deny(clippy::nursery)]

pub mod sim;
