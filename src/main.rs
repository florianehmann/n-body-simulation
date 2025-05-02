#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![deny(clippy::nursery)]

use std::{thread::sleep, time::Duration};

use macroquad::prelude::*;
use n_body_simulation::sim::universe::Universe;
use nalgebra::vector;

#[macroquad::main("N-Body Simulation")]
async fn main() {
    let mut universe =
        Universe::gaussian_nebula(2000, vector![0.0, 0.0, 0.0], vector![13.4, 13.4, 1.3], None)
            .zero_center_of_mass()
            .set_random_velocity(vector![0.0, 0.0, 0.0], vector![0.02, 0.02, 0.002], None)
            .zero_total_velocity()
            .set_rotation_period(5000.0);

    let mut camera = Camera2D {
        target: vec2(0.0, 0.0),
        ..Default::default()
    };
    set_camera(&camera);
    let mut zoom_factor = 15.0;
    let mut first_frame = true;
    let mut waited = false;

    loop {
        clear_background(BLACK);

        if first_frame {
            camera.target = vec2(0.0, 0.0);
            camera.zoom = vec2(zoom_factor / screen_width(), -zoom_factor / screen_height());
            first_frame = false;
        } else {
            zoom_and_pan(&mut camera, &mut zoom_factor);
            if !waited {
                sleep(Duration::from_secs(1));
                waited = true;
            }
        }

        set_camera(&camera);

        // draw_circle(0.0, 0.0, 10.0, YELLOW);
        draw_universe(&universe);

        universe.step();

        set_default_camera();

        // draw UI elements here
        draw_fps();

        next_frame().await;
    }
}

fn zoom_and_pan(camera: &mut Camera2D, zoom_factor: &mut f32) {
    // Save world pos under mouse *before* zoom
    let mouse_screen = vec2(mouse_position().0, mouse_position().1);
    let before_zoom = camera.screen_to_world(mouse_screen);

    // Scroll to zoom
    let scroll = mouse_wheel().1;
    if scroll != 0.0 {
        *zoom_factor *= 1.1_f32.powf(scroll); // exponential scale
    }

    camera.zoom = vec2(
        *zoom_factor * 1.0 / screen_width(),
        *zoom_factor * -1.0 / screen_height(),
    );

    // World pos under mouse *after* zoom
    let after_zoom = camera.screen_to_world(mouse_screen);

    // Adjust target to keep the mouse on the same world point
    camera.target += before_zoom - after_zoom;

    if is_mouse_button_down(MouseButton::Left) {
        let delta = mouse_delta_position();
        let zoom_inv = vec2(1.0 / camera.zoom.x, 1.0 / camera.zoom.y);
        camera.target += vec2(delta.x, delta.y) * zoom_inv;
    }
}

fn draw_universe<const D: usize>(universe: &Universe<D>) {
    let r = 0.1;
    for particle in &universe.particles {
        draw_circle(particle.pos[0], particle.pos[1], r, WHITE);
    }
}
