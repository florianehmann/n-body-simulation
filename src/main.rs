#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![deny(clippy::nursery)]

use std::{
    sync::{Arc, Mutex},
    thread::{self, sleep},
    time::{Duration, Instant},
};

use macroquad::prelude::*;
use n_body_simulation::sim::{
    ForceModel, Integrator, barnes_hut_force_model::BarnesHutForceModel,
    euler_integrator::EulerIntegrator, universe::Universe,
};
use nalgebra::vector;

#[macroquad::main("N-Body Simulation")]
async fn main() {
    let universe = Universe::gaussian_nebula(
        100_000,
        vector![0.0, 0.0, 0.0],
        vector![13.4, 13.4, 1.3],
        None,
    )
    .zero_center_of_mass()
    .set_random_velocity(vector![0.0, 0.0, 0.0], vector![0.02, 0.02, 0.002], None)
    .zero_total_velocity()
    .set_rotation_period(5000.0);

    let mut force_model = BarnesHutForceModel::new();
    let mut integrator = EulerIntegrator { dt: 1.0 };

    let render_buffer = Arc::new(Mutex::new(universe));
    let sim_render_buffer = Arc::clone(&render_buffer);
    let updates_per_second = Arc::new(Mutex::new(0.0));
    let ups_sensor = Arc::clone(&updates_per_second);
    thread::spawn(move || {
        let mut sim_universe = sim_render_buffer
            .lock()
            .expect("If this fails program is dead anyway")
            .clone();

        let target_sim_steps_per_second = 100.0;
        let dt = 1.0 / target_sim_steps_per_second;
        let dt_duration = Duration::from_secs_f32(dt);
        loop {
            let loop_start = Instant::now();
            force_model.compute_forces(&mut sim_universe);
            integrator.step(&mut sim_universe);

            // copy new frame to render buffer
            if let Ok(mut render_target) = sim_render_buffer.lock() {
                *render_target = sim_universe.clone();
            }

            let elapsed_loop = loop_start.elapsed();
            if let Ok(mut ups) = ups_sensor.lock() {
                *ups = 1.0 / elapsed_loop.as_secs_f32();
            }
            if elapsed_loop < dt_duration {
                thread::sleep(dt_duration - elapsed_loop);
            }
        }
    });

    let mut camera = Camera2D {
        target: vec2(0.0, 0.0),
        ..Default::default()
    };
    set_camera(&camera);
    let mut zoom_factor = 15.0;
    let mut first_frame = true;
    let mut waited = false;

    let mut ups = {
        let guard = updates_per_second
            .lock()
            .expect("If sim thread panicked, there's nothing we can do anyway");
        *guard
    };
    let mut fps = 0.0;

    #[allow(clippy::suboptimal_flops, clippy::cast_precision_loss)]
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
        if let Ok(render_universe) = render_buffer.lock() {
            draw_universe(&render_universe);
        }

        set_default_camera();

        // draw UI elements here
        let ups_new = {
            let guard = updates_per_second
                .lock()
                .expect("If sim thread panicked, there's nothing we can do anyway");
            *guard
        };
        ups = ups * 0.9 + ups_new * 0.1;
        let fps_new = get_fps() as f32;
        fps = fps * 0.9 + fps_new * 0.1;
        let status_text = format!("FPS: {fps:.1} / UPS: {ups:.1}");
        draw_text(status_text.as_str(), 20.0, 20.0, 30.0, WHITE);

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

fn draw_universe(universe: &Universe) {
    let r = 0.1;
    for particle in &universe.particles {
        draw_circle(particle.pos[0], particle.pos[1], r, WHITE);
    }
}
