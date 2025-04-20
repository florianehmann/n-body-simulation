use macroquad::prelude::*;
use n_body_simulation::sim::{particle::Particle, universe::Universe};
use nalgebra::vector;

#[macroquad::main("N-Body")]
async fn main() {
    let (w, h) = (screen_width(), screen_height());

    let mut universe = Universe::new(vec![
        Particle::new(vector![0.25 * w, 0.5 * h], vector![0.0, 0.0]),
        Particle::new(vector![0.75 * w, 0.5 * h], vector![0.0, 0.0]),
        Particle::new(vector![0.5 * w, 0.75 * h], vector![0.0, 0.0]),
        Particle::new(vector![0.3 * w, 0.8 * h], vector![0.0, 0.0]),
        Particle::new(vector![0.7 * w, 0.2 * h], vector![0.0, 0.0]),
        Particle::new(vector![0.2 * w, 0.2 * h], vector![0.0, 0.0]),
    ])
    .zero_total_velocity();

    let mut camera = Camera2D {
        target: vec2(0.0, 0.0),
        zoom: vec2(1.0 / screen_width() * 0.5, -1.0 / screen_height() * 0.5),
        ..Default::default()
    };
    let mut zoom_factor = 1.0;

    loop {
        zoom_and_pan(&mut camera, &mut zoom_factor);

        set_camera(&camera);

        clear_background(BLACK);
        draw_universe(&universe, &camera);

        universe.step();

        set_default_camera();

        // draw UI elements here

        next_frame().await
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

    // Optional: drag with right mouse to pan
    if is_mouse_button_down(MouseButton::Right) {
        let delta = mouse_delta_position();
        // world delta = screen delta scaled by inverse zoom
        let zoom_inv = vec2(1.0 / camera.zoom.x, 1.0 / camera.zoom.y);
        camera.target += vec2(delta.x, delta.y) * zoom_inv;
    }
}

fn draw_universe<const D: usize>(universe: &Universe<D>, camera: &Camera2D) {
    let r = 5.0;
    for particle in &universe.particles {
        let (x, y): (f32, f32) = camera
            .world_to_screen(vec2(particle.pos[0], particle.pos[1]))
            .into();
        draw_circle(x, y, r, WHITE);
    }
}
