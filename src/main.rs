use macroquad::prelude::*;
use n_body_simulation::sim::universe::Universe;
use nalgebra::vector;

#[macroquad::main("N-Body Simulation")]
async fn main() {
    let mut universe =
        Universe::gaussian_nebula(250, vector![0.0, 0.0], vector![600.0, 400.0], None)
            .zero_center_of_mass()
            .zero_total_velocity()
            .set_angular_momentum_xy_equally(100000.0);

    let mut camera = Camera2D {
        target: vec2(0.0, 0.0),
        ..Default::default()
    };
    set_camera(&camera);
    let mut zoom_factor = 1.0;
    let mut first_frame = true;

    loop {
        clear_background(BLACK);

        if first_frame {
            camera.target = vec2(0.0, 0.0);
            camera.zoom = vec2(zoom_factor / screen_width(), -zoom_factor / screen_height());
            first_frame = false;
        } else {
            zoom_and_pan(&mut camera, &mut zoom_factor);
        }

        set_camera(&camera);

        draw_circle(0.0, 0.0, 10.0, YELLOW);
        draw_universe(&universe);

        universe.step();

        set_default_camera();

        // draw UI elements here
        draw_fps();

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

    if is_mouse_button_down(MouseButton::Left) {
        let delta = mouse_delta_position();
        let zoom_inv = vec2(1.0 / camera.zoom.x, 1.0 / camera.zoom.y);
        camera.target += vec2(delta.x, delta.y) * zoom_inv;
    }
}

fn draw_universe<const D: usize>(universe: &Universe<D>) {
    let r = 5.0;
    for particle in &universe.particles {
        draw_circle(particle.pos[0], particle.pos[1], r, WHITE);
    }
}
