use macroquad::prelude::*;

#[macroquad::main("N-Body")]
async fn main() {
    let (w, h) = (screen_width(), screen_height());
    let r = 5.0;

    let mut x1: f32 = 0.25 * w;
    let mut x2: f32 = 0.75 * w;
    let mut y1: f32 = 0.5 * h;
    let mut y2: f32 = 0.5 * h;

    let mut v1_x = 0.0;
    let mut v1_y = 3.0;
    let mut v2_x = 0.0;
    let mut v2_y = -1.0;

    loop {
        clear_background(BLACK);

        // draw dots
        draw_circle(x1, y1, r, WHITE);
        draw_circle(x2, y2, r, WHITE);

        // determine forces
        let r12_x = x2 - x1;
        let r12_y = y2 - y1;
        let r12 = (r12_x.powf(2.0) + r12_y.powf(2.0)).powf(0.5);
        let f12 = 10000.0 / r12.powf(2.0);
        let f12_x = f12 * r12_x / r12;
        let f12_y = f12 * r12_y / r12;

        // update dots
        v1_x += f12_x;
        v1_y += f12_y;
        x1 += v1_x;
        y1 += v1_y;
        v2_x -= f12_x;
        v2_y -= f12_y;
        x2 += v2_x;
        y2 += v2_y;

        next_frame().await
    }
}
