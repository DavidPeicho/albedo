pub mod macros;
pub mod passes;
pub mod texture;
pub mod uniforms;

pub use uniforms::*;

pub fn get_dispatch_size(
    size: (u32, u32, u32),
    workgroup_size: (u32, u32, u32),
) -> (u32, u32, u32) {
    let x: f32 = (size.0 as f32) / workgroup_size.0 as f32;
    let y: f32 = (size.1 as f32) / workgroup_size.1 as f32;
    let z: f32 = (size.2 as f32) / workgroup_size.2 as f32;
    return (x.ceil() as u32, y.ceil() as u32, z.ceil() as u32);
}
