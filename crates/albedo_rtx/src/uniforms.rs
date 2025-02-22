use albedo_backend::{gpu, mesh};
use bytemuck::{Pod, Zeroable};

use glam::Vec4Swizzles;
use std::convert::TryInto;

pub static INVALID_INDEX: u32 = std::u32::MAX;

pub trait Uniform: Sized {
    fn size_in_bytes() -> u32 {
        std::mem::size_of::<Self>() as u32
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    pub model_to_world: glam::Mat4,
    pub world_to_model: glam::Mat4,
    pub material_index: u32,
    // @todo: migrate those parameter to an SSBO of offsets.
    pub bvh_root_index: u32,
    pub vertex_root_index: u32,
    pub bvh_primitive_index: u32,
}
impl Uniform for Instance {}

impl Instance {
    pub fn from_transform(model_to_world: glam::Mat4) -> Self {
        let world_to_model = model_to_world.inverse();
        Self {
            model_to_world,
            world_to_model,
            ..Default::default()
        }
    }

    pub fn set_transform(&mut self, model_to_world: glam::Mat4) {
        self.model_to_world = model_to_world;
        self.world_to_model = self.model_to_world.inverse();
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Material {
    pub color: glam::Vec4,
    pub roughness: f32,
    pub reflectivity: f32,
    pub albedo_texture: u32,
    pub mra_texture: u32,
}
unsafe impl bytemuck::Pod for Material {}
unsafe impl bytemuck::Zeroable for Material {}
impl Uniform for Material {}

impl Material {
    pub fn new(color: glam::Vec4, roughness: f32, reflectivity: f32) -> Material {
        Material {
            color,
            roughness,
            reflectivity,
            albedo_texture: INVALID_INDEX,
            mra_texture: INVALID_INDEX,
            ..Default::default()
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Vertex {
    pub position: [f32; 4],
    pub normal: [f32; 4],
}
unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}
impl Uniform for Vertex {}

impl Vertex {
    const DEFAULT_UV: [f32; 2] = [0.0, 0.0];

    pub fn new(position: &[f32; 3], normal: &[f32; 3], uv: Option<&[f32; 2]>) -> Self {
        let uv = uv.unwrap_or(&Self::DEFAULT_UV);
        Vertex {
            position: [position[0], position[1], position[2], uv[0]],
            normal: [normal[0], normal[1], normal[2], uv[1]],
        }
    }

    pub fn position(&self) -> &[f32; 3] {
        self.position[0..3].try_into().unwrap()
    }
}

impl mesh::AsVertexFormat for Vertex {
    fn as_vertex_formats() -> &'static [mesh::AttributeDescriptor] {
        static ATTRIBUTE_DESCRIPTORS: [mesh::AttributeDescriptor; 2] = [
            mesh::AttributeDescriptor {
                id: mesh::AttributeId::POSITION,
                format: wgpu::VertexFormat::Float32x4,
            },
            mesh::AttributeDescriptor {
                id: mesh::AttributeId::NORMAL,
                format: wgpu::VertexFormat::Float32x4,
            },
        ];
        &ATTRIBUTE_DESCRIPTORS
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Light {
    pub normal: glam::Vec4,
    pub tangent: glam::Vec4,
    pub bitangent: glam::Vec4,
    pub intensity: f32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
}

unsafe impl bytemuck::Pod for Light {}
unsafe impl bytemuck::Zeroable for Light {}
impl Uniform for Light {}

impl Light {
    pub fn new() -> Self {
        // `origin` is packed in `normal`, `tangent`, and `bitangent`.
        // By default, camera set at the origin.
        Light {
            normal: glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
            tangent: glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
            bitangent: glam::Vec4::new(0.0, -1.0, 0.0, 0.0),
            intensity: 1.0,
            ..Default::default()
        }
    }

    pub fn from_origin(origin: glam::Vec3) -> Self {
        Light {
            normal: glam::Vec4::new(0.0, 0.0, 1.0, origin.x),
            tangent: glam::Vec4::new(1.0, 0.0, 0.0, origin.y),
            bitangent: glam::Vec4::new(0.0, -1.0, 0.0, origin.z),
            intensity: 1.0,
            ..Default::default()
        }
    }

    pub fn from_matrix(local_to_world: glam::Mat4) -> Self {
        let mut light = Light::new();
        light.set_from_matrix(local_to_world, 1.0, 1.0);
        light
    }

    pub fn set_from_matrix(&mut self, local_to_world: glam::Mat4, width: f32, height: f32) {
        let mut origin = local_to_world.w_axis;
        self.normal = local_to_world * glam::Vec4::new(0.0, 0.0, 1.0, 0.0);
        self.tangent = local_to_world * glam::Vec4::new(width, 0.0, 0.0, 0.0);
        self.bitangent = local_to_world * glam::Vec4::new(0.0, -height, 0.0, 0.0);

        origin = origin - 0.5 * self.tangent - 0.5 * self.bitangent;

        // Pack origin into the normal, tangent, and bitangent vectors.
        self.normal.w = origin.x;
        self.tangent.w = origin.y;
        self.bitangent.w = origin.z;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PerDrawUniforms {
    pub frame_count: u32,
    pub seed: u32,
    pub bounces: u32,
    pub padding: u32,
    pub dimensions: [u32; 2],
}

impl PerDrawUniforms {
    pub fn new() -> Self {
        PerDrawUniforms {
            ..Default::default()
        }
    }
}

unsafe impl bytemuck::Pod for PerDrawUniforms {}
unsafe impl bytemuck::Zeroable for PerDrawUniforms {}
impl Uniform for PerDrawUniforms {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Camera {
    pub origin: glam::Vec3,
    pub v_fov: f32,
    pub up: glam::Vec3,
    pub padding_0: f32,
    pub right: glam::Vec3,
    pub padding_1: f32,
    pub dimensions: [u32; 2],
    pub padding_2: [u32; 2],
}

impl Camera {
    pub fn set_transform(&mut self, transform: &glam::Mat4) {
        self.right = transform.x_axis.xyz();
        self.up = transform.y_axis.xyz();
        self.origin = transform.w_axis.xyz();
    }

    pub fn perspective(&self, near: f32, far: f32) -> glam::Mat4 {
        let aspect = self.dimensions[0] as f32 / self.dimensions[1] as f32;
        glam::Mat4::perspective_lh(self.v_fov, aspect, near, far)
    }

    pub fn transform(&self) -> glam::Mat4 {
        let dir = self.up.cross(self.right).normalize().extend(0.0);
        let rot = glam::Mat4::from_cols(
            self.right.normalize().extend(0.0),
            self.up.normalize().extend(0.0),
            dir,
            glam::Vec4::W,
        );
        glam::Mat4::from_translation(self.origin) * rot
    }
}

impl Default for Camera {
    fn default() -> Camera {
        Camera {
            origin: glam::Vec3::new(0.0, 0.0, 2.0),
            v_fov: 0.78,
            up: glam::Vec3::new(0.0, 1.0, 0.0),
            right: glam::Vec3::new(1.0, 0.0, 0.0),
            padding_0: 0.0,
            padding_1: 0.0,
            padding_2: [0, 0],
            dimensions: [1, 1],
        }
    }
}

unsafe impl bytemuck::Pod for Camera {}
unsafe impl bytemuck::Zeroable for Camera {}
impl Uniform for Camera {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Ray {
    origin: glam::Vec4,
    dir: glam::Vec4,
    radiance: glam::Vec4,
    terminated: [u32; 4],
}
unsafe impl bytemuck::Pod for Ray {}
unsafe impl bytemuck::Zeroable for Ray {}
impl Uniform for Ray {}

impl Ray {
    pub fn new() -> Self {
        Ray {
            origin: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            dir: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            radiance: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            terminated: [0, 0, 0, 0],
        }
    }

    pub fn from_origin_dir(origin: &glam::Vec3, direction: glam::Vec3) -> Self {
        Ray {
            origin: glam::Vec4::new(origin.x, origin.y, origin.z, 1.0),
            dir: glam::Vec4::new(direction.x, direction.y, direction.z, 1.0),
            radiance: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            terminated: [0, 0, 0, 0],
        }
    }

    pub fn throughput(&self) -> glam::Vec3 {
        glam::Vec3::new(self.origin.w, self.dir.w, self.radiance.w)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Intersection {
    uv: glam::Vec2,
    index: u32,
    instance: u32,
    material_index: u32,
    emitter: u32,
    dist: f32,
    padding_0: f32,
}

unsafe impl bytemuck::Pod for Intersection {}
unsafe impl bytemuck::Zeroable for Intersection {}
impl Uniform for Intersection {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TextureInfo {
    x: u32,
    y: u32,
    width: u32,
    // Height should be stored in the first 24 bits, and atlas on the last 8 bits.
    layer_and_height: u32,
}

impl TextureInfo {
    pub fn pack_value(layer: u32, value: u32) -> u32 {
        layer << 24 | value
    }

    pub fn unpack_value(packed: u32) -> u32 {
        packed & 0x00FFFFFF
    }

    pub fn unpack_layer(packed: u32) -> u8 {
        ((packed & 0xFF000000) >> 24) as u8
    }

    pub fn new(layer: u8, x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            layer_and_height: Self::pack_value(layer as u32, height),
        }
    }

    pub fn x(&self) -> u32 {
        self.x
    }
    pub fn y(&self) -> u32 {
        self.y
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        Self::unpack_value(self.layer_and_height)
    }
    pub fn layer(&self) -> u8 {
        Self::unpack_layer(self.layer_and_height)
    }
}

unsafe impl bytemuck::Pod for TextureInfo {}
unsafe impl bytemuck::Zeroable for TextureInfo {}
impl Uniform for TextureInfo {}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct RadianceParameters {
    pub use_noise_texture: u32,
}

#[cfg(feature = "tinybvh")]
pub type BVHNode = tinybvh_rs::cwbvh::Node;
#[cfg(not(feature = "tinybvh"))]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
/// Layout from from tinybvh-rs.
///
/// Note: obvh has a different field ordering and shouldn't be used as-is.
pub struct BVHNode {
    /// AABB min.
    pub min: [f32; 3],
    /// Exponent used for child AABB decompression.
    pub exyz: [u8; 3],
    /// `1` for node, `0` for leaf.
    pub imask: u8,
    /// First child index.
    pub child_base_idx: u32,
    // First primitive index.
    pub primitive_base_idx: u32,
    /// Child [0..7] metadata.
    pub child_meta: [u8; 8],
    // AABB minimum x-axis compressed bound, one entry per child.
    pub qlo_x: [u8; 8],
    // AABB minimum y-axis compressed bound, one entry per child.
    pub qlo_y: [u8; 8],
    // AABB minimum z-axis compressed bound, one entry per child.
    pub qlo_z: [u8; 8],
    // AABB maximum x-axis compressed bound, one entry per child.
    pub qhi_x: [u8; 8],
    // AABB maximum y-axis compressed bound, one entry per child.
    pub qhi_y: [u8; 8],
    // AABB maximum z-axis compressed bound, one entry per child.
    pub qhi_z: [u8; 8],
}
impl Uniform for BVHNode {}

#[cfg(feature = "tinybvh")]
pub type BVHPrimitive = tinybvh_rs::cwbvh::Primitive;
#[cfg(not(feature = "tinybvh"))]
/// Layout from from tinybvh-rs.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BVHPrimitive {
    pub edge_1: [f32; 3],
    pub padding_0: u32,
    pub edge_2: [f32; 3],
    pub padding_1: u32,
    pub vertex_0: [f32; 3],
    pub original_primitive: u32,
}
impl Uniform for BVHPrimitive {}

pub struct RaytraceResources<'a> {
    pub rays: gpu::StorageBufferSlice<'a, Ray>,
    pub intersections: gpu::StorageBufferSlice<'a, Intersection>,
    pub global_uniforms: gpu::UniformBufferSlice<'a, PerDrawUniforms>,
    pub camera_uniforms: gpu::UniformBufferSlice<'a, Camera>,
}

#[derive(Clone, Copy)]
pub struct DenoiseResources<'a> {
    pub gbuffer_current: &'a wgpu::TextureView,
    pub gbuffer_previous: &'a wgpu::TextureView,
    pub motion: &'a wgpu::TextureView,
}

impl<'a> DenoiseResources<'a> {
    pub fn pong(&self) -> DenoiseResources<'a> {
        Self {
            gbuffer_current: self.gbuffer_previous,
            gbuffer_previous: self.gbuffer_current,
            motion: self.motion,
        }
    }
}
