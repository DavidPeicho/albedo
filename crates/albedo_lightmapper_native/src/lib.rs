use albedo_backend::gpu;
use albedo_bvh;
use albedo_bvh::{builders, BLASArray, Mesh};
use albedo_rtx::passes;
use albedo_rtx::uniforms::{Instance, PerDrawUniforms, Vertex};
use std::sync::Mutex;

mod app;
pub use app::*;

pub struct StridedSlice<V> {
    pub stride: u32,
    pub bytes: *mut u8,
    pub ptr: *mut V,
}

impl<V: Sized> StridedSlice<V> {
    pub fn from_raw(stride: u32, bytes: *mut u8) -> Self {
        StridedSlice::<V> {
            stride,
            bytes,
            ptr: bytes as *mut V
        }
    }

    pub fn get(&self, index: usize) -> &V {
        unsafe {
            let start = self.bytes.offset(self.stride as isize * index as isize);
            (start as *mut V).as_ref().unwrap()
        }
    }
}

pub struct MeshData<'a> {
    positions: StridedSlice<[f32; 3]>,
    normals: StridedSlice<[f32; 3]>,
    uvs: StridedSlice<[f32; 2]>,
    indices: &'a [u32],
    vertex_count: u32,
    index_count: u32,
}

#[repr(C)]
pub struct AttributeSlice {
    pub stride: u32,
    pub bytes: *mut u8,
}

#[repr(C)]
pub struct ImageSlice {
    pub width: u32,
    pub height: u32,
    pub data: *mut u8,
}

#[repr(C)]
pub struct MeshDescriptor {
    positions: AttributeSlice,
    normals: AttributeSlice,
    uvs: AttributeSlice,
    indices: *const u32,
    vertex_count: u32,
    index_count: u32,
}

impl<'a> Mesh<Vertex> for MeshData<'a> {
    fn index(&self, index: u32) -> Option<&u32> {
        self.indices.get(index as usize)
    }

    fn vertex(&self, index: u32) -> Vertex {
        let i: usize = index as usize;
        let pos: &[f32; 3] = self.positions.get(i);
        let normal: &[f32; 3] = self.normals.get(i);
        let uv: &[f32; 2] = self.uvs.get(i);
        Vertex::new(pos, normal, Some(uv))
    }

    fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    fn index_count(&self) -> u32 {
        self.index_count
    }

    fn position(&self, index: u32) -> Option<&[f32; 3]> {
        Some(self.positions.get(index as usize))
    }
}

pub struct Renderer {
    pub lightmap_pass: passes::LightmapPass,
    global_uniforms_buffer: gpu::Buffer<PerDrawUniforms>,
    lightmap_bindgroup: wgpu::BindGroup,
    size: (u32, u32),
}

impl Renderer {
    pub fn new(
        context: &GpuContext,
        size: (u32, u32),
        scene_resources: &SceneGPU,
        swapchain_format: wgpu::TextureFormat,
    ) -> Self {
        let global_uniforms_buffer: gpu::Buffer<PerDrawUniforms> =
            gpu::Buffer::new_uniform(context.device(), 1, None);
        let lightmap_pass = passes::LightmapPass::new(context.device(), swapchain_format);
        let lightmap_bindgroup: wgpu::BindGroup = lightmap_pass.create_frame_bind_groups(
            context.device(),
            &scene_resources.instance_buffer,
            &scene_resources.bvh_buffer.inner(),
            &scene_resources.index_buffer,
            &scene_resources.vertex_buffer.inner(),
            &global_uniforms_buffer,
        );
        Self {
            global_uniforms_buffer,
            lightmap_pass,
            lightmap_bindgroup,
            size,
        }
    }

    pub async fn lightmap(
        &mut self,
        context: &GpuContext,
        scene_resources: &SceneGPU,
    ) -> Result<Vec<u8>, &'static str> {
        let device = &context.device;
        let queue = &context.queue;

        let alignment = albedo_backend::Alignment2D::texture_buffer_copy(
            self.size.0 as usize,
            std::mem::size_of::<u32>(),
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Pixel Encoder"),
        });
        let (width, height) = self.size;
        let gpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: height as u64 * alignment.padded_bytes() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let texture_extent = wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            label: None,
            view_formats: &[],
        });
        let view: wgpu::TextureView = texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.lightmap_pass.draw(
            &mut encoder,
            &view,
            &self.lightmap_bindgroup,
            &scene_resources.instance_buffer,
            &scene_resources.index_buffer,
            &scene_resources.vertex_buffer.inner(),
        );

        device.poll(wgpu::Maintain::Wait);

        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::ImageCopyBuffer {
                buffer: &gpu_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        std::num::NonZeroU32::new(alignment.padded_bytes() as u32).unwrap(),
                    ),
                    rows_per_image: None,
                },
            },
            texture_extent,
        );
        queue.submit(Some(encoder.finish()));

        let buffer_slice = gpu_buffer.slice(..);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = receiver.receive().await {
            let padded_buffer = buffer_slice.get_mapped_range();
            let mut bytes: Vec<u8> = vec![0; alignment.unpadded_bytes_per_row * height as usize];
            // from the padded_buffer we write just the unpadded bytes into the image
            for (padded, bytes) in padded_buffer
                .chunks_exact(alignment.padded_bytes_per_row)
                .zip(bytes.chunks_exact_mut(alignment.unpadded_bytes_per_row))
            {
                bytes.copy_from_slice(&padded[..alignment.unpadded_bytes_per_row]);
            }
            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(padded_buffer);
            gpu_buffer.unmap();
            Ok(bytes)
        } else {
            Err("Fail to read pixels in texture to buffer")
        }
    }
}

static app: Mutex<Option<App>> = Mutex::new(None);

#[no_mangle]
pub extern "C" fn init() {
    println!("Hello from Rust");
    println!("CWD: {}", std::env::current_dir().unwrap().as_os_str().to_str().unwrap());

    unsafe {
        *app.lock().unwrap() = Some(App::new());
    }
    println!("{}", app.lock().unwrap().is_none());
}

#[no_mangle]
pub extern "C" fn set_mesh_data(desc: MeshDescriptor) {
    if desc.index_count % 3 != 0 {
        panic!("Vertex count must be a multiple of 3");
    }

    let mut guard = app.lock().unwrap();
    println!("Seting mesh data...");
    println!("{}", guard.is_none());

    let runtime: &mut App = guard.as_mut().unwrap();

    let mesh_data = MeshData {
        positions: StridedSlice::<[f32; 3]>::from_raw(desc.positions.stride, desc.positions.bytes),
        normals: StridedSlice::<[f32; 3]>::from_raw(desc.normals.stride, desc.positions.bytes),
        uvs: StridedSlice::<[f32; 2]>::from_raw(desc.uvs.stride, desc.uvs.bytes),
        indices: unsafe { std::slice::from_raw_parts(desc.indices, desc.index_count as usize) },
        vertex_count: desc.vertex_count,
        index_count: desc.index_count,
    };

    // @todo: Skip conversion by making the BVH / GPU struct split the vertex.
    let mut vertices: Vec<Vertex> = Vec::with_capacity(desc.vertex_count as usize);
    for i in 0..desc.vertex_count as usize {
        vertices.push(Vertex::new(
            mesh_data.positions.get(i),
            mesh_data.normals.get(i),
            Some(mesh_data.uvs.get(i)),
        ));
    }

    let mut builder = builders::SAHBuilder::new();
    let result = BLASArray::new(&[mesh_data], &mut builder);

    let blas = match result {
        Ok(val) => val,
        Err(str) => return,
    };

    let entry = blas.entries.get(0).unwrap();
    let instance = Instance {
        vertex_root_index: entry.vertex,
        index_root_index: entry.index,
        bvh_root_index: entry.node,
        ..Default::default()
    };

    runtime.scene = Some(SceneGPU::new(
        runtime.context.device(),
        &[instance],
        &blas.nodes,
        &blas.indices,
        &blas.vertices,
    ));
}

#[no_mangle]
pub extern "C" fn bake(raw_slice: ImageSlice) {
    println!("Baking...");
    let mut guard = app.lock().unwrap();
    let mut runtime = guard.as_mut().unwrap();

    let mut context = &mut runtime.context;

    if let Some(renderdoc) = &mut context.renderdoc {
        renderdoc.start_frame_capture(std::ptr::null(), std::ptr::null());
    }

    println!("\n============================================================");
    println!("                   🚀 Lightmapper 🚀                           ");
    println!("============================================================\n");

    let init_size = (512, 512);

    let mut scene = match &runtime.scene {
        Some(val) => val,
        None => panic!("No scene provided before bake()"),
    };

    let mut renderer = Renderer::new(
        &context,
        (init_size.0, init_size.1),
        &scene,
        wgpu::TextureFormat::Rgba8Unorm,
    );

    let data = futures::executor::block_on(renderer.lightmap(&context, scene)).unwrap();

    println!("[ALBEDO]: Read {} bytes ", data.len());
    println!(
        "[ALBEDO]: First few pixels: [{}, {}, {}, {}], [{}, {}, {}, {}]",
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
    );

    let byte_count = (raw_slice.width * raw_slice.height * 4) as usize;
    let out = unsafe { std::slice::from_raw_parts_mut(raw_slice.data, byte_count) };
    out.copy_from_slice(&data);

    if let Some(renderdoc) = &mut context.renderdoc {
        renderdoc.end_frame_capture(std::ptr::null(), std::ptr::null());
    }

    println!("[ALBEDO] Copy is done!");
}
