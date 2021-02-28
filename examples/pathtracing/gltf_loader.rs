use albedo_rtx::{accel::{BVH, BVHNodeGPU, BVHBuilder, SAHBuilder}, mesh::Mesh};
use albedo_rtx::renderer;
use gltf::{self, json::Index};
use std::path::Path;

pub struct ProxyMesh {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
}
impl Mesh for ProxyMesh {

    fn index(&self, index: u32) -> Option<&u32> {
        self.indices.get(index as usize)
    }

    fn normal(&self, index: u32) -> Option<&[f32; 3]> {
        self.normals.get(index as usize)
    }

    // @todo: instead of reading vertex / buffer etc, why not ask user to fill
    // our data stucture?
    // If data are linear, user can do a memcpy, otherwise he must memcpy with
    // stride, but at least it's up to him and can give a nice perf boost.

    fn has_normal(&self) -> bool {
        // @todo: do not assume model has normals.
        true
    }
    fn has_tangent(&self) -> bool {
        false
    }
    fn has_uv0(&self) -> bool {
        false
    }

    fn vertex_count(&self) -> u32 {
        self.positions.len() as u32
    }

    fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    fn position(&self, index: u32) -> Option<&[f32; 3]> {
        self.positions.get(index as usize)
    }
}

pub struct Scene {
    pub meshes: Vec<ProxyMesh>,
    pub bvhs: Vec<BVH>,
    pub instances: Vec<renderer::resources::InstanceGPU>,
    pub node_buffer: Vec<BVHNodeGPU>,
    pub vertex_buffer: Vec<renderer::resources::VertexGPU>,
    pub index_buffer: Vec<u32>,
}

pub fn load_gltf<P: AsRef<Path>>(file_path: &P) -> Scene {
    let (doc, buffers, images) = match gltf::import(file_path) {
        Ok(tuple) => tuple,
        Err(err) => {
            panic!("glTF import failed: {:?}", err);
            // if let gltf::Error::Io(_) = err {
            //     error!("Hint: Are the .bin file(s) referenced by the .gltf file available?")
            // }
        }
    };
    let mut meshes: Vec<ProxyMesh> = Vec::new();
    let mut instances: Vec<renderer::resources::InstanceGPU> = Vec::new();

    for mesh in doc.meshes() {
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            positions.extend(reader.read_positions().unwrap());
            normals.extend(reader.read_normals().unwrap());
            indices.extend(reader
                .read_indices()
                .map(|read_indices| read_indices.into_u32())
                .unwrap()
            );
        }
        meshes.push(ProxyMesh {
            positions,
            normals,
            indices,
        });
    }

    let mut bvhs: Vec<BVH> = meshes
        .iter()
        .map(|mesh| {
            // @todo: allow user to choose builder.
            let mut builder = SAHBuilder::new();
            builder.build(mesh).unwrap()
        })
        .collect();

    let gpu_resources = renderer::utils::build_acceleration_structure_gpu(
        &bvhs,
        &meshes
    );

    for node in doc.nodes() {
        // @todo: handle scene graph.
        // User should have their own scene graph. However, for pure pathtracing
        // from format like glTF, a small footprint hierarchy handler should be
        // provided.
        if let Some(mesh) = node.mesh() {
            let index = mesh.index();
            let offset_table = gpu_resources.offset_table.get(index).unwrap();
            instances.push(renderer::resources::InstanceGPU {
                world_to_model: glam::Mat4::from_cols_array_2d(&node.transform().matrix()).inverse(),
                material_index: 0,
                bvh_root_index: offset_table.node(),
                vertex_root_index: offset_table.vertex(),
                index_root_index: offset_table.index(),
            });
        }
    }

    Scene {
        meshes,
        instances,
        bvhs,
        node_buffer: gpu_resources.nodes_buffer,
        vertex_buffer: gpu_resources.vertex_buffer,
        index_buffer: gpu_resources.index_buffer
    }
}
