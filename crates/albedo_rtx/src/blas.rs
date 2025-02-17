use std::time::Duration;

use crate::{uniforms::Instance, BVHNode, BVHPrimitive, Vertex};
use obvhs::{self, triangle::Triangle};
use tinybvh_rs::cwbvh::{self, Primitive};

#[derive(Copy, Clone)]
pub struct MeshDescriptor<'a> {
    pub positions: pas::Slice<'a, [f32; 4]>,
    pub normals: Option<pas::Slice<'a, [f32; 3]>>,
    pub texcoords0: Option<pas::Slice<'a, [f32; 2]>>,
}

#[derive(Copy, Clone)]
pub struct IndexedMeshDescriptor<'a> {
    pub mesh: MeshDescriptor<'a>,
    pub indices: &'a [u32],
}

/// Node, vertex, and index offset of an entry
///
/// This is used to retrieve a flattened BVH into a buffer
#[derive(Default, Copy, Clone)]
pub struct BLASEntryDescriptor {
    pub node: u32,
    pub primitive: u32,
    pub vertex: u32,
}

/// Data-oriented storage for a list of BVH.
///
/// Data are stored in separate buffers:
///
/// `[vertex_0, vertex_1, vertex_2, ..., vertex_n]`
/// `[index_0, index_1, index_2, ..., index_j]`
/// `[entry_0, entry_1, entry_2, ..., entry_k]`
///
/// Entries are used to find the start index of each
/// BVH.
#[derive(Default)]
pub struct BLASArray {
    /// Node, vertex, and index offset for each entry
    pub entries: Vec<BLASEntryDescriptor>,
    /// List of nodes of all entries
    pub nodes: Vec<BVHNode>,
    /// List of indices of all entries
    pub primitives: Vec<BVHPrimitive>,
    pub vertices: Vec<Vertex>,
    pub instances: Vec<Instance>,
}

fn test(positions: pas::Slice<'_, [f32; 4]>) -> (Vec<BVHNode>, Vec<BVHPrimitive>) {
    let count = positions.len() / 3;
    let mut tris = Vec::with_capacity(count);
    for i in 0..count {
        let index = i * 3;
        tris.push(Triangle {
            v0: glam::Vec3A::new(
                positions[index][0],
                positions[index][1],
                positions[index][2],
            ),
            v1: glam::Vec3A::new(
                positions[index + 1][0],
                positions[index + 1][1],
                positions[index + 1][2],
            ),
            v2: glam::Vec3A::new(
                positions[index + 2][0],
                positions[index + 2][1],
                positions[index + 2][2],
            ),
        });
    }
    let bvh = obvhs::cwbvh::builder::build_cwbvh_from_tris(
        &tris,
        obvhs::BvhBuildParams::medium_build(),
        &mut Duration::default(),
    );

    let mut nodes = Vec::with_capacity(bvh.nodes.len());
    for node in &bvh.nodes {
        nodes.push(BVHNode {
            min: node.p.to_array(),
            exyz: [
                // Traversal code performs the exp2 unpacking, because
                // tinybvh doesn't bake it in exyz, at the opposite of obvh.
                node.e[0].wrapping_sub(127),
                node.e[1].wrapping_sub(127),
                node.e[2].wrapping_sub(127),
            ],
            imask: node.imask,
            child_base_idx: node.child_base_idx,
            primitive_base_idx: node.primitive_base_idx,
            child_meta: node.child_meta,
            qlo_x: node.child_min_x,
            qlo_y: node.child_min_y,
            qlo_z: node.child_min_z,
            qhi_x: node.child_max_x,
            qhi_y: node.child_max_y,
            qhi_z: node.child_max_z,
        });
    }

    let mut primitives: Vec<Primitive> = Vec::with_capacity(bvh.primitive_indices.len());
    for index in bvh.primitive_indices {
        let tri = &tris[index as usize];
        let edge_1 = tri.v1 - tri.v0;
        let edge_2 = tri.v2 - tri.v0;
        primitives.push(Primitive {
            edge_1: [edge_1.x, edge_1.y, edge_1.z],
            padding_0: 0,
            edge_2: [edge_2.x, edge_2.y, edge_2.z],
            padding_1: 0,
            vertex_0: [tri.v0.x, tri.v0.y, tri.v0.z],
            original_primitive: index as u32,
        });
    }

    (nodes, primitives)
}

impl BLASArray {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn add_bvh(&mut self, mesh: MeshDescriptor) {
        self.entries.push(BLASEntryDescriptor {
            node: self.nodes.len() as u32,
            primitive: self.primitives.len() as u32,
            vertex: self.vertices.len() as u32,
        });

        let start = self.vertices.len();
        self.vertices
            .resize(start + mesh.positions.len(), Vertex::default());
        let vertices: &mut [Vertex] = &mut self.vertices[start..];

        for i in 0..mesh.positions.len() {
            let pos = &mesh.positions[i];
            vertices[i].position = [pos[0], pos[1], pos[2], 0.0];
        }
        if let Some(normals) = mesh.normals {
            for i in 0..normals.len() {
                let normal = &normals[i];
                vertices[i].normal = [normal[0], normal[1], normal[2], 0.0];
            }
        }
        if let Some(texcoord) = mesh.texcoords0 {
            for i in 0..texcoord.len() {
                let uv = &texcoord[i];
                vertices[i].position[3] = uv[0];
                vertices[i].normal[3] = uv[1];
            }
        }
        // let bvh = cwbvh::BVH::new_hq(mesh.positions);
        // self.nodes.extend(bvh.nodes());
        // self.primitives.extend(bvh.primitives());

        // DEBUG
        let val = test(mesh.positions);
        self.nodes.extend(val.0);
        self.primitives.extend(val.1);
        // END DEBUG
    }

    pub fn add_bvh_indexed(&mut self, desc: IndexedMeshDescriptor) {
        self.entries.push(BLASEntryDescriptor {
            node: self.nodes.len() as u32,
            primitive: self.primitives.len() as u32,
            vertex: self.vertices.len() as u32,
        });

        let vertex_count = desc.indices.len();
        let start = self.vertices.len();
        self.vertices
            .resize(start + vertex_count, Vertex::default());

        let vertices: &mut [Vertex] = &mut self.vertices[start..];
        for (i, index) in desc.indices.into_iter().enumerate() {
            let position = &desc.mesh.positions[*index as usize];
            vertices[i].position = *position;
        }
        if let Some(normals) = desc.mesh.normals {
            for (i, index) in desc.indices.into_iter().enumerate() {
                let normal = &normals[*index as usize];
                vertices[i].normal = [normal[0], normal[1], normal[2], 0.0];
            }
        }
        if let Some(uvs) = desc.mesh.texcoords0 {
            for (i, index) in desc.indices.into_iter().enumerate() {
                let uv = &uvs[*index as usize];
                vertices[i].position[3] = uv[0];
                vertices[i].normal[3] = uv[1];
            }
        }

        let vertices: &[Vertex] = &self.vertices[start..];
        let positions: pas::Slice<[f32; 4]> = pas::Slice::new(vertices, 0);

        let bvh = cwbvh::BVH::new_hq(positions);
        let nodes = bvh.nodes();
        println!("Nodes = {}", nodes.len());
        // self.nodes.extend(bvh.nodes());
        // self.primitives.extend(bvh.primitives());
        // DEBUG
        let val = test(positions);
        self.nodes.extend(val.0);
        self.primitives.extend(val.1);
        // END DEBUG
    }

    pub fn add_instance(&mut self, bvh_index: u32, model_to_world: glam::Mat4, material: u32) {
        let entry = self.entries.get(bvh_index as usize).unwrap();
        self.instances.push(Instance {
            model_to_world,
            world_to_model: model_to_world.inverse(),
            material_index: material,
            bvh_root_index: entry.node,
            vertex_root_index: entry.vertex,
            bvh_primitive_index: entry.primitive,
        });
    }
}
