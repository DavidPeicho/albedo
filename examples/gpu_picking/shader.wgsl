struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) instance_index: u32,
};

struct Uniforms {
    mvpMatrix: mat4x4<f32>,
    color: vec4<f32>,
}

@binding(0) @group(0) var<storage, read> uniforms : array<Uniforms>;

@vertex
fn vs_main(
    @builtin(instance_index) idx : u32,
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
) -> VertexOutput {
    var result: VertexOutput;
    result.instance_index = idx;
    result.position = uniforms[idx].mvpMatrix * vec4(position.xyz, 1.0);
    return result;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(uniforms[vertex.instance_index].color.rgb, 1.0);
    // return vec4<f32>(vec3<f32>(1.0, 0.0, 0.0), 1.0);
}
