use albedo_backend::gpu;
use wgpu::Extent3d;

use crate::macros::path_separator;
use crate::{get_dispatch_size, uniforms};

pub struct ATrousPass {
    frame_bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,

    count: u8,
}

impl ATrousPass {
    const WORKGROUP_SIZE: (u32, u32, u32) = (8, 8, 1);

    const GBUFFER_BINDING: u32 = 0;
    const RADIANCE_BINDING: u32 = 1;
    const RADIANCE_OUT_BINDING: u32 = 2;
    const SAMPLER_BINDING: u32 = 3;

    pub fn new(
        device: &wgpu::Device,
        source: Option<wgpu::ShaderModuleDescriptor>,
    ) -> Self {
        let frame_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ATrous Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: Self::GBUFFER_BINDING,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: Self::RADIANCE_BINDING,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: Self::RADIANCE_OUT_BINDING,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            format: wgpu::TextureFormat::Rgba32Float,
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: Self::SAMPLER_BINDING,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ATrous Pipeline Layout"),
            bind_group_layouts: &[&frame_bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });

        let shader = match source {
            None => device.create_shader_module(wgpu::include_spirv!(concat!(
                "..",
                path_separator!(),
                "shaders",
                path_separator!(),
                "spirv",
                path_separator!(),
                "atrous.comp.spv"
            ))),
            Some(v) => device.create_shader_module(v),
        };

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ATrous Pipeline"),
            layout: Some(&pipeline_layout),
            entry_point: "main",
            module: &shader,
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            frame_bind_group_layout,
            pipeline,
            count: 3
        }
    }

    pub fn create_frame_bind_groups(
        &self,
        device: &wgpu::Device,
        out_radiance: &wgpu::TextureView,
        gbuffer: &wgpu::TextureView,
        radiance: &wgpu::TextureView,
        sampler: &wgpu::Sampler
    ) -> [wgpu::BindGroup; 2] {
        [
            // TODO: Probably cleaner to use 2 bind groups here
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ATrous Frame Bind Group"),
                layout: &self.frame_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: Self::GBUFFER_BINDING,
                        resource: wgpu::BindingResource::TextureView(gbuffer),
                    },
                    wgpu::BindGroupEntry {
                        binding: Self::RADIANCE_BINDING,
                        resource: wgpu::BindingResource::TextureView(radiance),
                    },
                    wgpu::BindGroupEntry {
                        binding: Self::RADIANCE_OUT_BINDING,
                        resource: wgpu::BindingResource::TextureView(out_radiance),
                    },
                    wgpu::BindGroupEntry {
                        binding: Self::SAMPLER_BINDING,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ATrous Frame Bind Group"),
                layout: &self.frame_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: Self::GBUFFER_BINDING,
                        resource: wgpu::BindingResource::TextureView(gbuffer),
                    },
                    wgpu::BindGroupEntry {
                        binding: Self::RADIANCE_BINDING,
                        resource: wgpu::BindingResource::TextureView(out_radiance),
                    },
                    wgpu::BindGroupEntry {
                        binding: Self::RADIANCE_OUT_BINDING,
                        resource: wgpu::BindingResource::TextureView(radiance),
                    },
                    wgpu::BindGroupEntry {
                        binding: Self::SAMPLER_BINDING,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ]
            })
        ]
    }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindgroups: &[wgpu::BindGroup; 2],
        first_output: &wgpu::Texture,
        retain: &wgpu::Texture,
        size: &(u32, u32, u32),
    ) {
        let workgroups = get_dispatch_size(&size, &Self::WORKGROUP_SIZE);
        for i in 0..self.count as u32 {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ATrous Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);

                let index = i % 2;
                pass.set_bind_group(0, &bindgroups[index as usize], &[]);
                {
                    let data = [(1 as u32) << i];
                    let data = bytemuck::cast_slice(&data);
                    pass.set_push_constants(0, data);
                }
                pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
            }
            if i == 0 {
                encoder.copy_texture_to_texture(wgpu::ImageCopyTexture {
                    texture: &first_output,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },wgpu::ImageCopyTexture {
                    texture: &retain,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                }, retain.size())
            }
        }
    }
}
