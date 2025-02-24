use guillotiere::{size2, Allocation, AtlasAllocator};

use crate::data::packing::Uint24_8;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TextureBlock {
    x: u32,
    y: u32,
    width: u32,
    // Height should be stored in the first 24 bits, and atlas on the last 8 bits.
    height_and_layer: Uint24_8,
}

impl TextureBlock {
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
            height_and_layer: Uint24_8::new(height, layer),
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
        self.height_and_layer.value_24()
    }
    pub fn layer(&self) -> u8 {
        self.height_and_layer.value_8()
    }
}

unsafe impl bytemuck::Pod for TextureBlock {}
unsafe impl bytemuck::Zeroable for TextureBlock {}

pub struct Atlas2D {
    atlas: Vec<AtlasAllocator>,
    blocks: Vec<TextureBlock>,
}

pub struct TextureId(u32);

impl TextureId {
    pub fn new(value: u32) -> Self {
        Self { 0: value }
    }
}

impl Atlas2D {
    fn create_atlas_allocator(size: u32) -> AtlasAllocator {
        AtlasAllocator::new(size2(size as i32, size as i32))
    }

    pub fn new(max_size: u32) -> Self {
        Self {
            atlas: vec![Self::create_atlas_allocator(max_size)],
            blocks: vec![],
        }
    }

    pub fn reserve(&mut self, width: u32, height: u32) -> TextureId {
        let tex_size = size2(width as i32, height as i32);
        for (i, atlas) in self.atlas.iter_mut().enumerate() {
            match atlas.allocate(tex_size) {
                Some(alloc) => return self.reserve_internal(i, alloc),
                _ => (),
            }
        }
        // No atlas found, allocate a new one.
        let layer = {
            let size = self.atlas.first().unwrap().size();
            let layer = self.atlas.len();
            self.atlas.push(AtlasAllocator::new(size));
            layer
        };
        let alloc = self.atlas[layer].allocate(tex_size).unwrap();
        self.reserve_internal(layer, alloc)
    }

    pub fn blocks(&self) -> &[TextureBlock] {
        self.blocks.as_slice()
    }

    pub fn layer_count(&self) -> u32 {
        self.atlas.len() as u32
    }

    pub fn size(&self) -> u32 {
        self.atlas.first().unwrap().size().width as u32
    }

    fn reserve_internal(&mut self, layer: usize, alloc: Allocation) -> TextureId {
        let x = alloc.rectangle.min.x as u32;
        let y = alloc.rectangle.min.y as u32;
        let width = alloc.rectangle.width() as u32;
        let height = alloc.rectangle.height() as u32;

        let id = self.blocks.len() as u32;
        self.blocks
            .push(TextureBlock::new(layer as u8, x, y, width, height));

        TextureId { 0: id }
    }
}

fn rgba_bytes_per_row(width: u32) -> u32 {
    4 * width
}

pub struct TextureAtlas {
    pub atlas: Atlas2D,
    texture_view: wgpu::TextureView,
    texture: wgpu::Texture,
    texture_blocks_view: wgpu::TextureView,
    texture_blocks: wgpu::Texture,
}

impl TextureAtlas {
    pub fn from_atlas2d(
        device: &wgpu::Device,
        atlas: Atlas2D,
        max_texture_count: Option<u32>,
    ) -> Self {
        let atlas_size = atlas.size();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture Atlas"),
            size: wgpu::Extent3d {
                width: atlas_size,
                height: atlas_size,
                depth_or_array_layers: atlas.layer_count(),
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let max_texture_count = max_texture_count.unwrap_or(atlas.blocks.len() as u32);
        let max_texture_count = u32::max(max_texture_count, 1);
        let texture_blocks = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture Blocks"),
            size: wgpu::Extent3d {
                width: max_texture_count,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D1,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        Self {
            atlas,

            texture_view: texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            }),
            texture,
            texture_blocks_view: texture_blocks
                .create_view(&wgpu::TextureViewDescriptor::default()),
            texture_blocks,
        }
    }

    pub fn new(device: &wgpu::Device, size: u32, max_texture_count: u32) -> Self {
        let atlas = Atlas2D::new(size);
        Self::from_atlas2d(device, atlas, Some(max_texture_count))
    }

    pub fn from_limits(device: &wgpu::Device) -> Self {
        let limits = device.limits();
        Self::new(
            device,
            limits.max_texture_dimension_1d,
            limits.max_texture_dimension_1d,
        )
    }

    pub fn upload(&self, queue: &wgpu::Queue, id: TextureId, data: &[u8]) {
        let block = self.atlas.blocks[id.0 as usize];

        // Write texture data.
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                aspect: wgpu::TextureAspect::All,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: block.x,
                    y: block.y,
                    z: block.layer() as u32,
                },
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(rgba_bytes_per_row(block.width())),
                rows_per_image: Some(block.height()),
            },
            wgpu::Extent3d {
                width: block.width(),
                height: block.height(),
                depth_or_array_layers: 1,
            },
        );

        // Write texture block
        let block_data = bytemuck::bytes_of(&block);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture_blocks,
                aspect: wgpu::TextureAspect::All,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: id.0,
                    y: 0,
                    z: 0,
                },
            },
            block_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16), // RGBA, 4 bytes each
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    // TODO: Batch upload.

    pub fn texture(&self) -> &wgpu::TextureView {
        &self.texture_view
    }

    pub fn texture_blocks(&self) -> &wgpu::TextureView {
        &self.texture_blocks_view
    }

    pub fn blocks(&self) -> &[TextureBlock] {
        &self.atlas.blocks
    }
}
