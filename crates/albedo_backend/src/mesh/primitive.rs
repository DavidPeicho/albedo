use core::panic;
use std::fmt::Debug;
use std::marker::PhantomData;

use bytemuck::Pod;

use crate::gpu;

fn compute_stride(formats: &[wgpu::VertexFormat]) -> usize {
    let mut stride = 0;
    for format in formats {
        stride += format.size();
    }
    stride as usize
}

fn byte_offset_for(formats: &[wgpu::VertexFormat], element: usize) -> usize {
    // Compute the original byte offset.
    let mut byte_offset = 0;
    for i in 0..element {
        byte_offset += formats[i].size();
    }
    byte_offset as usize
}

fn is_vertex_format_float(format: &wgpu::VertexFormat) -> bool {
    match format {
        wgpu::VertexFormat::Float32
        | wgpu::VertexFormat::Float64
        | wgpu::VertexFormat::Float16x2
        | wgpu::VertexFormat::Float16x4
        | wgpu::VertexFormat::Float32x2
        | wgpu::VertexFormat::Float32x3
        | wgpu::VertexFormat::Float32x4
        | wgpu::VertexFormat::Float64x2
        | wgpu::VertexFormat::Float64x3
        | wgpu::VertexFormat::Float64x4 => true,
        _ => false,
    }
}

fn is_vertex_format_unsigned(format: &wgpu::VertexFormat) -> bool {
    match format {
        wgpu::VertexFormat::Uint8x2
        | wgpu::VertexFormat::Uint8x4
        | wgpu::VertexFormat::Uint16x2
        | wgpu::VertexFormat::Uint16x4
        | wgpu::VertexFormat::Uint32
        | wgpu::VertexFormat::Uint32x2
        | wgpu::VertexFormat::Uint32x3
        | wgpu::VertexFormat::Uint32x4
        | wgpu::VertexFormat::Unorm8x2
        | wgpu::VertexFormat::Unorm8x4
        | wgpu::VertexFormat::Unorm16x2
        | wgpu::VertexFormat::Unorm16x4 => true,
        _ => false,
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
pub struct AttributeId(&'static str);

impl AttributeId {
    pub const POSITION: AttributeId = AttributeId { 0: "POSITION" };
    pub const NORMAL: AttributeId = AttributeId { 0: "NORMAL" };
    pub const TEX_COORDS_0: AttributeId = AttributeId { 0: "TEX_COORDS_0" };
}

pub struct AttributeDescriptor {
    pub id: AttributeId,
    pub format: wgpu::VertexFormat,
}

impl AttributeDescriptor {
    pub fn new<T: Into<String>>(id: AttributeId, format: wgpu::VertexFormat) -> Self {
        Self { id, format }
    }

    pub fn position(format: wgpu::VertexFormat) -> Self {
        Self {
            id: AttributeId::POSITION,
            format,
        }
    }

    pub fn normal(format: wgpu::VertexFormat) -> Self {
        Self {
            id: AttributeId::NORMAL,
            format,
        }
    }

    pub fn tex_coords_0(format: wgpu::VertexFormat) -> Self {
        Self {
            id: AttributeId::TEX_COORDS_0,
            format,
        }
    }
}

enum AttributeData {
    SoA(Vec<Vec<u8>>),
    Interleaved(Vec<u8>),
}

#[derive(Clone)]
pub enum IndexData {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

// Macro to generate a method to lookup an attribute on a
// Primitive.
macro_rules! float_slice_attribute {
    ($name:tt, $type:ty) => {
        pub fn $name<'a>(&'a mut self, index: usize) -> AttributeSlice<'a, $type> {
            let format = self.attribute_format(index);
            if !is_vertex_format_float(&format) {
                panic!("attribute format isn't float"); // @todo: implement Display for VertexFormat
            }
            self.attribute::<$type>(index)
        }
    };
}

// Macro to generate a method to lookup an attribute on a
// Primitive.
macro_rules! unsigned_slice_attribute {
    ($name:tt, $type:ty) => {
        pub fn $name<'a>(&'a mut self, index: usize) -> AttributeSlice<'a, $type> {
            let format = self.attribute_format(index);
            if !is_vertex_format_unsigned(&format) {
                panic!("attribute format isn't unsigned"); // @todo: implement Display for VertexFormat
            }
            self.attribute::<$type>(index)
        }
    };
}

pub struct Primitive {
    data: AttributeData,
    attribute_formats: Vec<wgpu::VertexFormat>,
    attribute_ids: Vec<AttributeId>,
    index_data: Option<IndexData>,
}

impl Primitive {
    pub fn interleaved_with_count(count: u64, descriptors: &[AttributeDescriptor]) -> Self {
        let attribute_ids: Vec<AttributeId> = descriptors.iter().map(|v| v.id).collect();
        let attribute_formats: Vec<wgpu::VertexFormat> =
            descriptors.iter().map(|v| v.format).collect();
        let sizes: Vec<usize> = attribute_formats
            .iter()
            .map(|v| v.size() as usize)
            .collect();
        let byte_count = count as usize * compute_stride(&attribute_formats);
        Self {
            data: AttributeData::Interleaved(vec![0; byte_count]),
            attribute_formats,
            attribute_ids,
            index_data: None,
        }
    }

    pub fn soa_with_count(count: u64, descriptors: &[AttributeDescriptor]) -> Self {
        let attribute_ids = descriptors.iter().map(|v| v.id).collect();
        let attribute_formats: Vec<wgpu::VertexFormat> =
            descriptors.iter().map(|v| v.format).collect();
        let data: Vec<Vec<u8>> = attribute_formats
            .iter()
            .map(|v| Vec::with_capacity(v.size() as usize * count as usize))
            .collect();
        Self {
            data: AttributeData::SoA(data),
            attribute_formats,
            attribute_ids,
            index_data: None,
        }
    }

    pub fn attribute<'a, T: Pod>(&'a mut self, attribute: usize) -> AttributeSlice<'a, T> {
        let byte_size = self.attribute_formats[attribute].size() as usize;
        let request_byte_size = std::mem::size_of::<T>();
        if request_byte_size > byte_size {
            panic!(
                "attribute contains element of {} bytes, but a slice of {} was requested",
                byte_size, request_byte_size
            );
        }
        match &mut self.data {
            AttributeData::Interleaved(v) => AttributeSlice {
                byte_end: v.len(),
                stride: compute_stride(&self.attribute_formats),
                byte_offset: byte_offset_for(&self.attribute_formats, attribute),
                data: v,
                _phantom_data: PhantomData,
            },
            AttributeData::SoA(ref mut soa) => {
                let byte_end = soa[attribute].len();
                AttributeSlice {
                    data: soa[attribute].as_mut(),
                    stride: byte_size,
                    byte_offset: 0,
                    byte_end,
                    _phantom_data: PhantomData,
                }
            }
        }
    }

    float_slice_attribute!(attribute_f32, f32);
    float_slice_attribute!(attribute_f32x2, [f32; 2]);
    float_slice_attribute!(attribute_f32x3, [f32; 3]);
    float_slice_attribute!(attribute_f32x4, [f32; 4]);
    float_slice_attribute!(attribute_f64, f64);
    float_slice_attribute!(attribute_f64x2, [f64; 2]);
    float_slice_attribute!(attribute_f64x3, [f64; 3]);
    float_slice_attribute!(attribute_f64x4, [f64; 4]);
    unsigned_slice_attribute!(attribute_u32, u32);
    unsigned_slice_attribute!(attribute_u32x2, [u32; 2]);
    unsigned_slice_attribute!(attribute_u32x3, [u32; 3]);
    unsigned_slice_attribute!(attribute_u32x4, [u32; 4]);
    // @todo: implement missing attributes

    pub fn attribute_index(&self, id: AttributeId) -> Option<usize> {
        self.attribute_ids.iter().position(|&val| val == id)
    }

    pub fn attribute_format(&self, index: usize) -> wgpu::VertexFormat {
        self.attribute_formats[index]
    }

    pub fn attribute_count(&self) -> usize {
        self.attribute_formats.len()
    }

    pub fn set_indices(&mut self, data: IndexData) {
        self.index_data = Some(data);
    }

    pub fn set_indices_u16(&mut self, data: Vec<u16>) {
        self.index_data = Some(IndexData::U16(data));
    }

    pub fn set_indices_u32(&mut self, data: Vec<u32>) {
        self.index_data = Some(IndexData::U32(data));
    }

    pub fn count(&self) -> usize {
        match &self.data {
            AttributeData::Interleaved(ref v) => v.len() / compute_stride(&self.attribute_formats),
            AttributeData::SoA(ref v) => v[0].len() / self.attribute_formats[0].size() as usize,
        }
    }
}

// @todo: add non-mutable slice.
pub struct AttributeSlice<'a, T: Pod> {
    data: &'a mut [u8],
    byte_offset: usize,
    byte_end: usize,
    stride: usize,
    _phantom_data: PhantomData<&'a T>,
}

impl<'a, T: Pod> AttributeSlice<'a, T> {
    pub fn iter(&'a self) -> AttributeSliceIter<'a, T> {
        AttributeSliceIter {
            slice: self,
            index: 0,
        }
    }

    pub fn set<V: Pod>(&mut self, data: &[V]) {
        let other_stride = std::mem::size_of::<V>();
        if other_stride > std::mem::size_of::<T>() {
            panic!(
                "`data` type is {} bytes, but slice format expected at most {} bytes",
                std::mem::size_of::<V>(),
                std::mem::size_of::<T>()
            );
        }

        let count = data.len();
        if count > self.len() {
            panic!("`data` is larger than the attribute slice");
        }

        let bytes: &[u8] = bytemuck::cast_slice(data);
        for i in 0..count {
            let dst_start = self.byte_offset + self.stride * i;
            let src_start = i * other_stride;
            self.data[dst_start..dst_start + other_stride]
                .copy_from_slice(&bytes[src_start..src_start + other_stride]);
        }
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.stride
    }
}

impl<'a, T> std::ops::Index<usize> for AttributeSlice<'a, T>
where
    T: Pod,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let start_byte = self.byte_offset + self.stride * index;
        if start_byte >= self.byte_end {
            panic!("index ouf of bounds");
        }
        let cast: &[T] =
            bytemuck::cast_slice(&self.data[start_byte..start_byte + std::mem::size_of::<T>()]);
        &cast[0]
    }
}

impl<'a, T> std::ops::IndexMut<usize> for AttributeSlice<'a, T>
where
    T: Pod,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let start_byte = self.byte_offset + self.stride * index;
        if start_byte >= self.byte_end {
            panic!("index ouf of bounds");
        }
        let cast: &mut [T] = bytemuck::cast_slice_mut(
            &mut self.data[start_byte..start_byte + std::mem::size_of::<T>()],
        );
        &mut cast[0]
    }
}

pub struct AttributeSliceIter<'a, T: Pod> {
    slice: &'a AttributeSlice<'a, T>,
    index: usize,
}

impl<'a, T> Iterator for AttributeSliceIter<'a, T>
where
    T: Pod,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.slice.len() {
            return None;
        }
        let index = self.index;
        self.index = self.index + 1;
        Some(self.slice[index])
    }
}

impl gpu::AsVertexBufferLayout for Primitive {
    fn as_vertex_buffer_layout(&self) -> gpu::VertexBufferLayoutBuilder {
        let mut builder = gpu::VertexBufferLayoutBuilder::new(self.attribute_count());
        for i in 0..self.attribute_count() {
            builder = builder.auto_attribute(self.attribute_format(i));
        }
        builder
    }
}

pub struct PrimitiveResourceBuilder<'a> {
    primitive: &'a Primitive,
    descriptor: Option<gpu::BufferInitDescriptor<'a>>,
}

impl<'a> PrimitiveResourceBuilder<'a> {
    pub fn new(primitive: &'a Primitive) -> Self {
        Self {
            primitive,
            descriptor: None,
        }
    }

    pub fn descriptor(mut self, desc: gpu::BufferInitDescriptor<'a>) -> Self {
        self.descriptor = Some(desc);
        self
    }
}

impl<'a> gpu::ResourceBuilder for PrimitiveResourceBuilder<'a> {
    type Resource = gpu::Primitive;

    fn build(self, device: &wgpu::Device) -> Result<Self::Resource, String> {
        let mut attributes = vec![];
        let descriptor = if let Some(desc) = self.descriptor {
            desc
        } else {
            gpu::BufferInitDescriptor::new(Some("Primitive Buffer"), wgpu::BufferUsages::VERTEX)
        };

        let count: usize = self.primitive.count();

        attributes.push(match &self.primitive.data {
            AttributeData::Interleaved(v) => {
                gpu::DynBuffer::new_with_data(device, v, count as u64, Some(descriptor))
            }
            AttributeData::SoA(ref _soa) => {
                todo!("unimplemented")
            }
        });

        // @todo: no unwrap.
        let indices = match self.primitive.index_data.as_ref().unwrap() {
            IndexData::U16(v) => gpu::IndexBuffer::new_with_data_16(device, &v, Some(descriptor)),
            IndexData::U32(v) => gpu::IndexBuffer::new_with_data_32(device, &v, Some(descriptor)),
        };

        Ok(gpu::Primitive {
            attributes,
            indices,
        })
    }
}

impl Debug for IndexData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U16(arg0) => f.debug_list().entries(arg0).finish(),
            Self::U32(arg0) => f.debug_list().entries(arg0).finish(),
        }
    }
}
