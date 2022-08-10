#[path = "../framework.rs"]
mod framework;

use bytemuck::{Pod, Zeroable};
use std::{
    borrow::Cow,
    f32::consts,
    future::Future,
    mem,
    pin::Pin,
    task,
    time::{Duration, Instant},
};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 4],
    _tex_coord: [f32; 2],
}

fn vertex(pos: [i8; 3], tc: [i8; 2]) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
        _tex_coord: [tc[0] as f32, tc[1] as f32],
    }
}

fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1, -1, 1], [0, 0]),
        vertex([1, -1, 1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([-1, 1, 1], [0, 1]),
        // bottom (0, 0, -1)
        vertex([-1, 1, -1], [1, 0]),
        vertex([1, 1, -1], [0, 0]),
        vertex([1, -1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // right (1, 0, 0)
        vertex([1, -1, -1], [0, 0]),
        vertex([1, 1, -1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([1, -1, 1], [0, 1]),
        // left (-1, 0, 0)
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, 1, 1], [0, 0]),
        vertex([-1, 1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // front (0, 1, 0)
        vertex([1, 1, -1], [1, 0]),
        vertex([-1, 1, -1], [0, 0]),
        vertex([-1, 1, 1], [0, 1]),
        vertex([1, 1, 1], [1, 1]),
        // back (0, -1, 0)
        vertex([1, -1, 1], [0, 0]),
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, -1, -1], [1, 1]),
        vertex([1, -1, -1], [0, 1]),
    ];

    let index_data: &[u16] = &[
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

fn create_texels(size: usize) -> Vec<u8> {
    (0..size * size)
        .map(|id| {
            // get high five for recognizing this ;)
            let cx = 3.0 * (id % size) as f32 / (size - 1) as f32 - 2.0;
            let cy = 2.0 * (id / size) as f32 / (size - 1) as f32 - 1.0;
            let (mut x, mut y, mut count) = (cx, cy, 0);
            while count < 0xFF && x * x + y * y < 4.0 {
                let old_x = x;
                x = x * x - y * y + cx;
                y = 2.0 * old_x * y + cy;
                count += 1;
            }
            count
        })
        .collect()
}

/// A wrapper for `pop_error_scope` futures that panics if an error occurs.
///
/// Given a future `inner` of an `Option<E>` for some error type `E`,
/// wait for the future to be ready, and panic if its value is `Some`.
///
/// This can be done simpler with `FutureExt`, but we don't want to add
/// a dependency just for this small case.
struct ErrorFuture<F> {
    inner: F,
}
impl<F: Future<Output = Option<wgpu::Error>>> Future for ErrorFuture<F> {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> task::Poll<()> {
        let inner = unsafe { self.map_unchecked_mut(|me| &mut me.inner) };
        inner.poll(cx).map(|error| {
            if let Some(e) = error {
                panic!("Rendering {}", e);
            }
        })
    }
}

struct Example {
    write_wm_buffer_count: i32,
    write_wm_buffer_duration: Duration,
    write_wm_buffer_max_duration: Duration,
    write_wm_buffer_min_duration: Duration,
    last_print_wm_buffer: Instant,

    during_render_ms: f64,
    last_render_time: Instant,

    y_v: f32,
    y_pos: f32,

    index_count: usize,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,

    view_matrix: wgpu::Buffer,
    project_matrix: wgpu::Buffer,

    bind_group_camera: wgpu::BindGroup,

    world_matrix: wgpu::Buffer,

    test_matrix: wgpu::Buffer,
    test_matrix_const: [glam::Mat4; 1024],

    bind_group_cube: wgpu::BindGroup,

    pipeline: wgpu::RenderPipeline,
    pipeline_wire: Option<wgpu::RenderPipeline>,
}

impl Example {
    fn create_world_matrix(x: f32, y: f32, z: f32) -> glam::Mat4 {
        glam::Mat4::from_translation(glam::Vec3::new(x, y, z))
    }

    fn create_view_matrix() -> glam::Mat4 {
        glam::Mat4::look_at_rh(
            glam::Vec3::new(1.5f32, -5.0, 3.0),
            glam::Vec3::ZERO,
            glam::Vec3::Z,
        )
    }

    fn create_project_matrix(aspect_ratio: f32) -> glam::Mat4 {
        glam::Mat4::perspective_rh(consts::FRAC_PI_4, aspect_ratio, 0.1, 100.0)
    }

    fn update_time(&mut self) {
        let now = Instant::now();

        self.during_render_ms =
            now.duration_since(self.last_render_time).as_micros() as f64 / 1000.0;

        self.last_render_time = now;
    }

    fn update_position(&mut self, queue: &wgpu::Queue) {
        self.y_pos += self.y_v * self.during_render_ms as f32;

        if self.y_pos > 10.0 {
            self.y_v *= -1.0;
            self.y_pos = 10.0;
        }

        if self.y_pos < 0.0 {
            self.y_v *= -1.0;
            self.y_pos = 0.0;
        }
        
        self.write_wm_buffer_count += 1;
        
        let mut size = 0;
        let mut time = Duration::default();
        {
            let world_matrix = Self::create_world_matrix(0.0, self.y_pos, 0.0);
            let world_matrix_contents = bytemuck::cast_slice(world_matrix.as_ref());
            size += world_matrix_contents.len();
    
            let begin = Instant::now();
            Self::write_buffer_no_print(queue, &self.world_matrix, 0, world_matrix_contents);
            let end = Instant::now();
            let t = end - begin;
            time += t;
        }

        {
            self.test_matrix_const[49] = Self::create_world_matrix(0.0, self.y_pos, 0.0);

            let test_matrix_slice = unsafe {
                let slice = self.test_matrix_const.as_slice();
                std::mem::transmute::<&[glam::Mat4], &[[f32; 16]]>(slice)
            };

            let test_matrix_contents = bytemuck::cast_slice(test_matrix_slice);
            size += test_matrix_contents.len();
    
            let begin = Instant::now();
            Self::write_buffer_no_print(queue, &self.test_matrix, 0, test_matrix_contents);
            let end = Instant::now();
            let t = end - begin;
            time += t;
        }
        
        if self.write_wm_buffer_max_duration < time {
            self.write_wm_buffer_max_duration = time;
        }
        if self.write_wm_buffer_min_duration > time || self.write_wm_buffer_min_duration == Duration::default() {
            self.write_wm_buffer_min_duration = time;
        }

        self.write_wm_buffer_duration += time;
        
        if Instant::now().duration_since(self.last_print_wm_buffer).as_millis() >= 1000 {
            println!(
                "write_buffer position: size = {}B, count = {}, avg_time = {:?}, max_time: {:?}, min_time = {:?}",
                size,
                self.write_wm_buffer_count,
                self.write_wm_buffer_duration / self.write_wm_buffer_count as u32,
                self.write_wm_buffer_max_duration,
                self.write_wm_buffer_min_duration,    
            );
        
            self.write_wm_buffer_max_duration = Duration::default();
            self.write_wm_buffer_min_duration = Duration::default();
        
            self.write_wm_buffer_count = 0;
            self.write_wm_buffer_duration = Duration::default();
            self.last_print_wm_buffer = Instant::now();
        }
    }

    fn create_shader_module(
        device: &wgpu::Device,
        label: &str,
        shader_source: &str,
    ) -> wgpu::ShaderModule {
        let begin = Instant::now();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
        });

        let end = Instant::now();
        println!("{}: create_shader_module, time = {:?}", label, end - begin);

        shader
    }

    fn create_buffer(
        device: &wgpu::Device,
        usage: wgpu::BufferUsages,
        label: &str,
        size: usize,
    ) -> wgpu::Buffer {
        let begin = Instant::now();
        let r = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        });
        let end = Instant::now();

        println!(
            "{}: create_buffer, size = {} B, time = {:?}",
            label,
            size,
            end - begin
        );

        r
    }

    fn create_buffer_init(
        device: &wgpu::Device,
        usage: wgpu::BufferUsages,
        label: &str,
        contents: &[u8],
    ) -> wgpu::Buffer {
        let begin = Instant::now();
        let r = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage,
        });
        let end = Instant::now();
        println!(
            "{}: create_buffer_init, size = {} B, time = {:?}",
            label,
            contents.len(),
            end - begin
        );

        r
    }

    fn write_buffer_no_print(
        queue: &wgpu::Queue,
        buf: &wgpu::Buffer,
        offset: u64,
        contents: &[u8],
    ) {
        queue.write_buffer(buf, offset, contents);
    }

    fn write_buffer(
        queue: &wgpu::Queue,
        buf: &wgpu::Buffer,
        label: &str,
        offset: u64,
        contents: &[u8],
    ) {
        let begin = Instant::now();

        Self::write_buffer_no_print(queue, buf, offset, contents);

        let end = Instant::now();

        println!(
            "{}: write_buffer, offset = {} B, size = {} B, time = {:?}",
            label,
            offset,
            contents.len(),
            end - begin
        );
    }

    fn create_texture(
        device: &wgpu::Device,
        usage: wgpu::TextureUsages,
        label: &str,
        size: wgpu::Extent3d,
    ) -> wgpu::Texture {
        let begin = Instant::now();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage,
        });
        let end = Instant::now();
        println!(
            "{}: create_texture, size = {:?}, time = {:?}",
            label,
            size,
            end - begin
        );

        texture
    }

    fn create_texture_view(label: &str, texture: &wgpu::Texture) -> wgpu::TextureView {
        let begin = Instant::now();
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let end = Instant::now();
        println!("{}: create_view, time = {:?}", label, end - begin);

        view
    }

    fn write_texture(
        queue: &wgpu::Queue,
        label: &str,
        texture: &wgpu::Texture,
        extent: wgpu::Extent3d,
        size: u32,
        data: &[u8],
    ) {
        let begin = Instant::now();

        queue.write_texture(
            texture.as_image_copy(),
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(size).unwrap()),
                rows_per_image: None,
            },
            extent,
        );

        let end = Instant::now();
        println!(
            "{}: write_texture, size = {:?}, time = {:?}",
            label,
            size,
            end - begin
        );
    }
}

impl framework::Example for Example {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::POLYGON_MODE_LINE
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let vertex_size = mem::size_of::<Vertex>();
        let (vertex_data, index_data) = create_vertices();

        let contents = bytemuck::cast_slice(&vertex_data);

        let label = "Vertex Buffer";
        let usage = wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        let vertex_buf = Self::create_buffer(device, usage, label, contents.len());
        Self::write_buffer(queue, &vertex_buf, label, 0, contents);

        let vertex_buf =
            Self::create_buffer_init(device, wgpu::BufferUsages::VERTEX, label, contents);

        let label = "Index Buffer";
        let contents = bytemuck::cast_slice(&index_data);
        let index_buf =
            Self::create_buffer_init(device, wgpu::BufferUsages::INDEX, label, contents);

        let usage = wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST;
        let index_buf = Self::create_buffer(device, usage, label, contents.len());
        Self::write_buffer(queue, &index_buf, label, 0, contents);

        // for i in 0..5 {
        //     let label = format!("Test Buffer {}", i + 1);
        //     let mut contents = Vec::with_capacity(1024 * 64 - 28);
        //     for i in 0..contents.capacity() {
        //         contents.push(i as u8);
        //     }
        //     let contents = contents.as_slice();

        //     let test_buf = Self::create_buffer_init(
        //         device,
        //         wgpu::BufferUsages::VERTEX,
        //         label.as_str(),
        //         contents,
        //     );

        //     let usage = wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        //     let test_buf = Self::create_buffer(device, usage, label.as_str(), contents.len());
        //     Self::write_buffer(queue, &test_buf, label.as_str(), 0, contents);
        // }

        let begin = Instant::now();

        let bind_group_layout_camera =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(64),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(64),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let end = Instant::now();
        println!(
            "bind_group_layout_camera create_bind_group_layout, time = {:?}",
            end - begin
        );

        let begin = Instant::now();

        let bind_group_layout_cube =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(64),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(16 * 4 * 50),
                        },
                        count: None,
                    },
                ],
            });

        let end = Instant::now();
        println!(
            "bind_group_layout_cube create_bind_group_layout, time = {:?}",
            end - begin
        );

        let begin = Instant::now();

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout_camera, &bind_group_layout_cube],
            push_constant_ranges: &[],
        });

        let end = Instant::now();
        println!(
            "pipeline_layout create_pipeline_layout, time = {:?}",
            end - begin
        );

        // Create the texture
        let size = 256u32;
        let texels = create_texels(size as usize);
        let extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        };

        let label = "texture";
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let texture = Self::create_texture(device, usage, label, extent);
        let texture_view = Self::create_texture_view(label, &texture);
        Self::write_texture(queue, label, &texture, extent, size, &texels);

        let view_matrix = Self::create_view_matrix();
        let label = "View_Matrix";
        let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let contents = bytemuck::cast_slice(view_matrix.as_ref());
        let view_matrix = Self::create_buffer_init(device, usage, label, contents);

        let project_matrix =
            Self::create_project_matrix(config.width as f32 / config.height as f32);
        let label = "Project_Matrix";
        let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let contents = bytemuck::cast_slice(project_matrix.as_ref());
        let project_matrix = Self::create_buffer_init(device, usage, label, contents);

        let y_pos = 0.0;
        let world_matrix = Self::create_world_matrix(0.0, y_pos, 0.0);
        let a = world_matrix.as_ref();
        let label = "World_Matrix";
        let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let contents = bytemuck::cast_slice(a);
        let world_matrix = Self::create_buffer_init(device, usage, label, contents);

        let label = "Test_Matrix";
        let test_matrix_const = [glam::Mat4::default(); 1024];
        let test_matrix = {
            let test_matrix_slice = unsafe {
                let slice = test_matrix_const.as_slice();
                std::mem::transmute::<&[glam::Mat4], &[[f32; 16]]>(slice)
            };
            let contents = bytemuck::cast_slice(test_matrix_slice);
            Self::create_buffer_init(device, usage, label, contents)
        };

        let begin = Instant::now();
        let bind_group_camera = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout_camera,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view_matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: project_matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
            ],
            label: None,
        });

        let end = Instant::now();
        println!(
            "bind_group_camera create_bind_group, time = {:?}",
            end - begin
        );

        let begin = Instant::now();

        let test_matrix_resource = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &test_matrix,
            offset: 0,
            size: std::num::NonZeroU64::new(64 * 4 * 50),
        });

        let bind_group_cube = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout_cube,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: world_matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: test_matrix_resource,
                },
            ],
            label: None,
        });

        let end = Instant::now();
        println!(
            "bind_group_cube create_bind_group, time = {:?}",
            end - begin
        );

        let shader = Self::create_shader_module(device, "Shader", include_str!("shader.wgsl"));

        let vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: vertex_size as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 4 * 4,
                    shader_location: 1,
                },
            ],
        }];

        let begin = Instant::now();

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let end = Instant::now();
        println!("create_render_pipeline, time = {:?}", end - begin);

        let begin = Instant::now();

        let pipeline_wire = if device.features().contains(wgpu::Features::POLYGON_MODE_LINE) {
            let pipeline_wire = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &vertex_buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_wire",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                operation: wgpu::BlendOperation::Add,
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Line,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });
            Some(pipeline_wire)
        } else {
            None
        };

        let end = Instant::now();
        println!(
            "pipeline_wire create_render_pipeline, time = {:?}",
            end - begin
        );

        Example {
            during_render_ms: 0.0,
            last_render_time: Instant::now(),

            test_matrix,
            test_matrix_const,

            write_wm_buffer_count: 0,
            write_wm_buffer_duration: Duration::default(),
            
            write_wm_buffer_max_duration: Duration::default(),
            write_wm_buffer_min_duration: Duration::default(),

            last_print_wm_buffer: Instant::now(),

            y_v: 0.001,
            y_pos,

            vertex_buf,
            index_buf,
            index_count: index_data.len(),

            bind_group_camera,

            world_matrix,
            view_matrix,
            project_matrix,

            bind_group_cube,

            pipeline,
            pipeline_wire,
        }
    }

    fn update(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _event: winit::event::WindowEvent,
    ) {
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let project_matrix =
            Self::create_project_matrix(config.width as f32 / config.height as f32);

        queue.write_buffer(
            &self.project_matrix,
            0,
            bytemuck::cast_slice(project_matrix.as_ref()),
        );
    }

    fn render(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spawner: &framework::Spawner,
    ) {
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        self.update_time();

        self.update_position(queue);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            
            rpass.push_debug_group("Prepare data for draw.");
            
            rpass.set_pipeline(&self.pipeline);

            rpass.set_bind_group(0, &&self.bind_group_camera, &[]);
            rpass.set_bind_group(1, &&self.bind_group_cube, &[]);

            rpass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
            rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            
            rpass.pop_debug_group();

            rpass.insert_debug_marker("Draw!");
            rpass.draw_indexed(0..self.index_count as u32, 0, 0..1);
            if let Some(ref pipe) = self.pipeline_wire {
                rpass.set_pipeline(pipe);
                rpass.draw_indexed(0..self.index_count as u32, 0, 0..1);
            }
        }

        queue.submit(Some(encoder.finish()));

        // If an error occurs, report it and panic.
        spawner.spawn_local(ErrorFuture {
            inner: device.pop_error_scope(),
        });
    }

    fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
    }

    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::empty(),
            shader_model: wgpu::ShaderModel::Sm5,
            ..wgpu::DownlevelCapabilities::default()
        }
    }

    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_webgl2_defaults() // These downlevel limits will allow the code to run on all possible hardware
    }
}

fn main() {
    framework::run::<Example>("cube");
}

#[test]
fn cube() {
    framework::test::<Example>(framework::FrameworkRefTest {
        image_path: "/examples/cube/screenshot.png",
        width: 1024,
        height: 768,
        optional_features: wgpu::Features::default(),
        base_test_parameters: framework::test_common::TestParameters::default(),
        tolerance: 1,
        max_outliers: 500, // Bounded by rpi4
    });
}

#[test]
fn cube_lines() {
    framework::test::<Example>(framework::FrameworkRefTest {
        image_path: "/examples/cube/screenshot-lines.png",
        width: 1024,
        height: 768,
        optional_features: wgpu::Features::POLYGON_MODE_LINE,
        base_test_parameters: framework::test_common::TestParameters::default(),
        tolerance: 2,
        max_outliers: 600, // Bounded by rpi4 on GL
    });
}
