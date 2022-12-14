use std::future::Future;
#[cfg(target_arch = "wasm32")]
use std::str::FromStr;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_sys::{ImageBitmapRenderingContext, OffscreenCanvas};
use winit::{
    event::{self, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

#[allow(dead_code)]
pub fn cast_slice<T>(data: &[T]) -> &[u8] {
    use std::{mem::size_of, slice::from_raw_parts};

    unsafe { from_raw_parts(data.as_ptr() as *const u8, data.len() * size_of::<T>()) }
}

#[allow(dead_code)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

pub trait Example: 'static + Sized {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::empty()
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
    fn init(
        config: &wgpu::SurfaceConfiguration,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self;
    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    );
    fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, event: WindowEvent);
    fn render(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spawner: &Spawner,
    );
}

struct Setup {
    window: winit::window::Window,
    event_loop: EventLoop<()>,
    instance: wgpu::Instance,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    #[cfg(target_arch = "wasm32")]
    offscreen_canvas_setup: Option<OffscreenCanvasSetup>,
}

#[cfg(target_arch = "wasm32")]
struct OffscreenCanvasSetup {
    offscreen_canvas: OffscreenCanvas,
    bitmap_renderer: ImageBitmapRenderingContext,
}

async fn setup<E: Example>(title: &str) -> Setup {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    };

    let event_loop = EventLoop::new();
    let mut builder = winit::window::WindowBuilder::new();
    builder = builder.with_title(title);
    #[cfg(windows_OFF)] // TODO
    {
        use winit::platform::windows::WindowBuilderExtWindows;
        builder = builder.with_no_redirection_bitmap(true);
    }
    let window = builder.build(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        let query_string = web_sys::window().unwrap().location().search().unwrap();
        let level: log::Level = parse_url_query_string(&query_string, "RUST_LOG")
            .and_then(|x| x.parse().ok())
            .unwrap_or(log::Level::Error);
        console_log::init_with_level(level).expect("could not initialize logger");
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
    }

    #[cfg(target_arch = "wasm32")]
    let mut offscreen_canvas_setup: Option<OffscreenCanvasSetup> = None;
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowExtWebSys;

        let query_string = web_sys::window().unwrap().location().search().unwrap();
        if let Some(offscreen_canvas_param) =
            parse_url_query_string(&query_string, "offscreen_canvas")
        {
            if FromStr::from_str(offscreen_canvas_param) == Ok(true) {
                log::info!("Creating OffscreenCanvasSetup");

                let offscreen_canvas =
                    OffscreenCanvas::new(1024, 768).expect("couldn't create OffscreenCanvas");

                let bitmap_renderer = window
                    .canvas()
                    .get_context("bitmaprenderer")
                    .expect("couldn't create ImageBitmapRenderingContext (Result)")
                    .expect("couldn't create ImageBitmapRenderingContext (Option)")
                    .dyn_into::<ImageBitmapRenderingContext>()
                    .expect("couldn't convert into ImageBitmapRenderingContext");

                offscreen_canvas_setup = Some(OffscreenCanvasSetup {
                    offscreen_canvas,
                    bitmap_renderer,
                })
            }
        }
    };

    log::info!("Initializing the surface...");

    // let backend = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);

    let backend = wgpu::Backends::VULKAN;
    // let backend = wgpu::Backends::DX12;
    // let backend = wgpu::Backends::GL;

    let instance = wgpu::Instance::new(backend);
    let (size, surface) = unsafe {
        let size = window.inner_size();

        #[cfg(not(target_arch = "wasm32"))]
        let surface = instance.create_surface(&window);
        #[cfg(target_arch = "wasm32")]
        let surface = {
            if let Some(offscreen_canvas_setup) = &offscreen_canvas_setup {
                log::info!("Creating surface from OffscreenCanvas");
                instance
                    .create_surface_from_offscreen_canvas(&offscreen_canvas_setup.offscreen_canvas)
            } else {
                instance.create_surface(&window)
            }
        };

        (size, surface)
    };

    // let power_preference = wgpu::PowerPreference::LowPower;
    let power_preference = wgpu::PowerPreference::HighPerformance;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .expect("No suitable GPU adapters found on the system!");

    println!("\n========= Adapter info = {:?}\n", adapter.get_info());

    println!("========= Adapter Features: {:?}\n", adapter.features());

    println!("========= Adapter Limits: {:?}\n", adapter.limits());

    println!("========= surface.get_supported_formats = {:?}\n", surface.get_supported_formats(&adapter));

    print_unsuppported_texture_formats(&adapter);

    #[cfg(not(target_arch = "wasm32"))]
    {
        // let adapter_info = adapter.get_info();
        // println!("Using {} ({:?})", adapter_info.name, adapter_info.backend);
    }

    let optional_features = E::optional_features();
    let required_features = E::required_features();
    let adapter_features = adapter.features();
    assert!(
        adapter_features.contains(required_features),
        "Adapter does not support required features for this example: {:?}",
        required_features - adapter_features
    );

    let required_downlevel_capabilities = E::required_downlevel_capabilities();
    let downlevel_capabilities = adapter.get_downlevel_capabilities();
    assert!(
        downlevel_capabilities.shader_model >= required_downlevel_capabilities.shader_model,
        "Adapter does not support the minimum shader model required to run this example: {:?}",
        required_downlevel_capabilities.shader_model
    );
    assert!(
        downlevel_capabilities
            .flags
            .contains(required_downlevel_capabilities.flags),
        "Adapter does not support the downlevel capabilities required to run this example: {:?}",
        required_downlevel_capabilities.flags - downlevel_capabilities.flags
    );

    println!("=============== downlevel_capabilities = {:?}\n", downlevel_capabilities);

    // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the surface.
    let needed_limits = E::required_limits().using_resolution(adapter.limits());

    let trace_dir = std::env::var("WGPU_TRACE");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: (optional_features & adapter_features) | required_features,
                limits: needed_limits,
            },
            trace_dir.ok().as_ref().map(std::path::Path::new),
        )
        .await
        .expect("Unable to find a suitable GPU adapter!");

    Setup {
        window,
        event_loop,
        instance,
        size,
        surface,
        adapter,
        device,
        queue,
        #[cfg(target_arch = "wasm32")]
        offscreen_canvas_setup,
    }
}

fn start<E: Example>(
    #[cfg(not(target_arch = "wasm32"))] Setup {
        window,
        event_loop,
        instance,
        size,
        surface,
        adapter,
        device,
        queue,
    }: Setup,
    #[cfg(target_arch = "wasm32")] Setup {
        window,
        event_loop,
        instance,
        size,
        surface,
        adapter,
        device,
        queue,
        offscreen_canvas_setup,
    }: Setup,
) {
    let spawner = Spawner::new();
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface.get_supported_formats(&adapter)[0],
        width: size.width,
        height: size.height,
        // present_mode: wgpu::PresentMode::Fifo,
        present_mode: wgpu::PresentMode::Mailbox,
    };
    surface.configure(&device, &config);

    log::info!("Initializing the example...");
    let mut example = E::init(&config, &adapter, &device, &queue);

    #[cfg(not(target_arch = "wasm32"))]
    let mut last_frame_inst = Instant::now();
    #[cfg(not(target_arch = "wasm32"))]
    let (mut frame_count, mut accum_time) = (0, 0.0);

    log::info!("Entering render loop...");
    event_loop.run(move |event, _, control_flow| {
        let _ = (&instance, &adapter); // force ownership by the closure
        *control_flow = if cfg!(feature = "metal-auto-capture") {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };
        match event {
            event::Event::RedrawEventsCleared => {
                #[cfg(not(target_arch = "wasm32"))]
                spawner.run_until_stalled();

                window.request_redraw();
            }
            event::Event::WindowEvent {
                event:
                    WindowEvent::Resized(size)
                    | WindowEvent::ScaleFactorChanged {
                        new_inner_size: &mut size,
                        ..
                    },
                ..
            } => {
                log::info!("Resizing to {:?}", size);
                config.width = size.width.max(1);
                config.height = size.height.max(1);
                example.resize(&config, &device, &queue);
                surface.configure(&device, &config);
            }
            event::Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(event::VirtualKeyCode::Escape),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                }
                | WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                #[cfg(not(target_arch = "wasm32"))]
                WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(event::VirtualKeyCode::R),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    println!("{:#?}", instance.generate_report());
                }
                _ => {
                    example.update(&device, &queue, event);
                }
            },
            event::Event::RedrawRequested(_) => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    accum_time += last_frame_inst.elapsed().as_secs_f32();
                    last_frame_inst = Instant::now();
                    frame_count += 1;
                    if accum_time > 1.0 {
                        println!(
                            "Avg frame time: {} ms",
                            accum_time * 1000.0 / frame_count as f32
                        );
                        accum_time = 0.0;
                        frame_count = 0;
                    }
                }

                let frame = match surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(_) => {
                        surface.configure(&device, &config);
                        surface
                            .get_current_texture()
                            .expect("Failed to acquire next surface texture!")
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                example.render(&view, &device, &queue, &spawner);

                frame.present();

                #[cfg(target_arch = "wasm32")]
                {
                    if let Some(offscreen_canvas_setup) = &offscreen_canvas_setup {
                        let image_bitmap = offscreen_canvas_setup
                            .offscreen_canvas
                            .transfer_to_image_bitmap()
                            .expect("couldn't transfer offscreen canvas to image bitmap.");
                        offscreen_canvas_setup
                            .bitmap_renderer
                            .transfer_from_image_bitmap(&image_bitmap);

                        log::info!("Transferring OffscreenCanvas to ImageBitmapRenderer");
                    }
                }
            }
            _ => {}
        }
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub struct Spawner<'a> {
    executor: async_executor::LocalExecutor<'a>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> Spawner<'a> {
    fn new() -> Self {
        Self {
            executor: async_executor::LocalExecutor::new(),
        }
    }

    #[allow(dead_code)]
    pub fn spawn_local(&self, future: impl Future<Output = ()> + 'a) {
        self.executor.spawn(future).detach();
    }

    fn run_until_stalled(&self) {
        while self.executor.try_tick() {}
    }
}

#[cfg(target_arch = "wasm32")]
pub struct Spawner {}

#[cfg(target_arch = "wasm32")]
impl Spawner {
    fn new() -> Self {
        Self {}
    }

    #[allow(dead_code)]
    pub fn spawn_local(&self, future: impl Future<Output = ()> + 'static) {
        wasm_bindgen_futures::spawn_local(future);
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run<E: Example>(title: &str) {
    let setup = pollster::block_on(setup::<E>(title));
    start::<E>(setup);
}

#[cfg(target_arch = "wasm32")]
pub fn run<E: Example>(title: &str) {
    use wasm_bindgen::{prelude::*, JsCast};

    let title = title.to_owned();
    wasm_bindgen_futures::spawn_local(async move {
        let setup = setup::<E>(&title).await;
        let start_closure = Closure::once_into_js(move || start::<E>(setup));

        // make sure to handle JS exceptions thrown inside start.
        // Otherwise wasm_bindgen_futures Queue would break and never handle any tasks again.
        // This is required, because winit uses JS exception for control flow to escape from `run`.
        if let Err(error) = call_catch(&start_closure) {
            let is_control_flow_exception = error.dyn_ref::<js_sys::Error>().map_or(false, |e| {
                e.message().includes("Using exceptions for control flow", 0)
            });

            if !is_control_flow_exception {
                web_sys::console::error_1(&error);
            }
        }

        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(catch, js_namespace = Function, js_name = "prototype.call.call")]
            fn call_catch(this: &JsValue) -> Result<(), JsValue>;
        }
    });
}

#[cfg(target_arch = "wasm32")]
/// Parse the query string as returned by `web_sys::window()?.location().search()?` and get a
/// specific key out of it.
pub fn parse_url_query_string<'a>(query: &'a str, search_key: &str) -> Option<&'a str> {
    let query_string = query.strip_prefix('?')?;

    for pair in query_string.split('&') {
        let mut pair = pair.split('=');
        let key = pair.next()?;
        let value = pair.next()?;

        if key == search_key {
            return Some(value);
        }
    }

    None
}

// This allows treating the framework as a standalone example,
// thus avoiding listing the example names in `Cargo.toml`.
#[allow(dead_code)]
fn main() {}

fn print_unsuppported_texture_formats(adapter: &wgpu::Adapter) {
    let formats = [
        wgpu::TextureFormat::R8Unorm,
        wgpu::TextureFormat::R8Snorm,
        wgpu::TextureFormat::R8Uint,
        wgpu::TextureFormat::R8Sint,
        wgpu::TextureFormat::R16Uint,
        wgpu::TextureFormat::R16Sint,
        wgpu::TextureFormat::R16Unorm,
        wgpu::TextureFormat::R16Snorm,
        wgpu::TextureFormat::R16Float,
        wgpu::TextureFormat::Rg8Unorm,
        wgpu::TextureFormat::Rg8Snorm,
        wgpu::TextureFormat::Rg8Uint,
        wgpu::TextureFormat::Rg8Sint,
        wgpu::TextureFormat::R32Uint,
        wgpu::TextureFormat::R32Sint,
        wgpu::TextureFormat::R32Float,
        wgpu::TextureFormat::Rg16Uint,
        wgpu::TextureFormat::Rg16Sint,
        wgpu::TextureFormat::Rg16Unorm,
        wgpu::TextureFormat::Rg16Snorm,
        wgpu::TextureFormat::Rg16Float,
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Rgba8Snorm,
        wgpu::TextureFormat::Rgba8Uint,
        wgpu::TextureFormat::Rgba8Sint,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Bgra8UnormSrgb,
        wgpu::TextureFormat::Rgb10a2Unorm,
        wgpu::TextureFormat::Rg11b10Float,
        wgpu::TextureFormat::Rg32Uint,
        wgpu::TextureFormat::Rg32Sint,
        wgpu::TextureFormat::Rg32Float,
        wgpu::TextureFormat::Rgba16Uint,
        wgpu::TextureFormat::Rgba16Sint,
        wgpu::TextureFormat::Rgba16Unorm,
        wgpu::TextureFormat::Rgba16Snorm,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Rgba32Uint,
        wgpu::TextureFormat::Rgba32Sint,
        wgpu::TextureFormat::Rgba32Float,
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureFormat::Depth32FloatStencil8,
        wgpu::TextureFormat::Depth24Plus,
        wgpu::TextureFormat::Depth24PlusStencil8,
        wgpu::TextureFormat::Depth24UnormStencil8,
        wgpu::TextureFormat::Rgb9e5Ufloat,
        wgpu::TextureFormat::Bc1RgbaUnorm,
        wgpu::TextureFormat::Bc1RgbaUnormSrgb,
        wgpu::TextureFormat::Bc2RgbaUnorm,
        wgpu::TextureFormat::Bc2RgbaUnormSrgb,
        wgpu::TextureFormat::Bc3RgbaUnorm,
        wgpu::TextureFormat::Bc3RgbaUnormSrgb,
        wgpu::TextureFormat::Bc4RUnorm,
        wgpu::TextureFormat::Bc4RSnorm,
        wgpu::TextureFormat::Bc5RgUnorm,
        wgpu::TextureFormat::Bc5RgSnorm,
        wgpu::TextureFormat::Bc6hRgbUfloat,
        wgpu::TextureFormat::Bc6hRgbSfloat,
        wgpu::TextureFormat::Bc7RgbaUnorm,
        wgpu::TextureFormat::Bc7RgbaUnormSrgb,
        wgpu::TextureFormat::Etc2Rgb8Unorm,
        wgpu::TextureFormat::Etc2Rgb8UnormSrgb,
        wgpu::TextureFormat::Etc2Rgb8A1Unorm,
        wgpu::TextureFormat::Etc2Rgb8A1UnormSrgb,
        wgpu::TextureFormat::Etc2Rgba8Unorm,
        wgpu::TextureFormat::Etc2Rgba8UnormSrgb,
        wgpu::TextureFormat::EacR11Unorm,
        wgpu::TextureFormat::EacR11Snorm,
        wgpu::TextureFormat::EacRg11Unorm,
        wgpu::TextureFormat::EacRg11Snorm,
    ];

    let mut unsuppported_formats = vec![];
    
    let mut not_render_formats = vec![];

    for f in formats {
        let features = adapter.get_texture_format_features(f);
        if features.allowed_usages.is_empty() {
            unsuppported_formats.push(f);
        } else if !features.allowed_usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
            not_render_formats.push(f);
        }
    }

    println!("========= Unsuppported Texture Format: {:?}\n", unsuppported_formats);

    println!("========= No Render Texture Format: {:?}\n", not_render_formats);

}
