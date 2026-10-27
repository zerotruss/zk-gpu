//! Thin wrapper over wgpu: one device, storage buffers, compute dispatches.
//! Everything here works on native (Metal/Vulkan/DX12) and in browsers (WebGPU).

use wgpu::util::DeviceExt;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::AdapterInfo,
    // Compiled pipelines by source text: plans with the same constants share them.
    pipelines: std::sync::Mutex<std::collections::HashMap<String, wgpu::ComputePipeline>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for Gpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Gpu {
    pub async fn new_async() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("zk-gpu"),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no GPU device: {e}"))?;
        Ok(Self { device, queue, adapter: adapter.get_info(), pipelines: Default::default() })
    }

    /// Calls `log` with a message when the GPU reports an error that no call is waiting for (a
    /// validation error, or the device being lost), which otherwise only shows up as a failed read.
    pub fn report_errors(&self, log: impl Fn(String) + Clone + Send + Sync + 'static) {
        let on_error = log.clone();
        self.device.on_uncaptured_error(std::sync::Arc::new(move |error| on_error(format!("GPU error: {error}"))));
        self.device
            .set_device_lost_callback(move |reason, message| log(format!("GPU device lost ({reason:?}): {message}")));
    }

    /// Blocking constructor for native programs; use `new_async` in browsers.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Self {
        pollster::block_on(Self::new_async()).expect("GPU init failed")
    }

    /// Storage buffer initialised with `data`.
    pub fn buffer(&self, data: &[u32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Zero-filled storage buffer of `words` u32s.
    pub fn empty_buffer(&self, words: usize) -> wgpu::Buffer {
        self.buffer(&vec![0u32; words])
    }

    pub fn pipeline(&self, wgsl: &str, entry: &str) -> wgpu::ComputePipeline {
        let key = format!("{entry}\n{wgsl}");
        if let Some(found) = self.pipelines.lock().unwrap().get(&key) {
            return found.clone();
        }
        let pipeline = self.compile(wgsl, entry);
        self.pipelines.lock().unwrap().insert(key, pipeline.clone());
        pipeline
    }

    fn compile(&self, wgsl: &str, entry: &str) -> wgpu::ComputePipeline {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            // Our kernels write every workgroup slot before reading it, so the mandatory
            // zero-fill is pure overhead (on Metal one thread zeroes the whole tile).
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        })
    }

    /// Grid for `workgroups` workgroups that respects the 65535-per-dimension dispatch limit.
    /// Kernels recover their linear thread id from `workgroup_id` and `num_workgroups`.
    pub fn grid_1d(workgroups: u32) -> [u32; 3] {
        const ROW: u32 = 32768;
        [workgroups.min(ROW), workgroups.div_ceil(ROW), 1]
    }

    /// Record one dispatch of `pipeline` with `buffers` bound to 0, 1, 2, ...
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
        workgroups: u32,
    ) {
        self.encode_grid(encoder, pipeline, buffers, [workgroups, 1, 1]);
    }

    pub fn encode_grid(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
        grid: [u32; 3],
    ) {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
    }

    /// Dispatch once and wait for the GPU to finish.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(&self, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer], workgroups: u32) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode(&mut encoder, pipeline, buffers, workgroups);
        self.queue.submit([encoder.finish()]);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Block until all submitted GPU work has finished.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Waits (without blocking a browser's event loop) until every submission so far has finished.
    pub async fn finished_async(&self) {
        let (tx, rx) = futures_channel();
        self.queue.on_submitted_work_done(move || tx(Ok(())));
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let _ = rx.await;
    }

    /// Reads `words` u32s from the start of `buffer`. Fails (instead of panicking, which in a browser
    /// would leave the caller waiting for ever) if the GPU cannot map the result.
    pub async fn try_read_async(&self, buffer: &wgpu::Buffer, words: usize) -> Result<Vec<u32>, String> {
        let bytes = (words * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = futures_channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| tx(r.map_err(|e| e.to_string())));
        // In a browser the map completes on the event loop while we `await`; natively we pump it.
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.await.map_err(|e| format!("GPU buffer map failed: {e}"))?;
        let out = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
        staging.unmap();
        Ok(out)
    }

    pub async fn read_async(&self, buffer: &wgpu::Buffer, words: usize) -> Vec<u32> {
        self.try_read_async(buffer, words).await.expect("GPU read failed")
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(&self, buffer: &wgpu::Buffer, words: usize) -> Vec<u32> {
        pollster::block_on(self.read_async(buffer, words))
    }
}

/// Minimal one-shot channel so we do not pull in an async-runtime dependency.
fn futures_channel(
) -> (impl FnOnce(Result<(), String>) + Send + 'static, impl std::future::Future<Output = Result<(), String>>) {
    use std::sync::{Arc, Mutex};
    use std::task::{Poll, Waker};
    struct State {
        value: Option<Result<(), String>>,
        waker: Option<Waker>,
    }
    let state = Arc::new(Mutex::new(State { value: None, waker: None }));
    let tx_state = state.clone();
    let send = move |v: Result<(), String>| {
        let mut s = tx_state.lock().unwrap();
        s.value = Some(v);
        if let Some(w) = s.waker.take() {
            w.wake();
        }
    };
    let recv = std::future::poll_fn(move |cx| {
        let mut s = state.lock().unwrap();
        match s.value.take() {
            Some(v) => Poll::Ready(v),
            None => {
                s.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    });
    (send, recv)
}
