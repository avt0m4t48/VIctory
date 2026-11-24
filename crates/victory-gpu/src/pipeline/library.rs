//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The first hash digit selects a
//!    separate library. Saves only serialize libraries with new entries.
//!    The host rejects blobs for another adapter or driver, and entries
//!    whose description differs.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//!
//! Calls on each individual library are serialized with its own mutex.

mod disk;

use std::collections::HashSet;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, TryLockError};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Vec<u8>,
}

impl Library {
    fn serialize(&self) -> windows::core::Result<Vec<u8>> {
        // SAFETY: a buffer of the size the library asks for, under the shard lock.
        unsafe {
            let mut bytes = vec![0u8; self.library.GetSerializedSize()];
            self.library.Serialize(&mut bytes).map(|()| bytes)
        }
    }
}

// SAFETY: D3D12 objects are free-threaded; every call is made under the
// mutex that holds the library.
unsafe impl Send for Library {}

type Shard = Mutex<disk::Stored<Library>>;
static SHARDS: [OnceLock<Option<Shard>>; disk::SHARD_COUNT] =
    [const { OnceLock::new() }; disk::SHARD_COUNT];
const LEGACY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

fn device() -> Option<&'static ID3D12Device1> {
    static DEVICE: OnceLock<Option<ID3D12Device1>> = OnceLock::new();
    DEVICE
        .get_or_init(|| crate::host::host()?.device.cast().ok())
        .as_ref()
}

fn shard(index: usize) -> Option<&'static Shard> {
    SHARDS[index]
        .get_or_init(|| {
            let device = device()?;
            let blob = match DIR.get() {
                Some(dir) => read_blob(&dir.join(disk::shard_file(index)))?,
                None => Vec::new(),
            };
            let library = open(device, blob);
            if library.is_none() {
                tracing::warn!(shard = index, "host pipeline library shard unavailable");
            }
            library.map(|library| Mutex::new(disk::Stored::new(library)))
        })
        .as_ref()
}

/// The old monolith stays available on shard misses. It is opened only when
/// needed, promoted entries are stored in the new shards, and the old file
/// is never written or removed here.
fn legacy() -> Option<&'static Mutex<Library>> {
    static LEGACY: OnceLock<Option<Mutex<Library>>> = OnceLock::new();
    LEGACY
        .get_or_init(|| {
            let device = device()?;
            let blob = read_blob(&DIR.get()?.join(LEGACY_FILE))?;
            if blob.is_empty() {
                return None;
            }
            match open_blob(device, blob) {
                Ok(library) => {
                    tracing::info!(
                        bytes = library._blob.len(),
                        "legacy host pipeline library loaded"
                    );
                    Some(Mutex::new(library))
                }
                Err(error) => {
                    tracing::info!(%error, "legacy host pipeline library rejected");
                    None
                }
            }
        })
        .as_ref()
}

/// A new path has no blob. Other read errors leave the file alone for this
/// process rather than letting an empty library overwrite it.
fn read_blob(path: &Path) -> Option<Vec<u8>> {
    match std::fs::read(path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Some(Vec::new()),
        Err(error) => {
            tracing::warn!(%error, file = %path.display(), "host pipeline library not read");
            None
        }
    }
}

fn open_blob(device: &ID3D12Device1, blob: Vec<u8>) -> windows::core::Result<Library> {
    // SAFETY: the bytes outlive the library (kept beside it).
    let library =
        unsafe { device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(&blob)? }.cast()?;
    Ok(Library {
        library,
        _blob: blob,
    })
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    if !blob.is_empty() {
        match open_blob(device, blob) {
            Ok(library) => {
                tracing::info!(
                    bytes = library._blob.len(),
                    "host pipeline library shard loaded"
                );
                return Some(library);
            }
            Err(error) => tracing::info!(%error, "host pipeline library shard rejected"),
        }
    }
    open_blob(device, Vec::new()).ok()
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root]
        .into_iter()
        .chain(code.iter().copied())
        .chain([state.as_bytes()])
    {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl Fn(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide_name = HSTRING::from(name);
    let wide = PCWSTR(wide_name.as_ptr());
    let shard = shard(disk::shard_index(name));
    if let Some(shard) = shard
        && let Ok(pso) = load(
            &shard
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .value
                .library,
            wide,
        )
    {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    let pso = match legacy().and_then(|legacy| {
        load(
            &legacy.lock().unwrap_or_else(|e| e.into_inner()).library,
            wide,
        )
        .ok()
    }) {
        Some(pso) => {
            LOADED.fetch_add(1, Ordering::Relaxed);
            pso
        }
        None => {
            // Compiled outside the lock: compiles take up to seconds.
            let pso = create()?;
            MADE.fetch_add(1, Ordering::Relaxed);
            pso
        }
    };
    if let Some(shard) = shard {
        let mut shard = shard.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: a live pipeline; a duplicate name leaves the library unchanged.
        if unsafe { shard.value.library.StorePipeline(wide, &pso) }.is_ok() {
            shard.note_store();
        }
    }
    Ok(pso)
}

/// Writes changed shards and the early list when due. Saves are serialized
/// so an older snapshot cannot overwrite a newer one.
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let Some(dir) = DIR.get() else { return };
    let mut last = if now {
        LAST.lock().unwrap_or_else(|e| e.into_inner())
    } else {
        match LAST.try_lock() {
            Ok(last) => last,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => return,
        }
    };
    if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
        return;
    }
    *last = Some(std::time::Instant::now());

    let mut saved_shards = 0;
    let mut bytes_written = 0;
    for (index, shard) in SHARDS.iter().enumerate() {
        let Some(Some(shard)) = shard.get() else {
            continue;
        };
        let path = dir.join(disk::shard_file(index));
        let saved = disk::write_dirty(shard, &path, Library::serialize);
        match saved {
            Ok(Some(bytes)) => {
                saved_shards += 1;
                bytes_written += bytes;
            }
            Ok(None) => {}
            Err(disk::SaveError::Serialize(error)) => {
                tracing::warn!(%error, shard = index, "host pipeline library shard not serialized");
            }
            Err(disk::SaveError::Write(error)) => {
                tracing::warn!(%error, shard = index, "host pipeline library shard not saved");
            }
        }
    }
    let early = save_early(dir);
    if saved_shards != 0 || early.is_some() {
        tracing::info!(
            bytes = bytes_written,
            shards = saved_shards,
            early,
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "pipeline cache saved"
        );
    }
}

fn save_early(dir: &Path) -> Option<usize> {
    if !USED_CHANGED.swap(false, Ordering::Relaxed) {
        return None;
    }
    let used: Vec<String> = USED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .flatten()
        .cloned()
        .collect();
    match disk::write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()) {
        Ok(()) => Some(used.len()),
        Err(error) => {
            USED_CHANGED.store(true, Ordering::Relaxed);
            tracing::warn!(%error, "early pipeline list not saved");
            None
        }
    }
}

/// Title pipeline keys drawn or dispatched with (this run and the last).
static USED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<HashSet<String>>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let set: HashSet<String> = listed
            .lines()
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect();
        tracing::info!(pipelines = set.len(), "title pipelines to start early");
        *used = Some(set);
    }
    used
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|s| s.contains(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    let mut used = used();
    let set = used.get_or_insert_with(HashSet::new);
    if set.len() < EARLY_MAX && set.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let library =
            open(&device, b"not a pipeline library".repeat(16)).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }

    #[test]
    fn a_stored_compute_pipeline_loads_from_the_saved_shard() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::s;

        let Some(device) = device() else {
            eprintln!("skipped: no D3D12 device");
            return;
        };
        let Some(library) = open(device, Vec::new()) else {
            eprintln!("skipped: this device does not support D3D12 pipeline libraries");
            return;
        };
        let source = br#"[RootSignature("RootFlags(0)")] [numthreads(1, 1, 1)] void main() {}"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: shader source and out pointer are live through compilation.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("pipeline-library-test"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
            .expect("compile the test shader");
        }
        let code = code.expect("test shader bytecode");
        // SAFETY: the blob lives through root and pipeline creation and the load below.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        let root: ID3D12RootSignature =
            unsafe { device.CreateRootSignature(0, bytes) }.expect("root from the test shader");
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        };
        let name = name("compute", b"", &[bytes], "");
        let wide = HSTRING::from(name.as_str());
        let wide_name = PCWSTR(wide.as_ptr());
        let dir = tempfile::tempdir().expect("cache directory");
        let path = dir.path().join(disk::shard_file(disk::shard_index(&name)));
        let shard = Mutex::new(disk::Stored::new(library));
        {
            let mut shard = shard.lock().unwrap();
            // SAFETY: the description and the pipeline's name and data are live.
            unsafe {
                let pso: ID3D12PipelineState = device
                    .CreateComputePipelineState(&desc)
                    .expect("create test pipeline");
                shard
                    .value
                    .library
                    .StorePipeline(wide_name, &pso)
                    .expect("store test pipeline");
            }
            shard.note_store();
        }
        let saved = disk::write_dirty(&shard, &path, Library::serialize).expect("save test shard");
        assert!(saved.is_some_and(|bytes| bytes > 0));
        drop(shard);
        let restored = open_blob(device, std::fs::read(path).expect("read test shard"))
            .expect("open the saved blob");
        // SAFETY: a live library and a valid description matching the stored pipeline.
        let _: ID3D12PipelineState =
            unsafe { restored.library.LoadComputePipeline(wide_name, &desc) }
                .expect("reuse the saved pipeline");
        eprintln!("reused a synthetic D3D12 compute pipeline from the saved shard");
    }
}
