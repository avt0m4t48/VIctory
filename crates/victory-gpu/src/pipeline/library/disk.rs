use std::io::{self, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) const SHARD_COUNT: usize = 16;

/// The first hex digit of the content hash selects a shard. The mapping is
/// stable across runs and requires no index alongside the library blobs.
pub(super) fn shard_index(name: &str) -> usize {
    name.chars()
        .next()
        .and_then(|digit| digit.to_digit(16))
        .unwrap_or_default() as usize
}

pub(super) fn shard_file(index: usize) -> String {
    format!("host-pipelines-v1-{index:x}.bin")
}

pub(super) struct Stored<T> {
    pub(super) value: T,
    added: u64,
}

impl<T> Stored<T> {
    pub(super) fn new(value: T) -> Self {
        Self { value, added: 0 }
    }

    pub(super) fn note_store(&mut self) {
        self.added += 1;
    }
}

#[derive(Debug)]
pub(super) enum SaveError<E> {
    Serialize(E),
    Write(io::Error),
}

/// Saves one dirty shard. The caller serializes saves against one another,
/// and stores can proceed after we release the shard lock for disk I/O.
pub(super) fn write_dirty<T, E>(
    shard: &Mutex<Stored<T>>,
    path: &Path,
    serialize: impl FnOnce(&T) -> Result<Vec<u8>, E>,
) -> Result<Option<usize>, SaveError<E>> {
    let (bytes, added) = {
        let shard = shard.lock().unwrap_or_else(|error| error.into_inner());
        if shard.added == 0 {
            return Ok(None);
        }
        (
            serialize(&shard.value).map_err(SaveError::Serialize)?,
            shard.added,
        )
    };
    write_atomically(path, &bytes).map_err(SaveError::Write)?;
    shard
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .added -= added;
    Ok(Some(bytes.len()))
}

/// Writes a same-directory temporary file before replacing the cache entry.
/// Competing processes use distinct temporary files.
pub(super) fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let (temporary, mut file) = loop {
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let temporary = path.with_extension(format!("{}.{id}.tmp", std::process::id()));
        match std::fs::File::create_new(&temporary) {
            Ok(file) => break (temporary, file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    let write = file.write_all(bytes);
    drop(file);
    let result = write.and_then(|()| std::fs::rename(&temporary, path));
    if result.is_err() {
        let _ = std::fs::remove_file(temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!(
                "victory-pipeline-shards-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn generated_names_map_to_every_shard_and_stable_paths() {
        assert_eq!(SHARD_COUNT, 16);
        for (index, hex) in "0123456789abcdef".chars().enumerate() {
            assert_eq!(shard_index(&format!("{hex}0123456789abcdef")), index);
            assert_eq!(shard_index(&format!("{hex}fedcba9876543210")), index);
            assert_eq!(shard_file(index), format!("host-pipelines-v1-{hex}.bin"));
        }
    }

    #[test]
    fn saves_only_the_dirty_shard() {
        let dir = TestDir::new();
        let changed = Mutex::new(Stored::new(vec![1_u8, 2]));
        let clean = Mutex::new(Stored::new(vec![3_u8, 4]));
        let changed_path = dir.0.join(shard_file(0));
        let clean_path = dir.0.join(shard_file(1));
        let legacy_path = dir.0.join("host-pipelines.bin");
        std::fs::write(&changed_path, [0, 0]).unwrap();
        std::fs::write(&clean_path, [3, 4]).unwrap();
        std::fs::write(&legacy_path, [5, 6]).unwrap();
        changed.lock().unwrap().note_store();

        assert_eq!(
            write_dirty(&changed, &changed_path, |v| Ok::<_, ()>(v.clone())).unwrap(),
            Some(2)
        );
        assert_eq!(
            write_dirty(&clean, &clean_path, |_| -> Result<Vec<u8>, ()> {
                panic!("clean shard serialized")
            })
            .unwrap(),
            None
        );
        assert_eq!(
            write_dirty(&changed, &changed_path, |_| -> Result<Vec<u8>, ()> {
                panic!("saved shard serialized again")
            })
            .unwrap(),
            None
        );
        assert_eq!(std::fs::read(&changed_path).unwrap(), [1, 2]);
        assert_eq!(std::fs::read(&clean_path).unwrap(), [3, 4]);
        assert_eq!(std::fs::read(&legacy_path).unwrap(), [5, 6]);
    }

    #[test]
    fn failed_serialization_and_failed_replace_keep_the_shard_dirty() {
        let dir = TestDir::new();
        let shard = Mutex::new(Stored::new(vec![9_u8]));
        let path = dir.0.join(shard_file(8));
        shard.lock().unwrap().note_store();
        assert!(matches!(
            write_dirty(&shard, &path, |_| Err("serialize")),
            Err(SaveError::Serialize("serialize"))
        ));
        std::fs::create_dir(&path).unwrap();
        assert!(matches!(
            write_dirty(&shard, &path, |v| Ok::<_, ()>(v.clone())),
            Err(SaveError::Write(error)) if error.raw_os_error().is_some()
        ));
        assert_eq!(
            std::fs::read_dir(&dir.0).unwrap().count(),
            1,
            "temporary file survived a failed replace"
        );
        std::fs::remove_dir(&path).unwrap();
        assert_eq!(
            write_dirty(&shard, &path, |v| Ok::<_, ()>(v.clone())).unwrap(),
            Some(1)
        );
        assert_eq!(std::fs::read(&path).unwrap(), [9]);
    }

    #[test]
    fn competing_writes_never_share_a_temporary_file() {
        let dir = TestDir::new();
        let path = dir.0.join(shard_file(0));
        let first = vec![1_u8; 4096];
        let second = vec![2_u8; 4096];
        std::thread::scope(|scope| {
            let first_write = scope.spawn(|| write_atomically(&path, &first));
            let second_write = scope.spawn(|| write_atomically(&path, &second));
            // Windows can refuse a rename while the other writer still has
            // an open handle. Either serialization remains a complete file.
            let first_result = first_write.join().unwrap();
            let second_result = second_write.join().unwrap();
            assert!(first_result.is_ok() || second_result.is_ok());
        });
        let saved = std::fs::read(&path).unwrap();
        assert!(saved == first || saved == second);
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 1);
    }
}
