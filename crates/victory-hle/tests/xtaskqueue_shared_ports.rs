use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::ptr::{null, null_mut};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use victory_hle::modules::xasync::{
    Queue, XTaskQueueCloseHandle, XTaskQueueCreate, XTaskQueueCreateComposite, XTaskQueueDispatch,
    XTaskQueueGetPort, XTaskQueueSubmitDelayedCallback, XTaskQueueTerminate,
};
use victory_hle::modules::xgameruntime::S_OK;

const WORK: u32 = 0;
const COMPLETION: u32 = 1;
const MANUAL: u32 = 0;
const SERIALIZED: u32 = 2;

struct Handle(*const Queue);

impl Drop for Handle {
    fn drop(&mut self) {
        XTaskQueueCloseHandle(self.0);
    }
}

fn shared_queues(completion_mode: u32) -> (Handle, Handle) {
    let mut original = Handle(null());
    assert_eq!(
        XTaskQueueCreate(MANUAL, completion_mode, &mut original.0),
        S_OK
    );
    let (mut work, mut completion) = (null(), null());
    assert_eq!(XTaskQueueGetPort(original.0, WORK, &mut work), S_OK);
    assert_eq!(
        XTaskQueueGetPort(original.0, COMPLETION, &mut completion),
        S_OK
    );
    let mut composite = Handle(null());
    assert_eq!(
        XTaskQueueCreateComposite(work, completion, &mut composite.0),
        S_OK
    );
    (original, composite)
}

type Events = RefCell<Vec<(u8, bool)>>;

struct Entry<'a> {
    events: &'a Events,
    tag: u8,
}

extern "system" fn record(context: *mut c_void, canceled: bool) {
    // SAFETY: each entry stays live for the duration of manual dispatch or termination.
    let entry = unsafe { &*context.cast::<Entry<'_>>() };
    entry.events.borrow_mut().push((entry.tag, canceled));
}

fn submit(queue: &Handle, port: u32, entry: &Entry<'_>) {
    assert_eq!(
        XTaskQueueSubmitDelayedCallback(
            queue.0,
            port,
            0,
            (entry as *const Entry<'_>).cast_mut().cast(),
            Some(record)
        ),
        S_OK
    );
}

#[test]
fn terminating_composite_preserves_original_callbacks_in_fifo_order() {
    let (original, composite) = shared_queues(MANUAL);
    let events = Events::default();
    let entries = [1, 2, 3].map(|tag| Entry {
        events: &events,
        tag,
    });
    submit(&original, WORK, &entries[0]);
    submit(&composite, WORK, &entries[1]);
    submit(&original, WORK, &entries[2]);

    assert_eq!(
        XTaskQueueTerminate(composite.0, true, null_mut(), None),
        S_OK
    );
    assert_eq!(events.borrow().as_slice(), [(2, true)]);
    assert!(XTaskQueueDispatch(original.0, WORK, 0));
    assert!(XTaskQueueDispatch(original.0, WORK, 0));
    assert!(!XTaskQueueDispatch(original.0, WORK, 0));
    assert_eq!(
        events.borrow().as_slice(),
        [(2, true), (1, false), (3, false)]
    );
}

#[test]
fn reentrant_dispatch_uses_the_submitting_queues_cancellation_state() {
    struct Reenter<'a> {
        original: &'a Handle,
        work: &'a Entry<'a>,
        dispatched: Cell<[bool; 2]>,
    }

    extern "system" fn dispatch_completion(context: *mut c_void, canceled: bool) {
        // SAFETY: termination runs this callback synchronously while its context is live.
        let context = unsafe { &*context.cast::<Reenter<'_>>() };
        context
            .work
            .events
            .borrow_mut()
            .push((context.work.tag, canceled));
        context.dispatched.set([
            XTaskQueueDispatch(context.original.0, COMPLETION, 0),
            XTaskQueueDispatch(context.original.0, COMPLETION, 0),
        ]);
    }

    let (original, composite) = shared_queues(MANUAL);
    let events = Events::default();
    let entries = [1, 2, 3].map(|tag| Entry {
        events: &events,
        tag,
    });
    let reenter = Reenter {
        original: &original,
        work: &entries[0],
        dispatched: Cell::default(),
    };
    assert_eq!(
        XTaskQueueSubmitDelayedCallback(
            composite.0,
            WORK,
            0,
            (&raw const reenter).cast_mut().cast(),
            Some(dispatch_completion),
        ),
        S_OK
    );
    submit(&composite, COMPLETION, &entries[1]);
    submit(&original, COMPLETION, &entries[2]);

    assert_eq!(
        XTaskQueueTerminate(composite.0, true, null_mut(), None),
        S_OK
    );
    assert_eq!(reenter.dispatched.get(), [true, true]);
    assert_eq!(
        events.borrow().as_slice(),
        [(1, true), (2, true), (3, false)]
    );
    assert!(!XTaskQueueDispatch(original.0, COMPLETION, 0));
}

#[test]
fn serialized_worker_cancels_completion_during_work_port_termination() {
    const TIMEOUT: Duration = Duration::from_secs(5);

    struct HoldWorker {
        started: Sender<()>,
        release: Receiver<()>,
    }

    extern "system" fn hold_worker(context: *mut c_void, _: bool) {
        // SAFETY: the callback owns the Box transferred at submission.
        let hold = unsafe { Box::from_raw(context.cast::<HoldWorker>()) };
        let _ = hold.started.send(());
        let _ = hold.release.recv();
    }

    extern "system" fn report_cancellation(context: *mut c_void, canceled: bool) {
        // SAFETY: the callback owns the Box transferred at submission.
        let done = unsafe { Box::from_raw(context.cast::<Sender<bool>>()) };
        let _ = done.send(canceled);
    }

    struct ReleaseWorker {
        release: Sender<()>,
        done: Receiver<bool>,
        observed: Cell<Option<bool>>,
    }

    extern "system" fn release_worker(context: *mut c_void, _: bool) {
        // SAFETY: manual work-port cancellation invokes this on the test thread.
        let release = unsafe { &*context.cast::<ReleaseWorker>() };
        let _ = release.release.send(());
        release
            .observed
            .set(release.done.recv_timeout(TIMEOUT).ok());
    }

    let (original, composite) = shared_queues(SERIALIZED);
    let (started_tx, started_rx) = channel();
    let (release_tx, release_rx) = channel();
    let (done_tx, done_rx) = channel::<bool>();
    let hold = Box::new(HoldWorker {
        started: started_tx,
        release: release_rx,
    });
    assert_eq!(
        XTaskQueueSubmitDelayedCallback(
            original.0,
            COMPLETION,
            0,
            Box::into_raw(hold).cast(),
            Some(hold_worker),
        ),
        S_OK
    );
    started_rx.recv_timeout(TIMEOUT).unwrap();

    let release = ReleaseWorker {
        release: release_tx,
        done: done_rx,
        observed: Cell::default(),
    };
    assert_eq!(
        XTaskQueueSubmitDelayedCallback(
            composite.0,
            WORK,
            0,
            (&raw const release).cast_mut().cast(),
            Some(release_worker),
        ),
        S_OK
    );
    assert_eq!(
        XTaskQueueSubmitDelayedCallback(
            composite.0,
            COMPLETION,
            0,
            Box::into_raw(Box::new(done_tx)).cast(),
            Some(report_cancellation),
        ),
        S_OK
    );

    assert_eq!(
        XTaskQueueTerminate(composite.0, true, null_mut(), None),
        S_OK
    );
    assert_eq!(release.observed.get(), Some(true));
}
