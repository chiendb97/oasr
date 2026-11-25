// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! One replica as the router and its supervisor share it: where it listens,
//! whether it is ready, and how much the router has open on it.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use metrics::{Counter, Gauge};
use oasr_metrics::{
    label, ROUTER_REPLICA_IN_FLIGHT, ROUTER_REPLICA_READY, ROUTER_REPLICA_RESTARTS,
};
use serde::{Deserialize, Serialize};

use super::plan::ReplicaSpec;

/// What a replica reports once both its listeners are bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundAddrs {
    /// The reporting process, so a file left by an earlier one is not read as
    /// this one's.
    pub pid: u32,
    pub http: SocketAddr,
    pub grpc: SocketAddr,
}

pub struct Replica {
    pub spec: ReplicaSpec,
    addrs: Mutex<Option<BoundAddrs>>,
    ready: AtomicBool,
    inflight: AtomicUsize,
    ready_gauge: Gauge,
    inflight_gauge: Gauge,
    pub restarts: Counter,
}

impl Replica {
    /// Resolves its metric handles, so the recorder must be installed first.
    pub fn new(spec: ReplicaSpec) -> Arc<Self> {
        let labels = [
            (label::REPLICA, spec.index.to_string()),
            (label::GPU, spec.device.clone()),
        ];
        Arc::new(Self {
            ready_gauge: metrics::gauge!(ROUTER_REPLICA_READY, &labels),
            inflight_gauge: metrics::gauge!(ROUTER_REPLICA_IN_FLIGHT, &labels),
            restarts: metrics::counter!(ROUTER_REPLICA_RESTARTS, &labels),
            spec,
            addrs: Mutex::new(None),
            ready: AtomicBool::new(false),
            inflight: AtomicUsize::new(0),
        })
    }

    pub fn index(&self) -> usize {
        self.spec.index
    }

    pub fn addrs(&self) -> Option<BoundAddrs> {
        *self.addrs.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn set_addrs(&self, addrs: Option<BoundAddrs>) {
        *self.addrs.lock().unwrap_or_else(|e| e.into_inner()) = addrs;
        if addrs.is_none() {
            self.set_ready(false);
        }
    }

    /// Where to send a request, if this replica takes requests now.
    pub fn routable(&self) -> Option<BoundAddrs> {
        if self.ready.load(Ordering::Acquire) {
            self.addrs()
        } else {
            None
        }
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Returns whether this changed the state.
    pub fn set_ready(&self, ready: bool) -> bool {
        let was = self.ready.swap(ready, Ordering::AcqRel);
        self.ready_gauge.set(if ready { 1.0 } else { 0.0 });
        was != ready
    }

    pub fn inflight(&self) -> usize {
        self.inflight.load(Ordering::Acquire)
    }

    /// Count one request, stream or session on this replica until the guard
    /// drops.  This is the load the router balances on.
    pub fn track(self: &Arc<Self>) -> InFlight {
        let n = self.inflight.fetch_add(1, Ordering::AcqRel) + 1;
        self.inflight_gauge.set(n as f64);
        InFlight(Arc::clone(self))
    }
}

/// One unit of [`Replica::track`]ed load.
pub struct InFlight(Arc<Replica>);

impl Drop for InFlight {
    fn drop(&mut self) {
        let n = self.0.inflight.fetch_sub(1, Ordering::AcqRel) - 1;
        self.0.inflight_gauge.set(n as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn spec(index: usize) -> ReplicaSpec {
        ReplicaSpec {
            index,
            device: index.to_string(),
        }
    }

    #[test]
    fn load_is_counted_while_a_guard_lives() {
        let r = Replica::new(spec(0));
        let a = r.track();
        let b = r.track();
        assert_eq!(r.inflight(), 2);
        drop(a);
        assert_eq!(r.inflight(), 1);
        drop(b);
        assert_eq!(r.inflight(), 0);
    }

    #[test]
    fn a_replica_routes_only_when_ready_and_bound() {
        let r = Replica::new(spec(0));
        let a = BoundAddrs {
            pid: 1,
            http: "127.0.0.1:1".parse().unwrap(),
            grpc: "127.0.0.1:2".parse().unwrap(),
        };
        assert!(r.set_ready(true));
        assert_eq!(r.routable(), None, "ready but not bound");
        r.set_addrs(Some(a));
        assert_eq!(r.routable(), Some(a));
        // Losing the process takes readiness with it.
        r.set_addrs(None);
        assert!(!r.is_ready());
        r.set_addrs(Some(a));
        assert_eq!(r.routable(), None);
    }
}
