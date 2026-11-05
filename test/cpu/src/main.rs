// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: BSD-3-Clause

//! The CPU target's Rust against the interpreter, bit for bit.
//!
//! checks.cpu writes, from Cases.hs, each case's kernels as the CPU target
//! lowers them (`src/gen`), the interpreter's inputs and outputs (`data`),
//! the code that runs the one on the other (`src/cases.rs`) and, per kernel,
//! the pieces of FuryKernel.Emit.Rust that runtimes generate
//! (`src/launch.rs`). This runs every case and compares every output.

mod cases;
mod launch;

use std::process::ExitCode;

/// An element of a buffer, as the interpreter writes it: little-endian.
trait Elem: Copy {
    const SIZE: usize;
    fn from_le(bytes: &[u8]) -> Self;
    fn bits(self) -> u32;
    /// For a NaN, whether it is quiet.
    fn quiet_nan(self) -> Option<bool>;
}

impl Elem for f32 {
    const SIZE: usize = 4;
    fn from_le(bytes: &[u8]) -> Self {
        f32::from_le_bytes(bytes.try_into().expect("four bytes"))
    }
    fn bits(self) -> u32 {
        self.to_bits()
    }
    fn quiet_nan(self) -> Option<bool> {
        self.is_nan().then_some(self.to_bits() & 0x0040_0000 != 0)
    }
}

impl Elem for half::f16 {
    const SIZE: usize = 2;
    fn from_le(bytes: &[u8]) -> Self {
        half::f16::from_bits(u16::from_le_bytes(bytes.try_into().expect("two bytes")))
    }
    fn bits(self) -> u32 {
        u32::from(self.to_bits())
    }
    fn quiet_nan(self) -> Option<bool> {
        self.is_nan().then_some(self.to_bits() & 0x0200 != 0)
    }
}

impl Elem for half::bf16 {
    const SIZE: usize = 2;
    fn from_le(bytes: &[u8]) -> Self {
        half::bf16::from_bits(u16::from_le_bytes(bytes.try_into().expect("two bytes")))
    }
    fn bits(self) -> u32 {
        u32::from(self.to_bits())
    }
    fn quiet_nan(self) -> Option<bool> {
        self.is_nan().then_some(self.to_bits() & 0x0040 != 0)
    }
}

impl Elem for i32 {
    const SIZE: usize = 4;
    fn from_le(bytes: &[u8]) -> Self {
        i32::from_le_bytes(bytes.try_into().expect("four bytes"))
    }
    fn bits(self) -> u32 {
        self.cast_unsigned()
    }
    fn quiet_nan(self) -> Option<bool> {
        None
    }
}

impl Elem for u32 {
    const SIZE: usize = 4;
    fn from_le(bytes: &[u8]) -> Self {
        u32::from_le_bytes(bytes.try_into().expect("four bytes"))
    }
    fn bits(self) -> u32 {
        self
    }
    fn quiet_nan(self) -> Option<bool> {
        None
    }
}

impl Elem for i8 {
    const SIZE: usize = 1;
    fn from_le(bytes: &[u8]) -> Self {
        i8::from_le_bytes([bytes[0]])
    }
    fn bits(self) -> u32 {
        u32::from(self.cast_unsigned())
    }
    fn quiet_nan(self) -> Option<bool> {
        None
    }
}

impl Elem for u8 {
    const SIZE: usize = 1;
    fn from_le(bytes: &[u8]) -> Self {
        bytes[0]
    }
    fn bits(self) -> u32 {
        u32::from(self)
    }
    fn quiet_nan(self) -> Option<bool> {
        None
    }
}

impl Elem for bool {
    const SIZE: usize = 1;
    fn from_le(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
    fn bits(self) -> u32 {
        u32::from(self)
    }
    fn quiet_nan(self) -> Option<bool> {
        None
    }
}

/// How an output is compared with the interpreter's.
#[derive(Clone, Copy)]
enum Compare {
    /// Bit for bit.
    Bits,
    /// Bit for bit, except that two NaNs of one class (both quiet or both
    /// signalling) agree whatever their signs and payloads, which the IR
    /// leaves unspecified.
    NanClass,
}

fn read(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn load<T: Elem>(path: &str) -> Vec<T> {
    read(path).chunks(T::SIZE).map(T::from_le).collect()
}

/// The elements of an output that differ from the interpreter's, the first few printed.
fn compare<T: Elem>(case: &str, name: &str, how: Compare, got: &[T], want: &[T]) -> usize {
    if got.len() != want.len() {
        println!(
            "{case} {name}: {} elements, the interpreter's {}",
            got.len(),
            want.len()
        );
        return got.len().max(want.len());
    }
    let mut bad = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let agree = g.bits() == w.bits()
            || matches!(how, Compare::NanClass)
                && g.quiet_nan().is_some()
                && g.quiet_nan() == w.quiet_nan();
        if !agree {
            if bad < 12 {
                println!(
                    "{case} {name}[{i}]: {:#x}, the interpreter's {:#x}",
                    g.bits(),
                    w.bits()
                );
            }
            bad += 1;
        }
    }
    bad
}

/// The message a call panics with, if it does, the panic itself not
/// printed.
fn refusal(f: impl FnOnce()) -> Option<String> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(hook);
    match outcome {
        Ok(()) => None,
        Err(payload) => Some(
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|m| (*m).to_owned()))
                .unwrap_or_default(),
        ),
    }
}

/// A launch the interpreter refuses, against the CPU target's panic.
fn refused(case: &str, got: Option<&str>, want: &str) -> usize {
    if got == Some(want) {
        0
    } else {
        println!("{case}: {got:?}, where the interpreter refuses it: {want}");
        1
    }
}

/// A launch size as Emit.Rust computes it, against the interpreter's.
fn size(case: &str, what: &str, got: u64, want: u64) -> usize {
    if got == want {
        0
    } else {
        println!("{case} {what}: {got}, the interpreter's {want}");
        1
    }
}

fn main() -> ExitCode {
    let bad = cases::run();
    println!("{bad} mismatches");
    if bad == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
