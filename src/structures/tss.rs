//! Provides a type for the task state segment structure.

use crate::VirtAddr;

/// In 64-bit mode the TSS holds information that is not
/// directly related to the task-switch mechanism,
/// but is used for finding kernel level stack
/// if interrupts arrive while in kernel mode.
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct TaskStateSegment {
    reserved_1: u32,
    /// The full 64-bit canonical forms of the stack pointers (RSP) for privilege levels 0-2.
    pub privilege_stack_table: [VirtAddr; 3],
    reserved_2: u64,
    /// The full 64-bit canonical forms of the interrupt stack table (IST) pointers.
    pub interrupt_stack_table: [VirtAddr; 7],
    reserved_3: u64,
    reserved_4: u16,
    /// The 16-bit offset to the I/O permission bit map from the 64-bit TSS base. It must not
    /// exceed `0xDFFF`.
    pub iomap_base: u16,
}

impl TaskStateSegment {
    /// Creates a new TSS with zeroed privilege and interrupt stack table and a zero
    /// `iomap_base`.
    #[inline]
    pub const fn new() -> TaskStateSegment {
        TaskStateSegment {
            privilege_stack_table: [VirtAddr::zero(); 3],
            interrupt_stack_table: [VirtAddr::zero(); 7],
            iomap_base: 0,
            reserved_1: 0,
            reserved_2: 0,
            reserved_3: 0,
            reserved_4: 0,
        }
    }
}

/// The given IO permissions bitmap is invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum InvalidIoMap {
    /// The IO permissions bitmap is too far from the TSS. It must be within `0xdfff` bytes of the
    /// start of the TSS.
    TooFarFromTss {
        /// The distance of the IO permissions bitmap from the beginning of the TSS.
        distance: usize,
    },
    /// The final byte of the IO permissions bitmap was not 0xff
    InvalidTerminatingByte {
        /// The byte found at the end of the IO permissions bitmap.
        byte: u8,
    },
    /// The IO permissions bitmap exceeds the maximum length (8193).
    TooLong {
        /// The length of the IO permissions bitmap.
        len: usize,
    },
    /// The `iomap_base` in the `TaskStateSegment` struct was not what was expected.
    InvalidBase {
        /// The expected `iomap_base` to be set in the `TaskStateSegment` struct.
        expected: u16,
        /// The actual `iomap_base` set in the `TaskStateSegment` struct.
        got: u16,
    },
}
