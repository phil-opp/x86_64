//! Physical and virtual addresses manipulation

use core::convert::TryFrom;
use core::fmt;
use core::hash::Hash;
#[cfg(feature = "step_trait")]
use core::iter::Step;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Sub, SubAssign};
#[cfg(feature = "memory_encryption")]
use core::sync::atomic::Ordering;

use crate::sealed::Sealed;
#[cfg(feature = "memory_encryption")]
use crate::structures::mem_encrypt::PHYSICAL_ADDRESS_MASK;
use crate::structures::paging::page_table::PageTableLevel;
use crate::structures::paging::{PageOffset, PageTableIndex};

use dep_const_fn::const_fn;

/// The width of a canonical virtual address.
///
/// On `x86_64`, virtual addresses are canonical if all bits above the most significant
/// valid bit are copies of that bit. How many bits are valid depends on the paging mode
/// that the CPU uses:
///
/// - With 4-level paging, the lower 48 bits are valid ([`Width48`]).
/// - With 5-level paging, the lower 57 bits are valid ([`Width57`]).
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait VirtAddrWidth: Copy + Eq + PartialOrd + Ord + Hash + Sealed {
    /// The number of valid bits in a virtual address of this width.
    const BITS: u32;
}

/// The width of virtual addresses with 4-level paging: 48 bits are valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Width48 {}

/// The width of virtual addresses with 5-level paging: 57 bits are valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Width57 {}

impl VirtAddrWidth for Width48 {
    const BITS: u32 = 48;
}

impl Sealed for Width48 {}

impl VirtAddrWidth for Width57 {
    const BITS: u32 = 57;
}

impl Sealed for Width57 {}

/// A canonical 64-bit virtual memory address of the given [width](VirtAddrWidth).
///
/// This is a wrapper type around an `u64`, so it is always 8 bytes, even when compiled
/// on non 64-bit systems. The
/// [`TryFrom`](https://doc.rust-lang.org/std/convert/trait.TryFrom.html) trait can be used for performing conversions
/// between `u64` and `usize`.
///
/// On `x86_64`, only the lower bits of a virtual address can be used. How many bits
/// exactly depends on the paging mode: 48 bits with 4-level paging and 57 bits with
/// 5-level paging. The remaining upper bits need to be copies of the most significant
/// valid bit (bit 47 or bit 56, respectively). Addresses that fulfil this criterion are
/// called “canonical”. This type guarantees that it always represents a canonical address
/// of its width.
///
/// Most code should use the [`VirtAddr48`] and [`VirtAddr57`] type aliases instead of
/// naming this type directly. [`VirtAddr`] is an alias for [`VirtAddr48`].
///
/// # Conversions
///
/// Every 48-bit canonical address is also a 57-bit canonical address, so a [`VirtAddr48`]
/// can be converted into a [`VirtAddr57`] using [`From`]. The reverse conversion can
/// fail and is available through [`TryFrom`].
///
/// # Representation
///
/// This struct has the same representation as a [`u64`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct VirtAddrGeneric<W: VirtAddrWidth>(u64, PhantomData<W>);

/// A canonical 48-bit virtual memory address, as used with 4-level paging.
///
/// The top 16 bits need to be copies of bit 47. See [`VirtAddrGeneric`] for details.
pub type VirtAddr48 = VirtAddrGeneric<Width48>;

/// A canonical 57-bit virtual memory address, as used with 5-level paging.
///
/// The top 7 bits need to be copies of bit 56. See [`VirtAddrGeneric`] for details.
pub type VirtAddr57 = VirtAddrGeneric<Width57>;

/// A canonical 48-bit virtual memory address.
///
/// This is an alias for [`VirtAddr48`]. Use [`VirtAddr57`] for 57-bit virtual addresses
/// (5-level paging).
pub type VirtAddr = VirtAddr48;

/// A virtual memory address as read from or written to the CPU, without any canonicality
/// guarantee.
///
/// This type is used at the boundary between this crate and the hardware, for values that
/// the CPU writes and that this crate cannot verify: the instruction and stack pointer in an
/// [`InterruptStackFrame`](crate::structures::idt::InterruptStackFrame), the contents of
/// `CR2`, segment base registers, and similar. Whether such a value is a valid 48-bit or
/// 57-bit address depends on the paging mode the CPU is running in, which only the kernel
/// knows. Convert it to a checked address type with [`try_into_48`](Self::try_into_48),
/// [`try_into_57`](Self::try_into_57), or [`TryFrom`]; both checked types convert into a
/// `RawVirtAddr` using [`From`].
///
/// Arithmetic on this type is plain `u64` arithmetic that panics on overflow. It does not
/// jump the non-canonical “gap” and does not keep the address canonical.
///
/// # Representation
///
/// This struct has the same representation as a [`u64`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct RawVirtAddr(u64);

/// A 64-bit physical memory address.
///
/// This is a wrapper type around an `u64`, so it is always 8 bytes, even when compiled
/// on non 64-bit systems. The
/// [`TryFrom`](https://doc.rust-lang.org/std/convert/trait.TryFrom.html) trait can be used for performing conversions
/// between `u64` and `usize`.
///
/// On `x86_64`, only the 52 lower bits of a physical address can be used. The top 12 bits need
/// to be zero. This type guarantees that it always represents a valid physical address.
///
/// # Representation
///
/// This struct has the same representation as a [`u64`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct PhysAddr(u64);

/// A passed `u64` was not a valid virtual address.
///
/// This means that the bits above the most significant valid bit (bit 47 for 48-bit
/// addresses, bit 56 for 57-bit addresses) are not a valid sign extension and are not
/// null either. So automatic sign extension would have overwritten possibly meaningful
/// bits. This likely indicates a bug, for example an invalid address calculation.
///
/// Contains the invalid address.
pub struct VirtAddrNotValid(pub u64);

impl core::fmt::Debug for VirtAddrNotValid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("VirtAddrNotValid")
            .field(&format_args!("{:#x}", self.0))
            .finish()
    }
}

impl<W: VirtAddrWidth> VirtAddrGeneric<W> {
    /// The number of valid bits in this virtual address type.
    ///
    /// This is 48 for [`VirtAddr48`] and 57 for [`VirtAddr57`].
    pub const BITS: u32 = W::BITS;

    /// The number of addresses in the (virtual) address space, i.e. `2^BITS`.
    const ADDRESS_SPACE_SIZE: u64 = 1 << W::BITS;

    /// The number of bits that must be a sign extension of the most significant valid bit.
    const SIGN_EXTENSION_BITS: u32 = 64 - W::BITS;

    /// Creates a new canonical virtual address.
    ///
    /// The provided address should already be canonical. If you want to check
    /// whether an address is canonical, use [`try_new`](Self::try_new).
    ///
    /// ## Panics
    ///
    /// This function panics if the bits above the most significant valid bit are invalid
    /// (i.e. are not a proper sign extension of bit 47 for 48-bit addresses or of bit 56
    /// for 57-bit addresses).
    #[inline]
    pub const fn new(addr: u64) -> Self {
        // TODO: Replace with .ok().expect(msg) when that works on stable.
        match Self::try_new(addr) {
            Ok(v) => v,
            Err(_) => {
                panic!("virtual address must be sign extended above its most significant valid bit")
            }
        }
    }

    /// Tries to create a new canonical virtual address.
    ///
    /// This function checks whether the given address is canonical
    /// and returns an error otherwise. An address is canonical
    /// if the bits above the most significant valid bit are a correct sign
    /// extension (i.e. copies of bit 47 for 48-bit addresses or of bit 56 for 57-bit
    /// addresses).
    #[inline]
    pub const fn try_new(addr: u64) -> Result<Self, VirtAddrNotValid> {
        let v = Self::new_truncate(addr);
        if v.0 == addr {
            Ok(v)
        } else {
            Err(VirtAddrNotValid(addr))
        }
    }

    /// Creates a new canonical virtual address, throwing out the bits above the most
    /// significant valid bit.
    ///
    /// This function performs sign extension of bit 47 (for 48-bit addresses) or bit 56
    /// (for 57-bit addresses) to make the address canonical, overwriting the bits above.
    /// If you want to check whether an address is canonical, use [`new`](Self::new) or
    /// [`try_new`](Self::try_new).
    #[inline]
    pub const fn new_truncate(addr: u64) -> Self {
        // By doing the right shift as a signed operation (on a i64), it will
        // sign extend the value, repeating the leftmost bit.
        VirtAddrGeneric(
            ((addr << Self::SIGN_EXTENSION_BITS) as i64 >> Self::SIGN_EXTENSION_BITS) as u64,
            PhantomData,
        )
    }

    /// Creates a new virtual address, without any checks.
    ///
    /// ## Safety
    ///
    /// You must make sure that the address is canonical, i.e. that all bits above the
    /// most significant valid bit are copies of that bit. This is not checked.
    #[inline]
    pub const unsafe fn new_unsafe(addr: u64) -> Self {
        VirtAddrGeneric(addr, PhantomData)
    }

    /// Creates a virtual address that points to `0`.
    #[inline]
    pub const fn zero() -> Self {
        VirtAddrGeneric(0, PhantomData)
    }

    /// Converts the address to an `u64`.
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Creates a virtual address from the given pointer
    #[cfg(target_pointer_width = "64")]
    #[inline]
    pub fn from_ptr<T: ?Sized>(ptr: *const T) -> Self {
        Self::new(ptr as *const () as u64)
    }

    /// Converts the address to a raw pointer.
    #[cfg(target_pointer_width = "64")]
    #[inline]
    pub const fn as_ptr<T>(self) -> *const T {
        self.as_u64() as *const T
    }

    /// Converts the address to a mutable raw pointer.
    #[cfg(target_pointer_width = "64")]
    #[inline]
    pub const fn as_mut_ptr<T>(self) -> *mut T {
        self.as_ptr::<T>() as *mut T
    }

    /// Convenience method for checking if a virtual address is null.
    #[inline]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Aligns the virtual address upwards to the given alignment.
    ///
    /// See the `align_up` function for more information.
    ///
    /// # Panics
    ///
    /// This function panics if the resulting address is higher than
    /// `0xffff_ffff_ffff_ffff`.
    #[inline]
    pub fn align_up<U>(self, align: U) -> Self
    where
        U: Into<u64>,
    {
        Self::new_truncate(align_up(self.0, align.into()))
    }

    /// Aligns the virtual address downwards to the given alignment.
    ///
    /// See the `align_down` function for more information.
    #[inline]
    pub fn align_down<U>(self, align: U) -> Self
    where
        U: Into<u64>,
    {
        self.align_down_u64(align.into())
    }

    /// Aligns the virtual address downwards to the given alignment.
    ///
    /// See the `align_down` function for more information.
    #[inline]
    pub(crate) const fn align_down_u64(self, align: u64) -> Self {
        Self::new_truncate(align_down(self.0, align))
    }

    /// Checks whether the virtual address has the demanded alignment.
    #[inline]
    pub fn is_aligned<U>(self, align: U) -> bool
    where
        U: Into<u64>,
    {
        self.is_aligned_u64(align.into())
    }

    /// Checks whether the virtual address has the demanded alignment.
    #[inline]
    pub(crate) const fn is_aligned_u64(self, align: u64) -> bool {
        self.align_down_u64(align).as_u64() == self.as_u64()
    }

    /// Returns the 12-bit page offset of this virtual address.
    #[inline]
    pub const fn page_offset(self) -> PageOffset {
        PageOffset::new_truncate(self.0 as u16)
    }

    /// Returns the 9-bit level 1 page table index.
    #[inline]
    pub const fn p1_index(self) -> PageTableIndex {
        PageTableIndex::new_truncate((self.0 >> 12) as u16)
    }

    /// Returns the 9-bit level 2 page table index.
    #[inline]
    pub const fn p2_index(self) -> PageTableIndex {
        PageTableIndex::new_truncate((self.0 >> 12 >> 9) as u16)
    }

    /// Returns the 9-bit level 3 page table index.
    #[inline]
    pub const fn p3_index(self) -> PageTableIndex {
        PageTableIndex::new_truncate((self.0 >> 12 >> 9 >> 9) as u16)
    }

    /// Returns the 9-bit level 4 page table index.
    #[inline]
    pub const fn p4_index(self) -> PageTableIndex {
        PageTableIndex::new_truncate((self.0 >> 12 >> 9 >> 9 >> 9) as u16)
    }

    /// Returns the 9-bit level 5 page table index.
    ///
    /// This index is only used by 5-level paging. For a 48-bit address, this index is
    /// either `0` (lower half) or `511` (upper half).
    #[inline]
    pub const fn p5_index(self) -> PageTableIndex {
        PageTableIndex::new_truncate((self.0 >> 12 >> 9 >> 9 >> 9 >> 9) as u16)
    }

    /// Returns the 9-bit level page table index.
    #[inline]
    pub const fn page_table_index(self, level: PageTableLevel) -> PageTableIndex {
        PageTableIndex::new_truncate((self.0 >> 12 >> ((level as u8 - 1) * 9)) as u16)
    }

    /// Returns the first address of the upper half of the canonical address space.
    ///
    /// This is the first canonical address after the non-canonical “gap” in the address
    /// space.
    #[inline]
    pub(crate) const fn upper_half_start() -> Self {
        Self::new_truncate(1 << (W::BITS - 1))
    }

    /// Returns the last address of the lower half of the canonical address space.
    ///
    /// This is the last canonical address before the non-canonical “gap” in the address
    /// space.
    #[inline]
    pub(crate) const fn lower_half_end() -> Self {
        Self::new_truncate((1 << (W::BITS - 1)) - 1)
    }

    // FIXME: Move this into the `Step` impl, once `Step` is stabilized.
    #[cfg(feature = "step_trait")]
    pub(crate) fn steps_between_impl(start: &Self, end: &Self) -> (usize, Option<usize>) {
        if let Some(steps) = Self::steps_between_u64(start, end) {
            let steps = usize::try_from(steps).ok();
            (steps.unwrap_or(usize::MAX), steps)
        } else {
            (0, None)
        }
    }

    /// An implementation of steps_between that returns u64. Note that this
    /// function always returns the exact bound, so it doesn't need to return a
    /// lower and upper bound like steps_between does.
    pub(crate) fn steps_between_u64(start: &Self, end: &Self) -> Option<u64> {
        let mut steps = end.0.checked_sub(start.0)?;

        // Mask away extra bits that appear while jumping the gap.
        steps &= Self::ADDRESS_SPACE_SIZE - 1;

        Some(steps)
    }

    // FIXME: Move this into the `Step` impl, once `Step` is stabilized.
    #[inline]
    pub(crate) fn forward_checked_impl(start: Self, count: usize) -> Option<Self> {
        Self::forward_checked_u64(start, u64::try_from(count).ok()?)
    }

    /// An implementation of forward_checked that takes u64 instead of usize.
    #[inline]
    pub(crate) fn forward_checked_u64(start: Self, count: u64) -> Option<Self> {
        if count > Self::ADDRESS_SPACE_SIZE {
            return None;
        }

        let mut addr = start.0.checked_add(count)?;

        // Look at the bits starting at the most significant valid bit.
        match addr >> (W::BITS - 1) {
            0x1 => {
                // Jump the gap by sign extending the most significant valid bit.
                addr |= u64::MAX << (W::BITS - 1);
            }
            0x2 => {
                // Address overflow
                return None;
            }
            _ => {}
        }

        Some(unsafe { Self::new_unsafe(addr) })
    }

    /// An implementation of backward_checked that takes u64 instead of usize.
    #[cfg(feature = "step_trait")]
    #[inline]
    pub(crate) fn backward_checked_u64(start: Self, count: u64) -> Option<Self> {
        if count > Self::ADDRESS_SPACE_SIZE {
            return None;
        }

        let mut addr = start.0.checked_sub(count)?;

        // The value of the bits starting at the most significant valid bit for
        // addresses in the upper half of the address space (all ones).
        let upper_half = u64::MAX >> (W::BITS - 1);

        // Look at the bits starting at the most significant valid bit.
        match addr >> (W::BITS - 1) {
            bits if bits == upper_half - 1 => {
                // Jump the gap by sign extending the most significant valid bit.
                addr &= (1 << (W::BITS - 1)) - 1;
            }
            bits if bits == upper_half - 2 => {
                // Address underflow
                return None;
            }
            _ => {}
        }

        Some(unsafe { Self::new_unsafe(addr) })
    }
}

impl<W: VirtAddrWidth> fmt::Debug for VirtAddrGeneric<W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("VirtAddr")
            .field(&format_args!("{:#x}", self.0))
            .finish()
    }
}

impl<W: VirtAddrWidth> fmt::Binary for VirtAddrGeneric<W> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Binary::fmt(&self.0, f)
    }
}

impl<W: VirtAddrWidth> fmt::LowerHex for VirtAddrGeneric<W> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl<W: VirtAddrWidth> fmt::Octal for VirtAddrGeneric<W> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Octal::fmt(&self.0, f)
    }
}

impl<W: VirtAddrWidth> fmt::UpperHex for VirtAddrGeneric<W> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::UpperHex::fmt(&self.0, f)
    }
}

impl<W: VirtAddrWidth> fmt::Pointer for VirtAddrGeneric<W> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&(self.0 as *const ()), f)
    }
}

impl<W: VirtAddrWidth> Add<u64> for VirtAddrGeneric<W> {
    type Output = Self;

    #[cfg_attr(not(feature = "step_trait"), allow(rustdoc::broken_intra_doc_links))]
    /// Add an offset to a virtual address.
    ///
    /// This function performs normal arithmetic addition and doesn't jump the
    /// address gap. If you're looking for a successor operation that jumps the
    /// address gap, use [`Step::forward`].
    ///
    /// # Panics
    ///
    /// This function will panic on overflow or if the result is not a
    /// canonical address.
    #[inline]
    fn add(self, rhs: u64) -> Self::Output {
        Self::try_new(
            self.0
                .checked_add(rhs)
                .expect("attempt to add with overflow"),
        )
        .expect("attempt to add resulted in non-canonical virtual address")
    }
}

impl<W: VirtAddrWidth> AddAssign<u64> for VirtAddrGeneric<W> {
    #[cfg_attr(not(feature = "step_trait"), allow(rustdoc::broken_intra_doc_links))]
    /// Add an offset to a virtual address.
    ///
    /// This function performs normal arithmetic addition and doesn't jump the
    /// address gap. If you're looking for a successor operation that jumps the
    /// address gap, use [`Step::forward`].
    ///
    /// # Panics
    ///
    /// This function will panic on overflow or if the result is not a
    /// canonical address.
    #[inline]
    fn add_assign(&mut self, rhs: u64) {
        *self = *self + rhs;
    }
}

impl<W: VirtAddrWidth> Sub<u64> for VirtAddrGeneric<W> {
    type Output = Self;

    #[cfg_attr(not(feature = "step_trait"), allow(rustdoc::broken_intra_doc_links))]
    /// Subtract an offset from a virtual address.
    ///
    /// This function performs normal arithmetic subtraction and doesn't jump
    /// the address gap. If you're looking for a predecessor operation that
    /// jumps the address gap, use [`Step::backward`].
    ///
    /// # Panics
    ///
    /// This function will panic on overflow or if the result is not a
    /// canonical address.
    #[inline]
    fn sub(self, rhs: u64) -> Self::Output {
        Self::try_new(
            self.0
                .checked_sub(rhs)
                .expect("attempt to subtract with overflow"),
        )
        .expect("attempt to subtract resulted in non-canonical virtual address")
    }
}

impl<W: VirtAddrWidth> SubAssign<u64> for VirtAddrGeneric<W> {
    #[cfg_attr(not(feature = "step_trait"), allow(rustdoc::broken_intra_doc_links))]
    /// Subtract an offset from a virtual address.
    ///
    /// This function performs normal arithmetic subtraction and doesn't jump
    /// the address gap. If you're looking for a predecessor operation that
    /// jumps the address gap, use [`Step::backward`].
    ///
    /// # Panics
    ///
    /// This function will panic on overflow or if the result is not a
    /// canonical address.
    #[inline]
    fn sub_assign(&mut self, rhs: u64) {
        *self = *self - rhs;
    }
}

impl<W: VirtAddrWidth> Sub<VirtAddrGeneric<W>> for VirtAddrGeneric<W> {
    type Output = u64;

    /// Returns the difference between two addresses.
    ///
    /// # Panics
    ///
    /// This function will panic on overflow.
    #[inline]
    fn sub(self, rhs: VirtAddrGeneric<W>) -> Self::Output {
        self.as_u64()
            .checked_sub(rhs.as_u64())
            .expect("attempt to subtract with overflow")
    }
}

impl From<VirtAddr48> for VirtAddr57 {
    /// Widens a 48-bit virtual address to a 57-bit virtual address.
    ///
    /// Every 48-bit canonical address is also a 57-bit canonical address, so this
    /// conversion never fails and doesn't change the address value.
    #[inline]
    fn from(addr: VirtAddr48) -> Self {
        // SAFETY: A correct sign extension of bit 47 is also a correct sign extension
        // of bit 56, so the address value is also a valid 57-bit address.
        unsafe { Self::new_unsafe(addr.as_u64()) }
    }
}

impl TryFrom<VirtAddr57> for VirtAddr48 {
    type Error = VirtAddrNotValid;

    /// Narrows a 57-bit virtual address to a 48-bit virtual address.
    ///
    /// This conversion fails if the address is not a valid 48-bit canonical address,
    /// i.e. if bits 48 to 64 are not a correct sign extension of bit 47.
    #[inline]
    fn try_from(addr: VirtAddr57) -> Result<Self, Self::Error> {
        Self::try_new(addr.as_u64())
    }
}

impl RawVirtAddr {
    /// Creates a new raw virtual address from the given value.
    ///
    /// No checks are performed.
    #[inline]
    pub const fn new(addr: u64) -> Self {
        RawVirtAddr(addr)
    }

    /// Creates a raw virtual address that points to `0`.
    #[inline]
    pub const fn zero() -> Self {
        RawVirtAddr(0)
    }

    /// Converts the address to an `u64`.
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Creates a raw virtual address from the given pointer.
    #[cfg(target_pointer_width = "64")]
    #[inline]
    pub fn from_ptr<T: ?Sized>(ptr: *const T) -> Self {
        Self::new(ptr as *const () as u64)
    }

    /// Converts the address to a raw pointer.
    #[cfg(target_pointer_width = "64")]
    #[inline]
    pub const fn as_ptr<T>(self) -> *const T {
        self.as_u64() as *const T
    }

    /// Converts the address to a mutable raw pointer.
    #[cfg(target_pointer_width = "64")]
    #[inline]
    pub const fn as_mut_ptr<T>(self) -> *mut T {
        self.as_ptr::<T>() as *mut T
    }

    /// Convenience method for checking if a virtual address is null.
    #[inline]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Returns whether the address is canonical for the given [width](VirtAddrWidth).
    #[inline]
    pub const fn is_canonical<W: VirtAddrWidth>(self) -> bool {
        VirtAddrGeneric::<W>::try_new(self.0).is_ok()
    }

    /// Tries to convert the address into a canonical address of the given
    /// [width](VirtAddrWidth).
    ///
    /// Fails if the address is not canonical for that width.
    #[inline]
    pub const fn try_into_width<W: VirtAddrWidth>(
        self,
    ) -> Result<VirtAddrGeneric<W>, VirtAddrNotValid> {
        VirtAddrGeneric::try_new(self.0)
    }

    /// Tries to convert the address into a canonical 48-bit address.
    ///
    /// Fails if bits 48 to 64 are not a correct sign extension of bit 47. This is the
    /// conversion to use in kernels that run with 4-level paging.
    #[inline]
    pub const fn try_into_48(self) -> Result<VirtAddr48, VirtAddrNotValid> {
        self.try_into_width()
    }

    /// Tries to convert the address into a canonical 57-bit address.
    ///
    /// Fails if bits 57 to 64 are not a correct sign extension of bit 56. This is the
    /// conversion to use in kernels that run with 5-level paging.
    #[inline]
    pub const fn try_into_57(self) -> Result<VirtAddr57, VirtAddrNotValid> {
        self.try_into_width()
    }
}

impl<W: VirtAddrWidth> From<VirtAddrGeneric<W>> for RawVirtAddr {
    /// Discards the canonicality guarantee of a checked virtual address.
    #[inline]
    fn from(addr: VirtAddrGeneric<W>) -> Self {
        RawVirtAddr(addr.as_u64())
    }
}

impl<W: VirtAddrWidth> TryFrom<RawVirtAddr> for VirtAddrGeneric<W> {
    type Error = VirtAddrNotValid;

    /// Checks that the raw address is canonical for the width `W`.
    #[inline]
    fn try_from(addr: RawVirtAddr) -> Result<Self, Self::Error> {
        addr.try_into_width()
    }
}

impl fmt::Debug for RawVirtAddr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("RawVirtAddr")
            .field(&format_args!("{:#x}", self.0))
            .finish()
    }
}

impl fmt::Binary for RawVirtAddr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Binary::fmt(&self.0, f)
    }
}

impl fmt::LowerHex for RawVirtAddr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl fmt::Octal for RawVirtAddr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Octal::fmt(&self.0, f)
    }
}

impl fmt::UpperHex for RawVirtAddr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::UpperHex::fmt(&self.0, f)
    }
}

impl fmt::Pointer for RawVirtAddr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&(self.0 as *const ()), f)
    }
}

impl Add<u64> for RawVirtAddr {
    type Output = Self;

    /// Adds an offset to the address using plain integer arithmetic.
    ///
    /// # Panics
    ///
    /// This function panics on overflow.
    #[inline]
    fn add(self, rhs: u64) -> Self::Output {
        RawVirtAddr(
            self.0
                .checked_add(rhs)
                .expect("attempt to add with overflow"),
        )
    }
}

impl AddAssign<u64> for RawVirtAddr {
    #[inline]
    fn add_assign(&mut self, rhs: u64) {
        *self = *self + rhs;
    }
}

impl Sub<u64> for RawVirtAddr {
    type Output = Self;

    /// Subtracts an offset from the address using plain integer arithmetic.
    ///
    /// # Panics
    ///
    /// This function panics on overflow.
    #[inline]
    fn sub(self, rhs: u64) -> Self::Output {
        RawVirtAddr(
            self.0
                .checked_sub(rhs)
                .expect("attempt to subtract with overflow"),
        )
    }
}

impl SubAssign<u64> for RawVirtAddr {
    #[inline]
    fn sub_assign(&mut self, rhs: u64) {
        *self = *self - rhs;
    }
}

impl Sub<RawVirtAddr> for RawVirtAddr {
    type Output = u64;

    /// Returns the difference between two addresses.
    ///
    /// # Panics
    ///
    /// This function panics on overflow.
    #[inline]
    fn sub(self, rhs: RawVirtAddr) -> Self::Output {
        self.0
            .checked_sub(rhs.0)
            .expect("attempt to subtract with overflow")
    }
}

#[cfg(feature = "step_trait")]
impl<W: VirtAddrWidth> Step for VirtAddrGeneric<W> {
    #[inline]
    fn steps_between(start: &Self, end: &Self) -> (usize, Option<usize>) {
        Self::steps_between_impl(start, end)
    }

    #[inline]
    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        Self::forward_checked_impl(start, count)
    }

    #[inline]
    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        Self::backward_checked_u64(start, u64::try_from(count).ok()?)
    }

    #[inline]
    fn forward_overflowing(start: Self, count: usize) -> (Self, bool) {
        match Self::forward_checked(start, count) {
            Some(next) => (next, false),
            None => (start, true),
        }
    }

    #[inline]
    fn backward_overflowing(start: Self, count: usize) -> (Self, bool) {
        match Self::backward_checked(start, count) {
            Some(next) => (next, false),
            None => (start, true),
        }
    }
}

#[cfg(kani)]
impl<W: VirtAddrWidth> kani::Arbitrary for VirtAddrGeneric<W> {
    fn any() -> Self {
        Self::new_truncate(kani::any())
    }
}

/// A passed `u64` was not a valid physical address.
///
/// This means that bits 52 to 64 were not all null.
///
/// Contains the invalid address.
pub struct PhysAddrNotValid(pub u64);

impl core::fmt::Debug for PhysAddrNotValid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PhysAddrNotValid")
            .field(&format_args!("{:#x}", self.0))
            .finish()
    }
}

impl PhysAddr {
    /// Creates a new physical address.
    ///
    /// ## Panics
    ///
    /// This function panics if a bit in the range 52 to 64 is set.
    ///
    /// If the `memory_encryption` feature has been enabled and an encryption bit has been
    /// configured, this also panics if the encryption bit is manually set in the address.
    #[inline]
    #[const_fn(cfg(not(feature = "memory_encryption")))]
    pub const fn new(addr: u64) -> Self {
        // TODO: Replace with .ok().expect(msg) when that works on stable.
        match Self::try_new(addr) {
            Ok(p) => p,
            Err(_) => panic!("physical addresses must not have any bits in the range 52 to 64 set"),
        }
    }

    /// Creates a new physical address, throwing bits 52..64 away.
    #[cfg(not(feature = "memory_encryption"))]
    #[inline]
    pub const fn new_truncate(addr: u64) -> PhysAddr {
        PhysAddr(addr % (1 << 52))
    }

    /// Creates a new physical address, throwing bits 52..64 and the encryption bit away.
    #[cfg(feature = "memory_encryption")]
    #[inline]
    pub fn new_truncate(addr: u64) -> PhysAddr {
        PhysAddr(addr & PHYSICAL_ADDRESS_MASK.load(Ordering::Relaxed))
    }

    /// Creates a new physical address, without any checks.
    ///
    /// ## Safety
    ///
    /// You must make sure bits 52..64 are zero and that no bits at or above
    /// the encryption bit (if one is configured) are set. This is not checked.
    #[inline]
    pub const unsafe fn new_unsafe(addr: u64) -> PhysAddr {
        PhysAddr(addr)
    }

    /// Tries to create a new physical address.
    ///
    /// Fails if any bits in the range 52 to 64 are set.
    /// If the `memory_encryption` feature has been enabled and an encryption bit has been
    /// configured, this also fails if the encryption bit is manually set in the address.
    #[inline]
    #[const_fn(cfg(not(feature = "memory_encryption")))]
    pub const fn try_new(addr: u64) -> Result<Self, PhysAddrNotValid> {
        let p = Self::new_truncate(addr);
        if p.0 == addr {
            Ok(p)
        } else {
            Err(PhysAddrNotValid(addr))
        }
    }

    /// Creates a physical address that points to `0`.
    #[inline]
    pub const fn zero() -> PhysAddr {
        PhysAddr(0)
    }

    /// Converts the address to an `u64`.
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Convenience method for checking if a physical address is null.
    #[inline]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Aligns the physical address upwards to the given alignment.
    ///
    /// See the `align_up` function for more information.
    ///
    /// # Panics
    ///
    /// This function panics if the resulting address has a bit in the range 52
    /// to 64 set.
    #[inline]
    pub fn align_up<U>(self, align: U) -> Self
    where
        U: Into<u64>,
    {
        PhysAddr::new(align_up(self.0, align.into()))
    }

    /// Aligns the physical address downwards to the given alignment.
    ///
    /// See the `align_down` function for more information.
    #[inline]
    pub fn align_down<U>(self, align: U) -> Self
    where
        U: Into<u64>,
    {
        self.align_down_u64(align.into())
    }

    /// Aligns the physical address downwards to the given alignment.
    ///
    /// See the `align_down` function for more information.
    #[inline]
    pub(crate) const fn align_down_u64(self, align: u64) -> Self {
        PhysAddr(align_down(self.0, align))
    }

    /// Checks whether the physical address has the demanded alignment.
    #[inline]
    pub fn is_aligned<U>(self, align: U) -> bool
    where
        U: Into<u64>,
    {
        self.is_aligned_u64(align.into())
    }

    /// Checks whether the physical address has the demanded alignment.
    #[inline]
    pub(crate) const fn is_aligned_u64(self, align: u64) -> bool {
        self.align_down_u64(align).as_u64() == self.as_u64()
    }
}

impl fmt::Debug for PhysAddr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("PhysAddr")
            .field(&format_args!("{:#x}", self.0))
            .finish()
    }
}

impl fmt::Binary for PhysAddr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Binary::fmt(&self.0, f)
    }
}

impl fmt::LowerHex for PhysAddr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl fmt::Octal for PhysAddr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Octal::fmt(&self.0, f)
    }
}

impl fmt::UpperHex for PhysAddr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::UpperHex::fmt(&self.0, f)
    }
}

impl fmt::Pointer for PhysAddr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&(self.0 as *const ()), f)
    }
}

impl Add<u64> for PhysAddr {
    type Output = Self;
    #[inline]
    fn add(self, rhs: u64) -> Self::Output {
        PhysAddr::new(self.0.checked_add(rhs).unwrap())
    }
}

impl AddAssign<u64> for PhysAddr {
    #[inline]
    fn add_assign(&mut self, rhs: u64) {
        *self = *self + rhs;
    }
}

impl Sub<u64> for PhysAddr {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: u64) -> Self::Output {
        PhysAddr::new(self.0.checked_sub(rhs).unwrap())
    }
}

impl SubAssign<u64> for PhysAddr {
    #[inline]
    fn sub_assign(&mut self, rhs: u64) {
        *self = *self - rhs;
    }
}

impl Sub<PhysAddr> for PhysAddr {
    type Output = u64;
    #[inline]
    fn sub(self, rhs: PhysAddr) -> Self::Output {
        self.as_u64().checked_sub(rhs.as_u64()).unwrap()
    }
}

#[cfg(kani)]
impl kani::Arbitrary for PhysAddr {
    fn any() -> Self {
        Self::new_truncate(kani::any())
    }
}

/// Align address downwards.
///
/// Returns the greatest `x` with alignment `align` so that `x <= addr`.
///
/// Panics if the alignment is not a power of two.
#[inline]
pub const fn align_down(addr: u64, align: u64) -> u64 {
    assert!(align.is_power_of_two(), "`align` must be a power of two");
    addr & !(align - 1)
}

/// Align address upwards.
///
/// Returns the smallest `x` with alignment `align` so that `x >= addr`.
///
/// Panics if the alignment is not a power of two or if an overflow occurs.
#[inline]
pub const fn align_up(addr: u64, align: u64) -> u64 {
    assert!(align.is_power_of_two(), "`align` must be a power of two");
    let align_mask = align - 1;
    if addr & align_mask == 0 {
        addr // already aligned
    } else {
        // FIXME: Replace with .expect, once `Option::expect` is const.
        if let Some(aligned) = (addr | align_mask).checked_add(1) {
            aligned
        } else {
            panic!("attempt to add with overflow")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constructs a `VirtAddr` without checks, mirroring the tuple constructor that was
    /// available before `VirtAddr` became an alias.
    #[allow(non_snake_case)]
    fn VirtAddr(addr: u64) -> VirtAddr {
        unsafe { VirtAddr::new_unsafe(addr) }
    }

    /// Constructs a `VirtAddr57` without checks.
    #[allow(non_snake_case)]
    fn VirtAddr57(addr: u64) -> VirtAddr57 {
        unsafe { VirtAddr57::new_unsafe(addr) }
    }

    #[test]
    fn virtaddr_is_48_bit() {
        let _: fn(u64) -> VirtAddr48 = VirtAddr::new;
        assert_eq!(VirtAddr::BITS, 48);
        assert_eq!(VirtAddr48::BITS, 48);
        assert_eq!(VirtAddr57::BITS, 57);
    }

    #[test]
    fn virtaddr_layout_is_transparent() {
        assert_eq!(
            core::mem::size_of::<VirtAddr48>(),
            core::mem::size_of::<u64>()
        );
        assert_eq!(
            core::mem::size_of::<VirtAddr57>(),
            core::mem::size_of::<u64>()
        );
        assert_eq!(
            core::mem::align_of::<VirtAddr48>(),
            core::mem::align_of::<u64>()
        );
        assert_eq!(
            core::mem::align_of::<VirtAddr57>(),
            core::mem::align_of::<u64>()
        );
    }

    #[test]
    fn virtaddr_const_construction() {
        const ADDR48: VirtAddr48 = VirtAddr48::new(0xffff_8000_0000_1234);
        const ADDR57: VirtAddr57 = VirtAddr57::new(0xff80_0000_0000_1234);
        const TRUNCATED57: VirtAddr57 = VirtAddr57::new_truncate(1 << 56);
        assert_eq!(ADDR48.as_u64(), 0xffff_8000_0000_1234);
        assert_eq!(ADDR57.as_u64(), 0xff80_0000_0000_1234);
        assert_eq!(TRUNCATED57.as_u64(), 0xff00_0000_0000_0000);
    }

    #[test]
    fn virtaddr_canonicality() {
        assert!(VirtAddr48::try_new(0x0000_7fff_ffff_ffff).is_ok());
        assert!(VirtAddr48::try_new(0x0000_8000_0000_0000).is_err());
        assert!(VirtAddr48::try_new(0xffff_7fff_ffff_ffff).is_err());
        assert!(VirtAddr48::try_new(0xffff_8000_0000_0000).is_ok());

        assert!(VirtAddr57::try_new(0x0000_7fff_ffff_ffff).is_ok());
        assert!(VirtAddr57::try_new(0x0000_8000_0000_0000).is_ok());
        assert!(VirtAddr57::try_new(0x00ff_ffff_ffff_ffff).is_ok());
        assert!(VirtAddr57::try_new(0x0100_0000_0000_0000).is_err());
        assert!(VirtAddr57::try_new(0xfeff_ffff_ffff_ffff).is_err());
        assert!(VirtAddr57::try_new(0xff00_0000_0000_0000).is_ok());
        assert!(VirtAddr57::try_new(0xffff_8000_0000_0000).is_ok());
    }

    #[test]
    fn virtaddr_conversions() {
        let addr48 = VirtAddr48::new(0xffff_8000_0000_1234);
        let addr57 = VirtAddr57::from(addr48);
        assert_eq!(addr57.as_u64(), addr48.as_u64());
        assert_eq!(VirtAddr48::try_from(addr57).unwrap(), addr48);

        let addr57: VirtAddr57 = VirtAddr48::new(0x1234).into();
        assert_eq!(addr57.as_u64(), 0x1234);

        let only57 = VirtAddr57::new(0x0000_8000_0000_0000);
        assert!(VirtAddr48::try_from(only57).is_err());
        let only57 = VirtAddr57::new(0xff00_0000_0000_0000);
        assert!(VirtAddr48::try_from(only57).is_err());
    }

    #[test]
    fn raw_virtaddr_conversions() {
        let raw = RawVirtAddr::new(0xffff_8000_0000_1234);
        assert_eq!(raw.try_into_48().unwrap(), VirtAddr48::new(raw.as_u64()));
        assert_eq!(raw.try_into_57().unwrap(), VirtAddr57::new(raw.as_u64()));
        assert!(raw.is_canonical::<Width48>());
        assert!(raw.is_canonical::<Width57>());

        let only57 = RawVirtAddr::new(0x0000_8000_0000_0000);
        assert!(only57.try_into_48().is_err());
        assert!(VirtAddr48::try_from(only57).is_err());
        assert_eq!(only57.try_into_57().unwrap().as_u64(), only57.as_u64());
        assert!(!only57.is_canonical::<Width48>());
        assert!(only57.is_canonical::<Width57>());

        let invalid = RawVirtAddr::new(0x0100_0000_0000_0000);
        assert!(invalid.try_into_48().is_err());
        assert!(invalid.try_into_57().is_err());

        let from48: RawVirtAddr = VirtAddr48::new(0x1000).into();
        let from57: RawVirtAddr = VirtAddr57::new(0x1000).into();
        assert_eq!(from48, from57);
        assert_eq!(from48.as_u64(), 0x1000);

        const RAW: RawVirtAddr = RawVirtAddr::new(0x42);
        const CHECKED: Result<VirtAddr48, VirtAddrNotValid> = RAW.try_into_48();
        assert!(CHECKED.is_ok());
    }

    #[test]
    fn raw_virtaddr_arithmetic() {
        // Raw arithmetic doesn't jump or check the gap.
        assert_eq!(
            RawVirtAddr::new(0x7fff_ffff_ffff) + 1,
            RawVirtAddr::new(0x8000_0000_0000)
        );
        assert_eq!(RawVirtAddr::new(0x2000) - RawVirtAddr::new(0x1000), 0x1000);
        let mut addr = RawVirtAddr::new(0x1000);
        addr += 2;
        addr -= 1;
        assert_eq!(addr.as_u64(), 0x1001);
    }

    #[test]
    #[should_panic]
    fn raw_virtaddr_add_overflow() {
        let _ = RawVirtAddr::new(u64::MAX) + 1;
    }

    #[test]
    fn virtaddr_page_table_indices() {
        let addr = VirtAddr57::new(0xff00_0000_0000_0000);
        assert_eq!(u16::from(addr.p5_index()), 0x100);
        assert_eq!(u16::from(addr.p4_index()), 0);
        let addr = VirtAddr57::new(0xffff_8000_0000_0000);
        assert_eq!(u16::from(addr.p5_index()), 0x1ff);
        assert_eq!(u16::from(addr.p4_index()), 0x100);
        let addr = VirtAddr57::new(0x00ff_8000_0000_0000);
        assert_eq!(u16::from(addr.p5_index()), 0xff);
        assert_eq!(u16::from(addr.p4_index()), 0x100);

        let addr = VirtAddr48::new(0xffff_8000_0000_0000);
        assert_eq!(u16::from(addr.p5_index()), 0x1ff);
        assert_eq!(u16::from(addr.p4_index()), 0x100);
        let addr = VirtAddr48::new(0x0000_7fff_ffff_ffff);
        assert_eq!(u16::from(addr.p5_index()), 0);
        assert_eq!(u16::from(addr.p4_index()), 0xff);
    }

    #[test]
    #[should_panic]
    pub fn add_overflow_virtaddr() {
        let _ = VirtAddr::new(0xffff_ffff_ffff_ffff) + 1;
    }

    #[test]
    #[should_panic]
    pub fn add_overflow_physaddr() {
        let _ = PhysAddr::new(0x000f_ffff_ffff_ffff) + 0xffff_0000_0000_0000;
    }

    #[test]
    #[should_panic]
    pub fn sub_underflow_virtaddr() {
        let _ = VirtAddr::new(0) - 1;
    }

    #[test]
    #[should_panic]
    pub fn sub_overflow_physaddr() {
        let _ = PhysAddr::new(0) - 1;
    }

    #[test]
    #[should_panic = "attempt to add resulted in non-canonical virtual address"]
    pub fn add_into_gap_virtaddr57() {
        let _ = VirtAddr57::new(0x00ff_ffff_ffff_ffff) + 1;
    }

    #[test]
    pub fn virtaddr_new_truncate() {
        assert_eq!(VirtAddr::new_truncate(0), VirtAddr(0));
        assert_eq!(VirtAddr::new_truncate(1 << 47), VirtAddr(0xfffff << 47));
        assert_eq!(VirtAddr::new_truncate(123), VirtAddr(123));
        assert_eq!(VirtAddr::new_truncate(123 << 47), VirtAddr(0xfffff << 47));
    }

    #[test]
    pub fn virtaddr57_new_truncate() {
        assert_eq!(VirtAddr57::new_truncate(0), VirtAddr57(0));
        assert_eq!(VirtAddr57::new_truncate(1 << 47), VirtAddr57(1 << 47));
        assert_eq!(VirtAddr57::new_truncate(1 << 56), VirtAddr57(0xff << 56));
        assert_eq!(VirtAddr57::new_truncate(123), VirtAddr57(123));
        assert_eq!(VirtAddr57::new_truncate(123 << 56), VirtAddr57(0xff << 56));
        assert_eq!(VirtAddr57::new_truncate(122 << 56), VirtAddr57(0));
    }

    #[test]
    #[cfg(feature = "step_trait")]
    fn virtaddr_step_forward() {
        assert_eq!(Step::forward(VirtAddr(0), 0), VirtAddr(0));
        assert_eq!(Step::forward(VirtAddr(0), 1), VirtAddr(1));
        assert_eq!(
            Step::forward(VirtAddr(0x7fff_ffff_ffff), 1),
            VirtAddr(0xffff_8000_0000_0000)
        );
        assert_eq!(
            Step::forward(VirtAddr(0xffff_8000_0000_0000), 1),
            VirtAddr(0xffff_8000_0000_0001)
        );
        assert_eq!(
            Step::forward_checked(VirtAddr(0xffff_ffff_ffff_ffff), 1),
            None
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::forward(VirtAddr(0x7fff_ffff_ffff), 0x1234_5678_9abd),
            VirtAddr(0xffff_9234_5678_9abc)
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::forward(VirtAddr(0x7fff_ffff_ffff), 0x8000_0000_0000),
            VirtAddr(0xffff_ffff_ffff_ffff)
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::forward(VirtAddr(0x7fff_ffff_ff00), 0x8000_0000_00ff),
            VirtAddr(0xffff_ffff_ffff_ffff)
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::forward_checked(VirtAddr(0x7fff_ffff_ff00), 0x8000_0000_0100),
            None
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::forward_checked(VirtAddr(0x7fff_ffff_ffff), 0x8000_0000_0001),
            None
        );
    }

    #[test]
    #[cfg(feature = "step_trait")]
    fn virtaddr57_step_forward() {
        assert_eq!(Step::forward(VirtAddr57(0), 0), VirtAddr57(0));
        assert_eq!(Step::forward(VirtAddr57(0), 1), VirtAddr57(1));
        // The 48-bit gap is not a gap for 57-bit addresses.
        assert_eq!(
            Step::forward(VirtAddr57(0x7fff_ffff_ffff), 1),
            VirtAddr57(0x8000_0000_0000)
        );
        assert_eq!(
            Step::forward(VirtAddr57(0x00ff_ffff_ffff_ffff), 1),
            VirtAddr57(0xff00_0000_0000_0000)
        );
        assert_eq!(
            Step::forward(VirtAddr57(0xff00_0000_0000_0000), 1),
            VirtAddr57(0xff00_0000_0000_0001)
        );
        assert_eq!(
            Step::forward_checked(VirtAddr57(0xffff_ffff_ffff_ffff), 1),
            None
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::forward(VirtAddr57(0x00ff_ffff_ffff_ffff), 0x0012_3456_789a_bcdf),
            VirtAddr57(0xff12_3456_789a_bcde)
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::forward(VirtAddr57(0x00ff_ffff_ffff_ffff), 0x0100_0000_0000_0000),
            VirtAddr57(0xffff_ffff_ffff_ffff)
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::forward_checked(VirtAddr57(0x00ff_ffff_ffff_ffff), 0x0100_0000_0000_0001),
            None
        );
    }

    #[test]
    #[cfg(feature = "step_trait")]
    fn virtaddr_step_backward() {
        assert_eq!(Step::backward(VirtAddr(0), 0), VirtAddr(0));
        assert_eq!(Step::backward_checked(VirtAddr(0), 1), None);
        assert_eq!(Step::backward(VirtAddr(1), 1), VirtAddr(0));
        assert_eq!(
            Step::backward(VirtAddr(0xffff_8000_0000_0000), 1),
            VirtAddr(0x7fff_ffff_ffff)
        );
        assert_eq!(
            Step::backward(VirtAddr(0xffff_8000_0000_0001), 1),
            VirtAddr(0xffff_8000_0000_0000)
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::backward(VirtAddr(0xffff_9234_5678_9abc), 0x1234_5678_9abd),
            VirtAddr(0x7fff_ffff_ffff)
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::backward(VirtAddr(0xffff_8000_0000_0000), 0x8000_0000_0000),
            VirtAddr(0)
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::backward(VirtAddr(0xffff_8000_0000_0000), 0x7fff_ffff_ff01),
            VirtAddr(0xff)
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::backward_checked(VirtAddr(0xffff_8000_0000_0000), 0x8000_0000_0001),
            None
        );
    }

    #[test]
    #[cfg(feature = "step_trait")]
    fn virtaddr57_step_backward() {
        assert_eq!(Step::backward(VirtAddr57(0), 0), VirtAddr57(0));
        assert_eq!(Step::backward_checked(VirtAddr57(0), 1), None);
        assert_eq!(Step::backward(VirtAddr57(1), 1), VirtAddr57(0));
        // The 48-bit gap is not a gap for 57-bit addresses.
        assert_eq!(
            Step::backward(VirtAddr57(0xffff_8000_0000_0000), 1),
            VirtAddr57(0xffff_7fff_ffff_ffff)
        );
        assert_eq!(
            Step::backward(VirtAddr57(0xff00_0000_0000_0000), 1),
            VirtAddr57(0x00ff_ffff_ffff_ffff)
        );
        assert_eq!(
            Step::backward(VirtAddr57(0xff00_0000_0000_0001), 1),
            VirtAddr57(0xff00_0000_0000_0000)
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::backward(VirtAddr57(0xff12_3456_789a_bcde), 0x0012_3456_789a_bcdf),
            VirtAddr57(0x00ff_ffff_ffff_ffff)
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::backward(VirtAddr57(0xff00_0000_0000_0000), 0x0100_0000_0000_0000),
            VirtAddr57(0)
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::backward_checked(VirtAddr57(0xff00_0000_0000_0000), 0x0100_0000_0000_0001),
            None
        );
    }

    #[test]
    #[cfg(feature = "step_trait")]
    fn virtaddr_steps_between() {
        assert_eq!(
            Step::steps_between(&VirtAddr(0), &VirtAddr(0)),
            (0, Some(0))
        );
        assert_eq!(
            Step::steps_between(&VirtAddr(0), &VirtAddr(1)),
            (1, Some(1))
        );
        assert_eq!(Step::steps_between(&VirtAddr(1), &VirtAddr(0)), (0, None));
        assert_eq!(
            Step::steps_between(
                &VirtAddr(0x7fff_ffff_ffff),
                &VirtAddr(0xffff_8000_0000_0000)
            ),
            (1, Some(1))
        );
        assert_eq!(
            Step::steps_between(
                &VirtAddr(0xffff_8000_0000_0000),
                &VirtAddr(0x7fff_ffff_ffff)
            ),
            (0, None)
        );
        assert_eq!(
            Step::steps_between(
                &VirtAddr(0xffff_8000_0000_0000),
                &VirtAddr(0xffff_8000_0000_0000)
            ),
            (0, Some(0))
        );
        assert_eq!(
            Step::steps_between(
                &VirtAddr(0xffff_8000_0000_0000),
                &VirtAddr(0xffff_8000_0000_0001)
            ),
            (1, Some(1))
        );
        assert_eq!(
            Step::steps_between(
                &VirtAddr(0xffff_8000_0000_0001),
                &VirtAddr(0xffff_8000_0000_0000)
            ),
            (0, None)
        );
        // Make sure that we handle `steps > u32::MAX` correctly on 32-bit
        // targets. On 64-bit targets, `0x1_0000_0000` fits into `usize`, so we
        // can return exact lower and upper bounds. On 32-bit targets,
        // `0x1_0000_0000` doesn't fit into `usize`, so we only return an lower
        // bound of `usize::MAX` and don't return an upper bound.
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::steps_between(&VirtAddr(0), &VirtAddr(0x1_0000_0000)),
            (0x1_0000_0000, Some(0x1_0000_0000))
        );
        #[cfg(not(target_pointer_width = "64"))]
        assert_eq!(
            Step::steps_between(&VirtAddr(0), &VirtAddr(0x1_0000_0000)),
            (usize::MAX, None)
        );
    }

    #[test]
    #[cfg(feature = "step_trait")]
    fn virtaddr57_steps_between() {
        assert_eq!(
            Step::steps_between(&VirtAddr57(0), &VirtAddr57(1)),
            (1, Some(1))
        );
        assert_eq!(
            Step::steps_between(&VirtAddr57(1), &VirtAddr57(0)),
            (0, None)
        );
        assert_eq!(
            Step::steps_between(
                &VirtAddr57(0x00ff_ffff_ffff_ffff),
                &VirtAddr57(0xff00_0000_0000_0000)
            ),
            (1, Some(1))
        );
        assert_eq!(
            Step::steps_between(
                &VirtAddr57(0xff00_0000_0000_0000),
                &VirtAddr57(0x00ff_ffff_ffff_ffff)
            ),
            (0, None)
        );
        // The 48-bit gap is not a gap for 57-bit addresses, but the 57-bit gap
        // between the two addresses is skipped. The number of steps doesn't fit
        // into `usize` on 32-bit targets.
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Step::steps_between(
                &VirtAddr57(0x7fff_ffff_ffff),
                &VirtAddr57(0xffff_8000_0000_0000)
            ),
            (0x01ff_0000_0000_0001, Some(0x01ff_0000_0000_0001))
        );
        #[cfg(not(target_pointer_width = "64"))]
        assert_eq!(
            Step::steps_between(
                &VirtAddr57(0x7fff_ffff_ffff),
                &VirtAddr57(0xffff_8000_0000_0000)
            ),
            (usize::MAX, None)
        );
    }

    #[test]
    #[cfg(feature = "step_trait")]
    fn virtaddr_step_overflowing() {
        assert_eq!(
            Step::forward_overflowing(VirtAddr(0x7fff_ffff_ffff), 1),
            (VirtAddr(0xffff_8000_0000_0000), false)
        );
        assert_eq!(
            Step::backward_overflowing(VirtAddr(0xffff_8000_0000_0000), 1),
            (VirtAddr(0x7fff_ffff_ffff), false)
        );
        assert_eq!(
            Step::forward_overflowing(VirtAddr(0), 0),
            (VirtAddr(0), false)
        );

        assert!(Step::forward_overflowing(VirtAddr(0xffff_ffff_ffff_ffff), 1).1);
        assert!(Step::backward_overflowing(VirtAddr(0), 1).1);
    }

    #[test]
    pub fn test_align_up() {
        // align 1
        assert_eq!(align_up(0, 1), 0);
        assert_eq!(align_up(1234, 1), 1234);
        assert_eq!(align_up(0xffff_ffff_ffff_ffff, 1), 0xffff_ffff_ffff_ffff);
        // align 2
        assert_eq!(align_up(0, 2), 0);
        assert_eq!(align_up(1233, 2), 1234);
        assert_eq!(align_up(0xffff_ffff_ffff_fffe, 2), 0xffff_ffff_ffff_fffe);
        // address 0
        assert_eq!(align_up(0, 128), 0);
        assert_eq!(align_up(0, 1), 0);
        assert_eq!(align_up(0, 2), 0);
        assert_eq!(align_up(0, 0x8000_0000_0000_0000), 0);
    }

    #[test]
    fn test_virt_addr_align_up() {
        // Make sure the 47th bit is extended.
        assert_eq!(
            VirtAddr::new(0x7fff_ffff_ffff).align_up(2u64),
            VirtAddr::new(0xffff_8000_0000_0000)
        );
        // Make sure the 56th bit is extended.
        assert_eq!(
            VirtAddr57::new(0x00ff_ffff_ffff_ffff).align_up(2u64),
            VirtAddr57::new(0xff00_0000_0000_0000)
        );
    }

    #[test]
    fn test_virt_addr_align_down() {
        // Make sure the 47th bit is extended.
        assert_eq!(
            VirtAddr::new(0xffff_8000_0000_0000).align_down(1u64 << 48),
            VirtAddr::new(0)
        );
        // Make sure the 56th bit is extended.
        assert_eq!(
            VirtAddr57::new(0xff00_0000_0000_0000).align_down(1u64 << 57),
            VirtAddr57::new(0)
        );
    }

    #[test]
    #[should_panic]
    fn test_virt_addr_align_up_overflow() {
        VirtAddr::new(0xffff_ffff_ffff_ffff).align_up(2u64);
    }

    #[test]
    #[should_panic]
    fn test_phys_addr_align_up_overflow() {
        PhysAddr::new(0x000f_ffff_ffff_ffff).align_up(2u64);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn test_from_ptr_array() {
        let slice = &[1, 2, 3, 4, 5];
        // Make sure that from_ptr(slice) is the address of the first element
        assert_eq!(
            VirtAddr::from_ptr(slice.as_slice()),
            VirtAddr::from_ptr(&slice[0])
        );
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    // The next two proof harnesses prove the correctness of the `forward`
    // implementation of VirtAddr.

    // This harness proves that our implementation can correctly take 0 or 1
    // step starting from any address.
    #[kani::proof]
    fn forward_base_case() {
        let start = kani::any::<VirtAddr>();
        let start_raw = start.as_u64();

        // Adding 0 to any address should always yield the same address.
        let same = Step::forward(start, 0);
        assert!(start == same);

        // Manually calculate the expected address after stepping once.
        let expected = match start_raw {
            // Adding 1 to addresses in this range don't require gap jumps, so
            // we can just add 1.
            0x0000_0000_0000_0000..=0x0000_7fff_ffff_fffe => Some(start_raw + 1),
            // Adding 1 to this address jumps the gap.
            0x0000_7fff_ffff_ffff => Some(0xffff_8000_0000_0000),
            // The range of non-canonical addresses.
            0x0000_8000_0000_0000..=0xffff_7fff_ffff_ffff => unreachable!(),
            // Adding 1 to addresses in this range don't require gap jumps, so
            // we can just add 1.
            0xffff_8000_0000_0000..=0xffff_ffff_ffff_fffe => Some(start_raw + 1),
            // Adding 1 to this address causes an overflow.
            0xffff_ffff_ffff_ffff => None,
        };
        if let Some(expected) = expected {
            // Verify that `expected` is a valid address.
            assert!(VirtAddr::try_new(expected).is_ok());
        }
        // Verify `forward_checked`.
        let next = Step::forward_checked(start, 1);
        assert!(next.map(VirtAddr::as_u64) == expected);
    }

    // This harness proves that our implementation can correctly take 0 or 1
    // step starting from any 57-bit address.
    #[kani::proof]
    fn forward_base_case_57() {
        let start = kani::any::<VirtAddr57>();
        let start_raw = start.as_u64();

        // Adding 0 to any address should always yield the same address.
        let same = Step::forward(start, 0);
        assert!(start == same);

        // Manually calculate the expected address after stepping once.
        let expected = match start_raw {
            // Adding 1 to addresses in this range don't require gap jumps, so
            // we can just add 1.
            0x0000_0000_0000_0000..=0x00ff_ffff_ffff_fffe => Some(start_raw + 1),
            // Adding 1 to this address jumps the gap.
            0x00ff_ffff_ffff_ffff => Some(0xff00_0000_0000_0000),
            // The range of non-canonical addresses.
            0x0100_0000_0000_0000..=0xfeff_ffff_ffff_ffff => unreachable!(),
            // Adding 1 to addresses in this range don't require gap jumps, so
            // we can just add 1.
            0xff00_0000_0000_0000..=0xffff_ffff_ffff_fffe => Some(start_raw + 1),
            // Adding 1 to this address causes an overflow.
            0xffff_ffff_ffff_ffff => None,
        };
        if let Some(expected) = expected {
            // Verify that `expected` is a valid address.
            assert!(VirtAddr57::try_new(expected).is_ok());
        }
        // Verify `forward_checked`.
        let next = Step::forward_checked(start, 1);
        assert!(next.map(VirtAddr57::as_u64) == expected);
    }

    // This harness proves that the result of taking two small steps is the
    // same as taking one combined large step.
    #[kani::proof]
    fn forward_induction_step() {
        let start = kani::any::<VirtAddr>();

        let count1: usize = kani::any();
        let count2: usize = kani::any();
        // If we can take two small steps...
        let Some(next1) = Step::forward_checked(start, count1) else {
            return;
        };
        let Some(next2) = Step::forward_checked(next1, count2) else {
            return;
        };

        // ...then we can also take one combined large step.
        let count_both = count1 + count2;
        let next_both = Step::forward(start, count_both);
        assert!(next2 == next_both);
    }

    // The next two proof harnesses prove the correctness of the `backward`
    // implementation of VirtAddr using the `forward` implementation which
    // we've already proven to be correct.
    // They do this by proving the symmetry between those two functions.

    // This harness proves the correctness of the implementation of `backward`
    // for all inputs for which `forward_checked` succeeds.
    #[kani::proof]
    fn forward_implies_backward() {
        let start = kani::any::<VirtAddr>();
        let count: usize = kani::any();

        // If `forward_checked` succeeds...
        let Some(end) = Step::forward_checked(start, count) else {
            return;
        };

        // ...then `backward` succeeds as well.
        let start2 = Step::backward(end, count);
        assert!(start == start2);
    }

    // This harness proves that for all inputs for which `backward_checked`
    // succeeds, `forward` succeeds as well.
    #[kani::proof]
    fn backward_implies_forward() {
        let end = kani::any::<VirtAddr>();
        let count: usize = kani::any();

        // If `backward_checked` succeeds...
        let Some(start) = Step::backward_checked(end, count) else {
            return;
        };

        // ...then `forward` succeeds as well.
        let end2 = Step::forward(start, count);
        assert!(end == end2);
    }

    // The next two proof harnesses prove the correctness of the
    // `steps_between` implementation of VirtAddr using the `forward`
    // implementation which we've already proven to be correct.
    // They do this by proving the symmetry between those two functions.

    // This harness proves the correctness of the implementation of
    // `steps_between` for all inputs for which `forward_checked` succeeds.
    #[kani::proof]
    fn forward_implies_steps_between() {
        let start = kani::any::<VirtAddr>();
        let count: usize = kani::any();

        // If `forward_checked` succeeds...
        let Some(end) = Step::forward_checked(start, count) else {
            return;
        };

        // ...then `steps_between` succeeds as well.
        assert!(Step::steps_between(&start, &end) == (count, Some(count)));
    }

    // This harness proves that for all inputs for which `steps_between`
    // succeeds, `forward` succeeds as well.
    #[kani::proof]
    fn steps_between_implies_forward() {
        let start = kani::any::<VirtAddr>();
        let end = kani::any::<VirtAddr>();

        // If `steps_between` succeeds...
        let Some(count) = Step::steps_between(&start, &end).1 else {
            return;
        };

        // ...then `forward` succeeds as well.
        assert!(Step::forward(start, count) == end);
    }
}
