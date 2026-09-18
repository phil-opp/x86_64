//! Abstractions for default-sized and huge virtual memory pages.

use crate::VirtAddr;
use crate::sealed::Sealed;
use crate::structures::paging::PageTableIndex;
use crate::structures::paging::page_table::PageTableLevel;
use core::convert::TryFrom;
use core::fmt;
#[cfg(feature = "step_trait")]
use core::iter::Step;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Sub, SubAssign};

/// Trait for abstracting over the three possible page sizes on x86_64, 4KiB, 2MiB, 1GiB.
pub trait PageSize: Copy + Eq + PartialOrd + Ord + Sealed {
    /// The page size in bytes.
    const SIZE: u64;

    /// A string representation of the page size for debug output.
    const DEBUG_STR: &'static str;
}

/// This trait is implemented for 4KiB and 2MiB pages, but not for 1GiB pages.
pub trait NotGiantPageSize: PageSize {}

/// A standard 4KiB page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Size4KiB {}

/// A “huge” 2MiB page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Size2MiB {}

/// A “giant” 1GiB page.
///
/// (Only available on newer x86_64 CPUs.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Size1GiB {}

impl PageSize for Size4KiB {
    const SIZE: u64 = 4096;
    const DEBUG_STR: &'static str = "4KiB";
}

impl NotGiantPageSize for Size4KiB {}

impl Sealed for super::Size4KiB {}

impl PageSize for Size2MiB {
    const SIZE: u64 = Size4KiB::SIZE * 512;
    const DEBUG_STR: &'static str = "2MiB";
}

impl NotGiantPageSize for Size2MiB {}

impl Sealed for super::Size2MiB {}

impl PageSize for Size1GiB {
    const SIZE: u64 = Size2MiB::SIZE * 512;
    const DEBUG_STR: &'static str = "1GiB";
}

impl Sealed for super::Size1GiB {}

/// A virtual memory page.
///
/// # Representation
///
/// This struct has the same representation as a [`u64`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Page<S: PageSize = Size4KiB> {
    start_address: VirtAddr,
    size: PhantomData<S>,
}

impl<S: PageSize> Page<S> {
    /// The page size in bytes.
    pub const SIZE: u64 = S::SIZE;

    /// Returns the page that starts at the given virtual address.
    ///
    /// Returns an error if the address is not correctly aligned (i.e. is not a valid page start).
    #[inline]
    pub const fn from_start_address(address: VirtAddr) -> Result<Self, AddressNotAligned> {
        if !address.is_aligned_u64(S::SIZE) {
            return Err(AddressNotAligned);
        }
        Ok(Page::containing_address(address))
    }

    /// Returns the page that starts at the given virtual address.
    ///
    /// ## Safety
    ///
    /// The address must be correctly aligned.
    #[inline]
    pub const unsafe fn from_start_address_unchecked(start_address: VirtAddr) -> Self {
        Page {
            start_address,
            size: PhantomData,
        }
    }

    /// Returns the page that contains the given virtual address.
    #[inline]
    pub const fn containing_address(address: VirtAddr) -> Self {
        Page {
            start_address: address.align_down_u64(S::SIZE),
            size: PhantomData,
        }
    }

    /// Returns the start address of the page.
    #[inline]
    pub const fn start_address(self) -> VirtAddr {
        self.start_address
    }

    /// Returns the size the page (4KB, 2MB or 1GB).
    #[inline]
    pub const fn size(self) -> u64 {
        S::SIZE
    }

    /// Returns the level 4 page table index of this page.
    #[inline]
    pub const fn p4_index(self) -> PageTableIndex {
        self.start_address().p4_index()
    }

    /// Returns the level 3 page table index of this page.
    #[inline]
    pub const fn p3_index(self) -> PageTableIndex {
        self.start_address().p3_index()
    }

    /// Returns the table index of this page at the specified level.
    #[inline]
    pub const fn page_table_index(self, level: PageTableLevel) -> PageTableIndex {
        self.start_address().page_table_index(level)
    }

    /// Returns a range of pages, exclusive `end`.
    #[inline]
    pub const fn range(start: Self, end: Self) -> PageRange<S> {
        PageRange { start, end }
    }

    /// Returns a range of pages, inclusive `end`.
    #[inline]
    pub const fn range_inclusive(start: Self, end: Self) -> PageRangeInclusive<S> {
        PageRangeInclusive { start, end }
    }

    // FIXME: Move this into the `Step` impl, once `Step` is stabilized.
    pub(crate) fn steps_between_u64(start: &Self, end: &Self) -> Option<u64> {
        VirtAddr::steps_between_u64(&start.start_address(), &end.start_address())
            .map(|steps| steps / S::SIZE)
    }

    // FIXME: Move this into the `Step` impl, once `Step` is stabilized.
    #[cfg(any(feature = "instructions", feature = "step_trait"))]
    pub(crate) fn steps_between_impl(start: &Self, end: &Self) -> (usize, Option<usize>) {
        if let Some(steps) = Self::steps_between_u64(start, end) {
            let steps = usize::try_from(steps).ok();
            (steps.unwrap_or(usize::MAX), steps)
        } else {
            (0, None)
        }
    }

    // FIXME: Move this into the `Step` impl, once `Step` is stabilized.
    #[cfg(any(feature = "instructions", feature = "step_trait"))]
    pub(crate) fn forward_checked_impl(start: Self, count: usize) -> Option<Self> {
        Self::forward_checked_u64(start, u64::try_from(count).ok()?)
    }

    /// Returns the page `count` pages after `start`, skipping the non-canonical gap.
    ///
    /// Returns `None` if there is no such page.
    #[inline]
    pub(crate) fn forward_checked_u64(start: Self, count: u64) -> Option<Self> {
        let count = count.checked_mul(S::SIZE)?;
        let start_address = VirtAddr::forward_checked_u64(start.start_address, count)?;
        Some(Self {
            start_address,
            size: PhantomData,
        })
    }

    /// Returns the page `count` pages before `start`, skipping the non-canonical gap.
    ///
    /// Returns `None` if there is no such page.
    #[inline]
    pub(crate) fn backward_checked_u64(start: Self, count: u64) -> Option<Self> {
        let count = count.checked_mul(S::SIZE)?;
        let start_address = VirtAddr::backward_checked_u64(start.start_address, count)?;
        Some(Self {
            start_address,
            size: PhantomData,
        })
    }
}

impl<S: NotGiantPageSize> Page<S> {
    /// Returns the level 2 page table index of this page.
    #[inline]
    pub const fn p2_index(self) -> PageTableIndex {
        self.start_address().p2_index()
    }
}

impl Page<Size1GiB> {
    /// Returns the 1GiB memory page with the specified page table indices.
    #[inline]
    pub const fn from_page_table_indices_1gib(
        p4_index: PageTableIndex,
        p3_index: PageTableIndex,
    ) -> Self {
        let mut addr = 0;
        addr |= p4_index.into_u64() << 39;
        addr |= p3_index.into_u64() << 30;
        Page::containing_address(VirtAddr::new_truncate(addr))
    }
}

impl Page<Size2MiB> {
    /// Returns the 2MiB memory page with the specified page table indices.
    #[inline]
    pub const fn from_page_table_indices_2mib(
        p4_index: PageTableIndex,
        p3_index: PageTableIndex,
        p2_index: PageTableIndex,
    ) -> Self {
        let mut addr = 0;
        addr |= p4_index.into_u64() << 39;
        addr |= p3_index.into_u64() << 30;
        addr |= p2_index.into_u64() << 21;
        Page::containing_address(VirtAddr::new_truncate(addr))
    }
}

impl Page<Size4KiB> {
    /// Returns the 4KiB memory page with the specified page table indices.
    #[inline]
    pub const fn from_page_table_indices(
        p4_index: PageTableIndex,
        p3_index: PageTableIndex,
        p2_index: PageTableIndex,
        p1_index: PageTableIndex,
    ) -> Self {
        let mut addr = 0;
        addr |= p4_index.into_u64() << 39;
        addr |= p3_index.into_u64() << 30;
        addr |= p2_index.into_u64() << 21;
        addr |= p1_index.into_u64() << 12;
        Page::containing_address(VirtAddr::new_truncate(addr))
    }

    /// Returns the level 1 page table index of this page.
    #[inline]
    pub const fn p1_index(self) -> PageTableIndex {
        self.start_address.p1_index()
    }
}

impl<S: PageSize> fmt::Debug for Page<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!(
            "Page[{}]({:#x})",
            S::DEBUG_STR,
            self.start_address().as_u64()
        ))
    }
}

impl<S: PageSize> Add<u64> for Page<S> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: u64) -> Self::Output {
        Page::containing_address(self.start_address() + rhs * S::SIZE)
    }
}

impl<S: PageSize> AddAssign<u64> for Page<S> {
    #[inline]
    fn add_assign(&mut self, rhs: u64) {
        *self = *self + rhs;
    }
}

impl<S: PageSize> Sub<u64> for Page<S> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: u64) -> Self::Output {
        Page::containing_address(self.start_address() - rhs * S::SIZE)
    }
}

impl<S: PageSize> SubAssign<u64> for Page<S> {
    #[inline]
    fn sub_assign(&mut self, rhs: u64) {
        *self = *self - rhs;
    }
}

impl<S: PageSize> Sub<Self> for Page<S> {
    type Output = u64;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        (self.start_address - rhs.start_address) / S::SIZE
    }
}

#[cfg(feature = "step_trait")]
impl<S: PageSize> Step for Page<S> {
    fn steps_between(start: &Self, end: &Self) -> (usize, Option<usize>) {
        Self::steps_between_impl(start, end)
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        Self::forward_checked_impl(start, count)
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        Self::backward_checked_u64(start, u64::try_from(count).ok()?)
    }

    fn forward_overflowing(start: Self, count: usize) -> (Self, bool) {
        match Self::forward_checked(start, count) {
            Some(next) => (next, false),
            None => (start, true),
        }
    }

    fn backward_overflowing(start: Self, count: usize) -> (Self, bool) {
        match Self::backward_checked(start, count) {
            Some(next) => (next, false),
            None => (start, true),
        }
    }
}

/// A range of pages with exclusive upper bound.
///
/// The range contains all canonical pages from `start` up to but excluding `end`. Like a
/// [`core::ops::Range`] of pages, a range that spans the non-canonical “gap” of the address
/// space skips the gap: iterating it and [`len`](Self::len) only consider canonical pages.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PageRange<S: PageSize = Size4KiB> {
    /// The start of the range, inclusive.
    pub start: Page<S>,
    /// The end of the range, exclusive.
    pub end: Page<S>,
}

impl<S: PageSize> PageRange<S> {
    /// Returns whether this range contains no pages.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Returns the number of pages in the range.
    ///
    /// Pages in the non-canonical “gap” of the address space are not counted.
    #[inline]
    pub fn len(&self) -> u64 {
        Page::steps_between_u64(&self.start, &self.end).unwrap_or(0)
    }

    /// Returns the size in bytes of all pages within the range.
    #[inline]
    pub fn size(&self) -> u64 {
        S::SIZE * self.len()
    }
}

impl<S: PageSize> Iterator for PageRange<S> {
    type Item = Page<S>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let page = self.start;
            // `end` is a page after `start`, so `start` has a successor.
            self.start = Page::forward_checked_u64(page, 1)
                .expect("a page before the end of a range has a successor");
            Some(page)
        } else {
            None
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        // Convert to `u64`. If the value doesn't fit just use `u64::MAX`, which
        // is larger than any possible length.
        let n = u64::try_from(n).unwrap_or(u64::MAX);

        if n >= self.len() {
            // Skipping all remaining pages exhausts the range.
            self.start = self.end;
            return None;
        }

        self.start =
            Page::forward_checked_u64(self.start, n).expect("`n` is smaller than the length");
        self.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        usize::try_from(len)
            .map(|len| (len, Some(len)))
            .unwrap_or((usize::MAX, None))
    }
}

impl<S: PageSize> DoubleEndedIterator for PageRange<S> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            // `start` is a page before `end`, so `end` has a predecessor.
            self.end = Page::backward_checked_u64(self.end, 1)
                .expect("a page after the start of a range has a predecessor");
            Some(self.end)
        } else {
            None
        }
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        // Convert to `u64`. If the value doesn't fit just use `u64::MAX`, which
        // is larger than any possible length.
        let n = u64::try_from(n).unwrap_or(u64::MAX);

        if n >= self.len() {
            // Skipping all remaining pages exhausts the range.
            self.end = self.start;
            return None;
        }

        self.end = Page::backward_checked_u64(self.end, n).expect("`n` is smaller than the length");
        self.next_back()
    }
}

impl PageRange<Size2MiB> {
    /// Converts the range of 2MiB pages to a range of 4KiB pages.
    #[inline]
    pub fn as_4kib_page_range(self) -> PageRange<Size4KiB> {
        PageRange {
            start: Page::containing_address(self.start.start_address()),
            end: Page::containing_address(self.end.start_address()),
        }
    }
}

impl<S: PageSize> fmt::Debug for PageRange<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("PageRange")
            .field("start", &self.start)
            .field("end", &self.end)
            .finish()
    }
}

/// A range of pages with inclusive upper bound.
///
/// The range contains all canonical pages from `start` up to and including `end`. Like a
/// [`core::ops::RangeInclusive`] of pages, a range that spans the non-canonical “gap” of the
/// address space skips the gap: iterating it and [`len`](Self::len) only consider canonical
/// pages.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PageRangeInclusive<S: PageSize = Size4KiB> {
    /// The start of the range, inclusive.
    pub start: Page<S>,
    /// The end of the range, inclusive.
    pub end: Page<S>,
}

impl<S: PageSize> PageRangeInclusive<S> {
    /// Returns whether this range contains no pages.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start > self.end
    }

    /// Returns the number of pages in the range.
    ///
    /// Pages in the non-canonical “gap” of the address space are not counted.
    #[inline]
    pub fn len(&self) -> u64 {
        Page::steps_between_u64(&self.start, &self.end).map_or(0, |steps| steps + 1)
    }

    /// Makes the range empty, i.e. moves `start` past `end`.
    ///
    /// The range must not be empty already.
    fn exhaust(&mut self) {
        debug_assert!(!self.is_empty());
        if self.start < self.end {
            // Swapping the bounds makes the range empty, even if it covers the whole
            // address space.
            core::mem::swap(&mut self.start, &mut self.end);
        } else if let Some(after_end) = Page::forward_checked_u64(self.end, 1) {
            self.start = after_end;
        } else {
            // The single page of the range is the last page of the address space, so it
            // has a predecessor.
            self.end = Page::backward_checked_u64(self.start, 1)
                .expect("the last page of the address space has a predecessor");
        }
    }

    /// Returns the size in bytes of all pages within the range.
    #[inline]
    pub fn size(&self) -> u64 {
        S::SIZE * self.len()
    }
}

impl<S: PageSize> Iterator for PageRangeInclusive<S> {
    type Item = Page<S>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start <= self.end {
            let page = self.start;
            if page < self.end {
                // `end` is a page after `start`, so `start` has a successor.
                self.start = Page::forward_checked_u64(page, 1)
                    .expect("a page before the end of a range has a successor");
            } else {
                self.exhaust();
            }
            Some(page)
        } else {
            None
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        // Convert to `u64`. If the value doesn't fit just use `u64::MAX`, which
        // is larger than any possible length.
        let n = u64::try_from(n).unwrap_or(u64::MAX);

        if n >= self.len() {
            // Skipping all remaining pages exhausts the range.
            if !self.is_empty() {
                self.exhaust();
            }
            return None;
        }

        self.start =
            Page::forward_checked_u64(self.start, n).expect("`n` is smaller than the length");
        self.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        usize::try_from(len)
            .map(|len| (len, Some(len)))
            .unwrap_or((usize::MAX, None))
    }
}

impl<S: PageSize> DoubleEndedIterator for PageRangeInclusive<S> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start <= self.end {
            let page = self.end;
            if self.start < page {
                // `start` is a page before `end`, so `end` has a predecessor.
                self.end = Page::backward_checked_u64(page, 1)
                    .expect("a page after the start of a range has a predecessor");
            } else {
                self.exhaust();
            }
            Some(page)
        } else {
            None
        }
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        // Convert to `u64`. If the value doesn't fit just use `u64::MAX`, which
        // is larger than any possible length.
        let n = u64::try_from(n).unwrap_or(u64::MAX);

        if n >= self.len() {
            // Skipping all remaining pages exhausts the range.
            if !self.is_empty() {
                self.exhaust();
            }
            return None;
        }

        self.end = Page::backward_checked_u64(self.end, n).expect("`n` is smaller than the length");
        self.next_back()
    }
}

impl<S: PageSize> fmt::Debug for PageRangeInclusive<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("PageRangeInclusive")
            .field("start", &self.start)
            .field("end", &self.end)
            .finish()
    }
}

#[cfg(kani)]
impl<S: PageSize> kani::Arbitrary for Page<S> {
    fn any() -> Self {
        Self::containing_address(kani::any())
    }
}

/// The given address was not sufficiently aligned.
#[derive(Debug)]
pub struct AddressNotAligned;

impl fmt::Display for AddressNotAligned {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the given address was not sufficiently aligned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_is_hash<T: core::hash::Hash>() {}

    #[test]
    pub fn test_page_is_hash() {
        test_is_hash::<Page<Size4KiB>>();
        test_is_hash::<Page<Size2MiB>>();
        test_is_hash::<Page<Size1GiB>>();
    }

    #[test]
    pub fn test_page_ranges() {
        let page_size = Size4KiB::SIZE;
        let number = 1000;

        let start_addr = VirtAddr::new(0xdead_beaf);
        let start: Page = Page::containing_address(start_addr);
        let end = start + number;

        let mut range = Page::range(start, end);
        for i in 0..number {
            assert_eq!(
                range.next(),
                Some(Page::containing_address(start_addr + page_size * i))
            );
        }
        assert_eq!(range.next(), None);

        let mut range_inclusive = Page::range_inclusive(start, end);
        for i in 0..=number {
            assert_eq!(
                range_inclusive.next(),
                Some(Page::containing_address(start_addr + page_size * i))
            );
        }
        assert_eq!(range_inclusive.next(), None);
    }

    #[test]
    pub fn test_page_range_inclusive_overflow() {
        let page_size = Size4KiB::SIZE;
        let number = 1000;

        let start_addr = VirtAddr::new(u64::MAX).align_down(page_size) - number * page_size;
        let start: Page = Page::containing_address(start_addr);
        let end = start + number;

        let mut range_inclusive = Page::range_inclusive(start, end);
        for i in 0..=number {
            assert_eq!(
                range_inclusive.next(),
                Some(Page::containing_address(start_addr + page_size * i))
            );
        }
        assert_eq!(range_inclusive.next(), None);
    }

    /// The last page of the lower half and the first page of the upper half of the
    /// address space.
    fn pages_around_gap() -> (Page<Size4KiB>, Page<Size4KiB>) {
        let before = Page::from_start_address(VirtAddr::new(0x7fff_ffff_f000)).unwrap();
        let after = Page::from_start_address(VirtAddr::new(0xffff_8000_0000_0000)).unwrap();
        (before, after)
    }

    #[test]
    fn test_page_range_skips_gap() {
        let (before, after) = pages_around_gap();

        let range = Page::range(before, after);
        assert_eq!(range.len(), 1);
        assert_eq!(range.clone().collect::<Vec<_>>(), [before]);
        assert_eq!(range.clone().rev().collect::<Vec<_>>(), [before]);

        let range = Page::range(before - 1, after + 1);
        assert_eq!(range.len(), 3);
        assert_eq!(
            range.clone().collect::<Vec<_>>(),
            [before - 1, before, after]
        );
        assert_eq!(
            range.clone().rev().collect::<Vec<_>>(),
            [after, before, before - 1]
        );
        assert_eq!(range.clone().nth(2), Some(after));
        assert_eq!(range.clone().nth(3), None);
        assert_eq!(range.clone().nth_back(2), Some(before - 1));
        assert_eq!(range.clone().nth_back(3), None);

        // `PageRange` behaves like `Range<Page>`.
        #[cfg(feature = "step_trait")]
        assert_eq!(
            range.collect::<Vec<_>>(),
            (before - 1..after + 1).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_page_range_inclusive_skips_gap() {
        let (before, after) = pages_around_gap();

        let range = Page::range_inclusive(before, after);
        assert_eq!(range.len(), 2);
        assert_eq!(range.clone().collect::<Vec<_>>(), [before, after]);
        assert_eq!(range.clone().rev().collect::<Vec<_>>(), [after, before]);
        assert_eq!(range.clone().nth(1), Some(after));
        assert_eq!(range.clone().nth(2), None);
        assert_eq!(range.clone().nth_back(1), Some(before));
        assert_eq!(range.clone().nth_back(2), None);

        // A single page next to the gap.
        let range = Page::range_inclusive(before, before);
        assert_eq!(range.len(), 1);
        assert_eq!(range.clone().collect::<Vec<_>>(), [before]);
        assert_eq!(range.clone().rev().collect::<Vec<_>>(), [before]);
        let range = Page::range_inclusive(after, after);
        assert_eq!(range.len(), 1);
        assert_eq!(range.clone().collect::<Vec<_>>(), [after]);
        assert_eq!(range.clone().rev().collect::<Vec<_>>(), [after]);

        // `PageRangeInclusive` behaves like `RangeInclusive<Page>`.
        #[cfg(feature = "step_trait")]
        assert_eq!(
            Page::range_inclusive(before - 1, after + 1).collect::<Vec<_>>(),
            (before - 1..=after + 1).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_page_range_inclusive_address_space_bounds() {
        let first = Page::<Size4KiB>::containing_address(VirtAddr::new(0));
        let last = Page::<Size4KiB>::containing_address(VirtAddr::new(u64::MAX));

        let mut range = Page::range_inclusive(first, first);
        assert_eq!(range.next_back(), Some(first));
        assert_eq!(range.next_back(), None);
        assert!(range.is_empty());

        let mut range = Page::range_inclusive(last, last);
        assert_eq!(range.next(), Some(last));
        assert_eq!(range.next(), None);
        assert!(range.is_empty());

        let mut range = Page::range_inclusive(last - 1, last);
        assert_eq!(range.nth(5), None);
        assert!(range.is_empty());

        let mut range = Page::range_inclusive(first, first + 1);
        assert_eq!(range.nth_back(5), None);
        assert!(range.is_empty());

        // The full address space is a valid range, even though it can't be exhausted by
        // moving only one of its bounds.
        let mut range = Page::range_inclusive(first, last);
        assert_eq!(range.len(), 1 << 36);
        assert_eq!(range.next(), Some(first));
        assert_eq!(range.next_back(), Some(last));
        // On 32-bit targets `usize` can't express a skip count that exhausts the range.
        #[cfg(target_pointer_width = "64")]
        {
            let mut range = Page::range_inclusive(first, last);
            assert_eq!(range.nth(usize::MAX), None);
            assert!(range.is_empty());
            assert_eq!(range.len(), 0);
            assert_eq!(range.next(), None);
            assert_eq!(range.next_back(), None);

            let mut range = Page::range_inclusive(first, last);
            assert_eq!(range.nth_back(usize::MAX), None);
            assert!(range.is_empty());

            let mut range = Page::range_inclusive(first, last);
            assert_eq!(range.nth((1 << 36) - 1), Some(last));
            assert_eq!(range.next(), None);
            assert!(range.is_empty());
        }
    }

    #[test]
    pub fn test_page_range_len() {
        let start_addr = VirtAddr::new(0xdead_beaf);
        let start = Page::<Size4KiB>::containing_address(start_addr);
        let end = start + 50;

        let range = PageRange { start, end };
        assert_eq!(range.len(), 50);

        let range_inclusive = PageRangeInclusive { start, end };
        assert_eq!(range_inclusive.len(), 51);
    }

    #[test]
    #[cfg(feature = "step_trait")]
    fn page_step_forward() {
        let test_cases = [
            (0, 0, Some(0)),
            (0, 1, Some(0x1000)),
            (0x1000, 1, Some(0x2000)),
            (0x7fff_ffff_f000, 1, Some(0xffff_8000_0000_0000)),
            (0xffff_8000_0000_0000, 1, Some(0xffff_8000_0000_1000)),
            (0xffff_ffff_ffff_f000, 1, None),
            #[cfg(target_pointer_width = "64")]
            (0x7fff_ffff_f000, 0x1_2345_6789, Some(0xffff_9234_5678_8000)),
            #[cfg(target_pointer_width = "64")]
            (0x7fff_ffff_f000, 0x8_0000_0000, Some(0xffff_ffff_ffff_f000)),
            #[cfg(target_pointer_width = "64")]
            (0x7fff_fff0_0000, 0x8_0000_00ff, Some(0xffff_ffff_ffff_f000)),
            #[cfg(target_pointer_width = "64")]
            (0x7fff_fff0_0000, 0x8_0000_0100, None),
            #[cfg(target_pointer_width = "64")]
            (0x7fff_ffff_f000, 0x8_0000_0001, None),
            // Make sure that we handle `steps * PAGE_SIZE > u32::MAX`
            // correctly on 32-bit targets.
            (0, 0x10_0000, Some(0x1_0000_0000)),
        ];
        for (start, count, result) in test_cases {
            let start = Page::<Size4KiB>::from_start_address(VirtAddr::new(start)).unwrap();
            let result = result
                .map(|result| Page::<Size4KiB>::from_start_address(VirtAddr::new(result)).unwrap());
            assert_eq!(Step::forward_checked(start, count), result);
        }
    }

    #[test]
    #[cfg(feature = "step_trait")]
    fn page_step_backwards() {
        let test_cases = [
            (0, 0, Some(0)),
            (0, 1, None),
            (0x1000, 1, Some(0)),
            (0xffff_8000_0000_0000, 1, Some(0x7fff_ffff_f000)),
            (0xffff_8000_0000_1000, 1, Some(0xffff_8000_0000_0000)),
            #[cfg(target_pointer_width = "64")]
            (0xffff_9234_5678_8000, 0x1_2345_6789, Some(0x7fff_ffff_f000)),
            #[cfg(target_pointer_width = "64")]
            (0xffff_8000_0000_0000, 0x8_0000_0000, Some(0)),
            #[cfg(target_pointer_width = "64")]
            (0xffff_8000_0000_0000, 0x7_ffff_ff01, Some(0xff000)),
            #[cfg(target_pointer_width = "64")]
            (0xffff_8000_0000_0000, 0x8_0000_0001, None),
            // Make sure that we handle `steps * PAGE_SIZE > u32::MAX`
            // correctly on 32-bit targets.
            (0x1_0000_0000, 0x10_0000, Some(0)),
        ];
        for (start, count, result) in test_cases {
            let start = Page::<Size4KiB>::from_start_address(VirtAddr::new(start)).unwrap();
            let result = result
                .map(|result| Page::<Size4KiB>::from_start_address(VirtAddr::new(result)).unwrap());
            assert_eq!(Step::backward_checked(start, count), result);
        }
    }

    #[test]
    #[cfg(feature = "step_trait")]
    fn page_steps_between() {
        let test_cases = [
            (0, 0, 0, Some(0)),
            (0, 0x1000, 1, Some(1)),
            (0x1000, 0, 0, None),
            (0x1000, 0x1000, 0, Some(0)),
            (0x7fff_ffff_f000, 0xffff_8000_0000_0000, 1, Some(1)),
            (0xffff_8000_0000_0000, 0x7fff_ffff_f000, 0, None),
            (0xffff_8000_0000_0000, 0xffff_8000_0000_0000, 0, Some(0)),
            (0xffff_8000_0000_0000, 0xffff_8000_0000_1000, 1, Some(1)),
            (0xffff_8000_0000_1000, 0xffff_8000_0000_0000, 0, None),
            (0xffff_8000_0000_1000, 0xffff_8000_0000_1000, 0, Some(0)),
            // Make sure that we handle `steps * PAGE_SIZE > u32::MAX` correctly on 32-bit
            // targets.
            (
                0x0000_0000_0000,
                0x0001_0000_0000,
                0x10_0000,
                Some(0x10_0000),
            ),
            // The returned bounds are different when `steps` doesn't fit in
            // into `usize`. On 64-bit targets, `0x1_0000_0000` fits into
            // `usize`, so we can return exact lower and upper bounds. On
            // 32-bit targets, `0x1_0000_0000` doesn't fit into `usize`, so we
            // only return an lower bound of `usize::MAX` and don't return an
            // upper bound.
            #[cfg(target_pointer_width = "64")]
            (
                0x0000_0000_0000,
                0x1000_0000_0000,
                0x1_0000_0000,
                Some(0x1_0000_0000),
            ),
            #[cfg(not(target_pointer_width = "64"))]
            (0x0000_0000_0000, 0x1000_0000_0000, usize::MAX, None),
        ];
        for (start, end, lower, upper) in test_cases {
            let start = Page::<Size4KiB>::from_start_address(VirtAddr::new(start)).unwrap();
            let end = Page::from_start_address(VirtAddr::new(end)).unwrap();
            assert_eq!(Step::steps_between(&start, &end), (lower, upper));
        }
    }

    #[test]
    #[cfg(feature = "step_trait")]
    fn page_step_overflowing() {
        let page = |addr| Page::<Size4KiB>::from_start_address(VirtAddr::new(addr)).unwrap();

        assert_eq!(
            Step::forward_overflowing(page(0x7fff_ffff_f000), 1),
            (page(0xffff_8000_0000_0000), false)
        );
        assert_eq!(
            Step::backward_overflowing(page(0xffff_8000_0000_0000), 1),
            (page(0x7fff_ffff_f000), false)
        );

        assert!(Step::forward_overflowing(page(0xffff_ffff_ffff_f000), 1).1);
        assert!(Step::backward_overflowing(page(0), 1).1);
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    // The range iterators must behave exactly like `core::ops::Range` and
    // `core::ops::RangeInclusive` over pages, which use the `Step` impl.

    #[kani::proof]
    fn page_range_next() {
        let start = kani::any::<Page<Size4KiB>>();
        let end = kani::any::<Page<Size4KiB>>();

        let mut our_range = Page::range(start, end);
        let mut native_range = start..end;
        // The first assert checks that we're returning the correct value.
        assert_eq!(our_range.next(), native_range.next());
        // The second assert checks that we're updating the range state correctly.
        assert_eq!(our_range.next(), native_range.next());
    }

    #[kani::proof]
    fn page_range_next_back() {
        let start = kani::any::<Page<Size4KiB>>();
        let end = kani::any::<Page<Size4KiB>>();

        let mut our_range = Page::range(start, end);
        let mut native_range = start..end;
        assert_eq!(our_range.next_back(), native_range.next_back());
        assert_eq!(our_range.next_back(), native_range.next_back());
    }

    #[kani::proof]
    fn page_range_inclusive_next() {
        let start = kani::any::<Page<Size4KiB>>();
        let end = kani::any::<Page<Size4KiB>>();

        let mut our_range = Page::range_inclusive(start, end);
        let mut native_range = start..=end;
        assert_eq!(our_range.next(), native_range.next());
        assert_eq!(our_range.next(), native_range.next());
    }

    #[kani::proof]
    fn page_range_inclusive_next_back() {
        let start = kani::any::<Page<Size4KiB>>();
        let end = kani::any::<Page<Size4KiB>>();

        let mut our_range = Page::range_inclusive(start, end);
        let mut native_range = start..=end;
        assert_eq!(our_range.next_back(), native_range.next_back());
        assert_eq!(our_range.next_back(), native_range.next_back());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn page_range_nth() {
        let start = kani::any::<Page>();
        let end = kani::any::<Page>();
        let m = kani::any::<usize>();
        let n = kani::any::<usize>();

        let mut our_range = Page::range(start, end);
        let mut native_range = start..end;
        assert_eq!(our_range.nth(m), native_range.nth(m));
        assert_eq!(our_range.nth(n), native_range.nth(n));
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn page_range_nth_back() {
        let start = kani::any::<Page>();
        let end = kani::any::<Page>();
        let m = kani::any::<usize>();
        let n = kani::any::<usize>();

        let mut our_range = Page::range(start, end);
        let mut native_range = start..end;
        assert_eq!(our_range.nth_back(m), native_range.nth_back(m));
        assert_eq!(our_range.nth_back(n), native_range.nth_back(n));
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn page_range_inclusive_nth() {
        let start = kani::any::<Page>();
        let end = kani::any::<Page>();
        let m = kani::any::<usize>();
        let n = kani::any::<usize>();

        let mut our_range = Page::range_inclusive(start, end);
        let mut native_range = start..=end;
        assert_eq!(our_range.nth(m), native_range.nth(m));
        assert_eq!(our_range.nth(n), native_range.nth(n));
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn page_range_inclusive_nth_back() {
        let start = kani::any::<Page>();
        let end = kani::any::<Page>();
        let m = kani::any::<usize>();
        let n = kani::any::<usize>();

        let mut our_range = Page::range_inclusive(start, end);
        let mut native_range = start..=end;
        assert_eq!(our_range.nth_back(m), native_range.nth_back(m));
        assert_eq!(our_range.nth_back(n), native_range.nth_back(n));
    }
}
