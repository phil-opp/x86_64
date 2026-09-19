//! Abstractions for page tables and other paging related structures.
//!
//! Page tables translate virtual memory “pages” to physical memory “frames”.
//!
//! ## Paging modes
//!
//! The [`Page`] type, the page range types, and the [`Mapper`], [`Translate`], and
//! [`mapper::CleanUp`] traits are generic over the [paging mode](crate::addr::PagingMode)
//! they work with. The paging mode defaults to
//! [`FourLevelPaging`](crate::addr::FourLevelPaging), so `Page<Size4KiB>` is a page in a
//! 48-bit address space. Use [`FiveLevelPaging`](crate::addr::FiveLevelPaging) for pages
//! in a 57-bit address space, e.g. `Page<Size4KiB, FiveLevelPaging>`.
//!
//! The mapper implementations in this module ([`MappedPageTable`], `OffsetPageTable`,
//! and `RecursivePageTable`) currently only support 4-level paging, i.e. they only
//! implement the traits for `FourLevelPaging`.

pub use self::frame::PhysFrame;
pub use self::frame_alloc::{FrameAllocator, FrameDeallocator};
#[doc(no_inline)]
pub use self::mapper::MappedPageTable;
#[cfg(all(feature = "instructions", target_arch = "x86_64"))]
#[doc(no_inline)]
pub use self::mapper::RecursivePageTable;
pub use self::mapper::{Mapper, Translate};
#[cfg(target_pointer_width = "64")]
#[doc(no_inline)]
pub use self::mapper::{OffsetPageTable, PhysOffset};
pub use self::page::{Page, PageSize, Size1GiB, Size2MiB, Size4KiB};
pub use self::page_table::{PageOffset, PageTable, PageTableFlags, PageTableIndex};

pub mod frame;
mod frame_alloc;
pub mod mapper;
pub mod page;
pub mod page_table;
