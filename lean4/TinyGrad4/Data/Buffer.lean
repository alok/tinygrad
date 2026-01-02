import TinyGrad4.DType
import TinyGrad4.Shape
import TinyGrad4.Data.Device
import Std.Data.HashMap

/-!
# Buffer - Device-Agnostic Buffer Protocol

Zero-copy buffer exchange inspired by DLPack, with typed views + tracked allocations.

## Design Principles
1. **Allocation vs view**: allocations own memory, views are cheap slices
2. **Device agnostic**: same types work across CPU/GPU/remote
3. **Zero-copy**: `RawBuffer` is a view descriptor, not a copy
4. **Ownership tracked**: tracked allocations via `BufferRegistry`
5. **API-enforced linearity**: `withBorrowed` pattern for safe access

## Usage
```lean
-- Create tracked buffer
let buf ← TrackedBuffer.alloc reg [256] .float32 .cpu

-- Safe scoped access (auto acquire/release)
withBorrowed buf fun view => do
  processData view.allocation.handle.ptr
```
-/

namespace TinyGrad4.Data

open Std (HashMap)

/-! ## Buffer Handle -/

/-- Opaque buffer handle - raw pointer or handle ID.
    The interpretation depends on the device. -/
structure BufferHandle where
  /-- Platform-sized pointer or handle -/
  ptr : USize
  /-- Device where the buffer resides -/
  device : Device
  deriving Repr, BEq

namespace BufferHandle

/-- Null handle (invalid) -/
def null : BufferHandle := { ptr := 0, device := .cpu }

/-- Check if handle is null -/
def isNull (h : BufferHandle) : Bool := h.ptr == 0

end BufferHandle

/-! ## Raw Buffer (FFI Descriptor) -/

/-- Raw buffer descriptor for zero-copy exchange.
    Analogous to DLTensor from DLPack. -/
structure RawBuffer where
  /-- Raw handle (pointer + device) -/
  handle : BufferHandle
  /-- Data type of elements -/
  dtype : DType
  /-- Shape of the tensor -/
  shape : Array Nat
  /-- Strides in elements (None = C-contiguous) -/
  strides : Option (Array Nat) := none
  /-- Byte offset into the buffer -/
  byteOffset : Nat := 0
  deriving Repr

namespace RawBuffer

/-- Number of dimensions -/
def ndim (d : RawBuffer) : Nat := d.shape.size

/-- Total number of elements -/
def numel (d : RawBuffer) : Nat := d.shape.foldl (· * ·) 1

/-- Byte size of the view -/
def bytes (d : RawBuffer) : Nat := d.numel * d.dtype.itemsize

private def computeCStridesList (shape : List Nat) : List Nat :=
  let (_, strides) :=
    shape.foldr (fun dim (acc, out) => (acc * dim, acc :: out)) (1, [])
  strides

/-- Compute C-contiguous strides for a shape.
    stride[i] = product of shape[i+1..n-1]
    For shape [2,3,4]: strides = [12, 4, 1] -/
def computeCStrides (shape : Array Nat) : Array Nat :=
  (computeCStridesList shape.toList).toArray

/-- Check if contiguous (strides match C-order) -/
def isContiguous (d : RawBuffer) : Bool :=
  match d.strides with
  | none => true
  | some strides => strides == computeCStrides d.shape

/-- Device where buffer resides -/
def device (d : RawBuffer) : Device := d.handle.device

/-- Create a view into this buffer at offset (in elements). -/
def slice (d : RawBuffer) (offset : Nat) (newShape : Array Nat) : RawBuffer :=
  { d with
    shape := newShape
    byteOffset := d.byteOffset + offset * d.dtype.itemsize
  }

end RawBuffer

/-! ## Buffer ID and Registry -/

/-- Unique identifier for a buffer allocation -/
structure BufferId where
  id : Nat
  deriving BEq, Hashable, DecidableEq, Repr

/-- Buffer registry entry -/
structure BufferEntry where
  device : Device
  refCount : Nat
  byteSize : Nat
  deriving Repr

/-- Global buffer registry for tracking allocations and reference counts.
    Provides runtime safety net for debugging and leak detection. -/
structure BufferRegistry where
  /-- Active buffers: id -> (device, refcount, size) -/
  active : IO.Ref (HashMap BufferId BufferEntry)
  /-- Next ID to allocate -/
  nextId : IO.Ref Nat

namespace BufferRegistry

/-- Create a new empty registry -/
def new : IO BufferRegistry := do
  let active ← IO.mkRef {}
  let nextId ← IO.mkRef 0
  pure { active, nextId }

/-- Register a new buffer, returning its ID -/
def register (reg : BufferRegistry) (device : Device) (byteSize : Nat) : IO BufferId := do
  let id ← reg.nextId.modifyGet fun n => ({ id := n }, n + 1)
  reg.active.modify fun m =>
    m.insert id { device, refCount := 1, byteSize }
  pure id

/-- Increment reference count -/
def acquire (reg : BufferRegistry) (id : BufferId) : IO Unit := do
  let m ← reg.active.get
  match m.get? id with
  | some entry =>
    let newEntry : BufferEntry := { entry with refCount := entry.refCount + 1 }
    reg.active.set (m.insert id newEntry)
  | none =>
    IO.eprintln s!"BufferRegistry.acquire: unknown buffer {repr id}"

/-- Decrement reference count, returns true if buffer should be freed -/
def release (reg : BufferRegistry) (id : BufferId) : IO Bool := do
  let m ← reg.active.get
  match m.get? id with
  | some entry =>
    if entry.refCount <= 1 then
      reg.active.set (m.erase id)
      pure true
    else
      let newEntry : BufferEntry := { entry with refCount := entry.refCount - 1 }
      reg.active.set (m.insert id newEntry)
      pure false
  | none =>
    IO.eprintln s!"BufferRegistry.release: unknown buffer {repr id}"
    pure false

/-- Get current reference count (for debugging) -/
def getRefCount (reg : BufferRegistry) (id : BufferId) : IO Nat := do
  let m ← reg.active.get
  match m.get? id with
  | some entry => pure entry.refCount
  | none => pure 0

/-- Get number of active buffers (for leak detection) -/
def activeCount (reg : BufferRegistry) : IO Nat := do
  let m ← reg.active.get
  pure m.size

/-- Check for leaks (non-zero active count) -/
def checkLeaks (reg : BufferRegistry) : IO (Array BufferId) := do
  let m ← reg.active.get
  pure (m.toArray.map Prod.fst)

end BufferRegistry

/-! ## Allocation -/

/-- Raw memory allocation - device-specific, reference-counted. -/
structure Allocation where
  handle : BufferHandle
  byteSize : Nat
  deriving Repr, BEq

/-- Tracked allocation with registry for ref-counting. -/
structure TrackedAllocation where
  id : BufferId
  allocation : Allocation
  registry : BufferRegistry

namespace TrackedAllocation

/-- Increment reference count -/
def acquire (a : TrackedAllocation) : IO Unit :=
  a.registry.acquire a.id

/-- Decrement reference count, returns true if allocation should be freed. -/
def release (a : TrackedAllocation) : IO Bool :=
  a.registry.release a.id

end TrackedAllocation

/-! ## Typed Buffer Views -/

/-- Typed view over an allocation - shape/dtype at compile time. -/
structure Buffer (shape : Shape) (dtype : DType) where
  allocation : Allocation
  /-- Strides in elements (None = C-contiguous) -/
  strides : Option (List Nat) := none
  /-- Byte offset into allocation -/
  byteOffset : Nat := 0

namespace Buffer

/-- Total number of elements. -/
def numel (_buf : Buffer shape dtype) : Nat :=
  Shape.numel shape

/-- Byte size of the view. -/
def bytes (_buf : Buffer shape dtype) : Nat :=
  Shape.numel shape * dtype.itemsize

private def computeCStridesList (shape : Shape) : List Nat :=
  let (_, strides) :=
    shape.foldr (fun dim (acc, out) => (acc * dim, acc :: out)) (1, [])
  strides

/-- Compute C-contiguous strides for a shape. -/
def computeCStrides (shape : Shape) : List Nat :=
  computeCStridesList shape

/-- Check if contiguous (strides match C-order). -/
def isContiguous (buf : Buffer shape dtype) : Bool :=
  match buf.strides with
  | none => true
  | some strides => strides == computeCStrides shape

/-- Device where buffer resides. -/
def device (buf : Buffer shape dtype) : Device :=
  buf.allocation.handle.device

/-- Create a view into an allocation (zero-copy). -/
def view (alloc : Allocation) (offset : Nat := 0) : Buffer shape dtype :=
  { allocation := alloc, strides := none, byteOffset := offset }

/-- Slice view at element offset, returning new shape. -/
def slice (buf : Buffer shape dtype) (offset : Nat) (newShape : Shape) : Buffer newShape dtype :=
  { allocation := buf.allocation
    strides := none
    byteOffset := buf.byteOffset + offset * dtype.itemsize }

/-- Convert typed view to raw descriptor (for FFI). -/
def toRaw (buf : Buffer shape dtype) : RawBuffer := {
  handle := buf.allocation.handle
  dtype := dtype
  shape := shape.toArray
  strides := buf.strides.map (·.toArray)
  byteOffset := buf.byteOffset
}

/-- Convert raw descriptor to typed view when shape/dtype match. -/
def ofRaw? (shape : Shape) (dtype : DType) (raw : RawBuffer) : Option (Buffer shape dtype) :=
  if raw.shape.toList == shape && raw.dtype == dtype then
    some {
      allocation := { handle := raw.handle, byteSize := raw.bytes }
      strides := raw.strides.map (·.toList)
      byteOffset := raw.byteOffset
    }
  else
    none

end Buffer

/-! ## Tracked Buffer Views -/

/-- Tracked view referencing a tracked allocation. -/
structure TrackedBuffer (shape : Shape) (dtype : DType) where
  allocation : TrackedAllocation
  /-- Strides in elements (None = C-contiguous) -/
  strides : Option (List Nat) := none
  /-- Byte offset into allocation -/
  byteOffset : Nat := 0

namespace TrackedBuffer

/-- Allocate a new tracked buffer (placeholder allocation via FFI). -/
def alloc (registry : BufferRegistry) (shape : Shape) (dtype : DType)
    (device : Device := .cpu) : IO (TrackedBuffer shape dtype) := do
  let numBytes := Shape.numel shape * dtype.itemsize
  let id ← registry.register device numBytes
  let handle : BufferHandle := { ptr := 0, device }
  let allocation : Allocation := { handle, byteSize := numBytes }
  let tracked : TrackedAllocation := { id, allocation, registry }
  pure { allocation := tracked, strides := none, byteOffset := 0 }

/-- Increment reference count. -/
def acquire (buf : TrackedBuffer shape dtype) : IO Unit :=
  buf.allocation.acquire

/-- Decrement reference count, free if last reference. -/
def release (buf : TrackedBuffer shape dtype) : IO Unit := do
  let shouldFree ← buf.allocation.release
  if shouldFree then
    -- In real implementation, call FFI to free on device
    pure ()

/-- Device where the allocation resides. -/
def device (buf : TrackedBuffer shape dtype) : Device :=
  buf.allocation.allocation.handle.device

/-- Convert tracked view to untracked view. -/
def toBuffer (buf : TrackedBuffer shape dtype) : Buffer shape dtype :=
  { allocation := buf.allocation.allocation
    strides := buf.strides
    byteOffset := buf.byteOffset }

/-- Convert tracked view to raw descriptor. -/
def toRaw (buf : TrackedBuffer shape dtype) : RawBuffer :=
  buf.toBuffer.toRaw

end TrackedBuffer

/-! ## Scoped Borrowing -/

/-- Scoped borrowing pattern - automatically acquires and releases. -/
def withBorrowed (buf : TrackedBuffer shape dtype) (action : Buffer shape dtype → IO α) : IO α := do
  buf.acquire
  try
    action buf.toBuffer
  finally
    buf.release

/-- Borrow multiple buffers for a single operation. -/
def withBorrowedAll (bufs : Array (TrackedBuffer shape dtype))
    (action : Array (Buffer shape dtype) → IO α) : IO α := do
  -- Acquire all
  for buf in bufs do
    buf.acquire
  try
    let views := bufs.map (·.toBuffer)
    action views
  finally
    -- Release in reverse order without allocating reversed array
    let n := bufs.size
    for i in [:n] do
      let idx := n - 1 - i
      match bufs[idx]? with
      | some buf => buf.release
      | none => pure ()

/-! ## Buffer Exchange Typeclasses -/

/-- Export capability: types that can produce a raw buffer descriptor. -/
class BufferExport (B : Type) where
  /-- Export as raw descriptor (zero-copy view). -/
  toRaw : B → IO RawBuffer
  /-- Get the device where B resides. -/
  device : B → Device

/-- Import capability: types that can be created from a raw descriptor.
    For tracked types, this requires a registry context. -/
class BufferImport (B : Type) (Ctx : Type) where
  /-- Import from descriptor with context. -/
  fromRaw : Ctx → RawBuffer → (borrow : Bool := true) → IO B

/-- Full exchange capability - both export and context-free import.
    Only for simple types like RawBuffer itself. -/
class BufferExchange (B : Type) extends BufferExport B where
  /-- Import from descriptor (only valid for types without lifecycle tracking). -/
  fromRaw : RawBuffer → (borrow : Bool := true) → IO B

/-- BufferExport for RawBuffer (identity). -/
instance : BufferExport RawBuffer where
  toRaw d := pure d
  device d := d.device

/-- BufferExchange for RawBuffer (identity, no lifecycle). -/
instance : BufferExchange RawBuffer where
  toRaw d := pure d
  device d := d.device
  fromRaw d _ := pure d

/-- BufferExport for Buffer (always safe). -/
instance {shape : Shape} {dtype : DType} : BufferExport (Buffer shape dtype) where
  toRaw buf := pure buf.toRaw
  device buf := buf.device

/-- BufferExport for TrackedBuffer (always safe). -/
instance {shape : Shape} {dtype : DType} : BufferExport (TrackedBuffer shape dtype) where
  toRaw buf := pure buf.toRaw
  device buf := buf.device

/-- BufferImport for TrackedBuffer requires BufferRegistry context. -/
instance {shape : Shape} {dtype : DType} : BufferImport (TrackedBuffer shape dtype) BufferRegistry where
  fromRaw registry raw _ := do
    let buf? := Buffer.ofRaw? shape dtype raw
    match buf? with
    | none =>
      throw (IO.userError "TrackedBuffer.fromRaw: shape/dtype mismatch")
    | some buf =>
      let id ← registry.register buf.device buf.allocation.byteSize
      let tracked : TrackedAllocation := { id, allocation := buf.allocation, registry }
      pure { allocation := tracked, strides := buf.strides, byteOffset := buf.byteOffset }

/-- Create TrackedBuffer from raw descriptor with explicit registry. -/
def TrackedBuffer.fromRaw (registry : BufferRegistry) (shape : Shape) (dtype : DType)
    (raw : RawBuffer) : IO (TrackedBuffer shape dtype) :=
  BufferImport.fromRaw registry raw

/-! ## Type Aliases -/

abbrev VectorBuffer (n : Nat) (dtype : DType) := Buffer [n] dtype
abbrev MatrixBuffer (m n : Nat) (dtype : DType) := Buffer [m, n] dtype
abbrev ImageBuffer (b c h w : Nat) := Buffer [b, c, h, w] .float32

end TinyGrad4.Data
