/-!
# Operations for TinyGrad4

All ~60 operations from tinygrad's UOp system.
Operations are categorized into groups for type checking and optimization.

## Categories
- **Unary**: exp2, log2, sin, sqrt, neg, etc.
- **Binary**: add, mul, sub, div, max, etc.
- **Ternary**: where, mulacc
- **Movement**: reshape, expand, permute, pad, shrink, flip
- **Reduce**: sum, max, etc.
- **Control**: range, if, barrier, sink
- **Memory**: buffer, load, store
-/

namespace TinyGrad4

/-- Scheduling axis types (mirrors tinygrad's `AxisType`). -/
inductive AxisType where
  | GLOBAL | WARP | LOCAL | LOOP | GROUP_REDUCE | REDUCE | UPCAST | UNROLL
  | THREAD | OUTER | PLACEHOLDER
  deriving DecidableEq, Repr, Hashable, Ord

/-- All operations in tinygrad's UOp system -/
inductive Ops where
  -- Defines (variable/buffer definitions)
  | DEFINE_GLOBAL   -- Global variable definition
  | DEFINE_VAR      -- Symbolic variable
  | BIND            -- Bind variable to value
  | SPECIAL         -- Special operations
  | DEFINE_LOCAL    -- Local variable
  | DEFINE_REG      -- Register variable

  -- Non-op (control flow markers)
  | NOOP            -- No operation
  | REWRITE_ERROR   -- Rewrite failure marker
  | PROGRAM         -- Program node (codegen pipeline root)
  | LINEAR          -- Linearized program (list of UOps)
  | SOURCE          -- Rendered source (string arg)
  | SINK            -- Graph sink (output)
  | AFTER           -- Dependency marker
  | GROUP           -- Group operations
  | GEP             -- Get element pointer
  | VECTORIZE       -- Create vector

  -- Load/Store
  | INDEX           -- Compute memory index
  | LOAD            -- Load from memory
  | STORE           -- Store to memory

  -- Math - Unary
  | CAST            -- Type cast
  | BITCAST         -- Reinterpret bits as different type
  | EXP2            -- 2^x
  | LOG2            -- log_2(x)
  | SIN             -- sin(x)
  | COS             -- cos(x)
  | TAN             -- tan(x)
  | SQRT            -- sqrt(x)
  | RECIPROCAL      -- 1/x
  | NEG             -- -x
  | TRUNC           -- Truncate to integer

  -- Math - Binary
  | ADD             -- x + y
  | MUL             -- x * y
  | SUB             -- x - y
  | FDIV            -- x / y (float division)
  | IDIV            -- x / y (integer division)
  | MAX             -- max(x, y)
  | MOD             -- x % y
  | POW             -- x^y
  | SHL             -- x << y (shift left)
  | SHR             -- x >> y (shift right)
  | CMPLT           -- x < y
  | CMPNE           -- x != y
  | CMPEQ           -- x == y
  | XOR             -- x ^ y (bitwise)
  | OR              -- x | y (bitwise)
  | AND             -- x & y (bitwise)
  | THREEFRY        -- Random number generation

  -- Math - Ternary
  | WHERE           -- cond ? x : y
  | MULACC          -- a * b + c (fused multiply-add)

  -- Control flow
  | BARRIER         -- Memory barrier
  | RANGE           -- Loop range
  | IF              -- Conditional
  | END             -- End of block
  | ENDIF           -- End of if

  -- Constants
  | VCONST          -- Vector constant
  | CONST           -- Scalar constant

  -- Custom operations
  | CUSTOM          -- Custom operation
  | CUSTOMI         -- Custom operation (indexed)

  -- Tensor graph operations
  | UNIQUE          -- Unique identifier
  | LUNIQUE         -- Local unique
  | DEVICE          -- Device specification
  | KERNEL          -- Kernel boundary
  | CUSTOM_KERNEL   -- Custom kernel boundary
  | ASSIGN          -- Assignment

  -- Contiguity
  | CONTIGUOUS           -- Make contiguous
  | CONTIGUOUS_BACKWARD  -- Backward for contiguous
  | DETACH               -- Detach from gradient graph

  -- Buffer operations
  | BUFFERIZE       -- Convert to buffer
  | COPY            -- Copy buffer
  | BUFFER          -- Buffer allocation
  | BUFFER_VIEW     -- Buffer view
  | MSELECT         -- Multi-device select
  | MSTACK          -- Multi-device stack
  | ENCDEC          -- Encode/decode

  -- Movement operations (the core 6!)
  | RESHAPE         -- Change shape, preserve elements
  | PERMUTE         -- Reorder dimensions
  | EXPAND          -- Broadcast dimensions
  | PAD             -- Add padding
  | SHRINK          -- Slice/crop
  | FLIP            -- Reverse dimensions
  | MULTI           -- Multi-device operation

  -- Reduce operations
  | REDUCE_AXIS     -- Reduce along axis
  | REDUCE          -- Generic reduce
  | ALLREDUCE       -- All-reduce (distributed)
  | WMMA            -- Tensor core operation

  -- Expander operations
  | UNROLL          -- Loop unrolling
  | CONTRACT        -- Tensor contraction
  | CAT             -- Concatenate
  | PTRCAT          -- Pointer concatenate
  deriving DecidableEq, Repr, Hashable, Ord

namespace Ops

/-- Operation arity -/
inductive Arity where
  | nullary   -- No inputs (constants, buffers)
  | unary     -- One input
  | binary    -- Two inputs
  | ternary   -- Three inputs
  | variadic  -- Variable number of inputs
  deriving DecidableEq, Repr

/-- Get arity of operation -/
def arity : Ops → Arity
  | .CONST | .VCONST | .BUFFER | .UNIQUE | .LUNIQUE | .DEVICE | .SOURCE
  | .DEFINE_GLOBAL | .DEFINE_LOCAL | .DEFINE_REG | .DEFINE_VAR => .nullary
  | .EXP2 | .LOG2 | .SIN | .COS | .TAN | .SQRT | .RECIPROCAL | .NEG | .TRUNC
  | .CAST | .BITCAST | .DETACH | .CONTIGUOUS | .CONTIGUOUS_BACKWARD
  | .RESHAPE | .PERMUTE | .EXPAND | .PAD | .SHRINK | .FLIP => .unary
  | .ADD | .MUL | .SUB | .FDIV | .IDIV | .MOD | .MAX | .POW
  | .SHL | .SHR | .XOR | .OR | .AND
  | .CMPLT | .CMPNE | .CMPEQ | .THREEFRY | .LOAD | .COPY => .binary
  | .WHERE | .MULACC | .WMMA | .STORE => .ternary
  | _ => .variadic

/-- Is this a unary operation? -/
def isUnary (op : Ops) : Bool := op.arity == .unary

/-- Is this a binary operation? -/
def isBinary (op : Ops) : Bool := op.arity == .binary

/-- Is this a ternary operation? -/
def isTernary (op : Ops) : Bool := op.arity == .ternary

/-- Is this an ALU (arithmetic/logic) operation? -/
def isALU (op : Ops) : Bool :=
  op.isUnary || op.isBinary || op.isTernary

/-- Is this a movement operation? -/
def isMovement : Ops → Bool
  | .RESHAPE | .PERMUTE | .EXPAND | .PAD | .SHRINK | .FLIP => true
  | _ => false

/-- Is this a reduce operation? -/
def isReduce : Ops → Bool
  | .REDUCE_AXIS | .REDUCE | .ALLREDUCE => true
  | _ => false

/-- Is this a commutative operation? -/
def isCommutative : Ops → Bool
  | .ADD | .MUL | .MAX | .CMPNE | .CMPEQ | .XOR | .AND | .OR => true
  | _ => false

/-- Is this an associative operation? -/
def isAssociative : Ops → Bool
  | .ADD | .MUL | .MAX | .AND | .OR => true
  | _ => false

/-- Is this a comparison operation? -/
def isComparison : Ops → Bool
  | .CMPLT | .CMPNE | .CMPEQ => true
  | _ => false

/-- Does this operation produce a boolean result? -/
def producesBoolean : Ops → Bool
  | .CMPLT | .CMPNE | .CMPEQ => true
  | _ => false

/-- Can this operation be fused with others? -/
def isFusable (op : Ops) : Bool :=
  op.isALU && !op.isComparison

end Ops

/-- Operation groups for optimization -/
structure GroupOp where
  unary : List Ops := [.EXP2, .LOG2, .SIN, .COS, .TAN, .SQRT, .RECIPROCAL, .NEG, .TRUNC]
  binary : List Ops := [.ADD, .MUL, .SUB, .FDIV, .IDIV, .MAX, .MOD, .POW,
                        .SHL, .SHR, .XOR, .OR, .AND,
                        .CMPLT, .CMPNE, .CMPEQ, .THREEFRY]
  ternary : List Ops := [.WHERE, .MULACC]
  movement : List Ops := [.RESHAPE, .EXPAND, .PERMUTE, .PAD, .SHRINK, .FLIP]
  reduce : List Ops := [.REDUCE_AXIS, .REDUCE, .ALLREDUCE]
  commutative : List Ops := [.ADD, .MUL, .MAX, .CMPNE, .CMPEQ, .XOR, .AND, .OR]

/-- Default operation groups -/
def defaultGroupOp : GroupOp := {}

end TinyGrad4
