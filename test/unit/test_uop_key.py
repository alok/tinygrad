import sys
import unittest

from tinygrad import dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp

class TestUOpKey(unittest.TestCase):
  def test_key_no_recursion_error(self):
    old_limit = sys.getrecursionlimit()
    try:
      sys.setrecursionlimit(256)
      u = UOp.const(dtypes.int, 0)
      for _ in range(600): u = u.f(Ops.NOOP)
      k = u.key
      self.assertIsInstance(k, bytes)
      self.assertEqual(len(k), 32)
    finally:
      sys.setrecursionlimit(old_limit)

