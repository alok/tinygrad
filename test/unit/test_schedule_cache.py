import functools
import unittest

from tinygrad import Tensor, Variable
from tinygrad.engine.schedule import pm_post_sched_cache, schedule_cache
from tinygrad.uop.ops import KernelInfo, Ops, UOp, dtypes, graph_rewrite


def custom_set0_kernel(A: UOp, num: int) -> UOp:
  return A[0].set(num).sink(arg=KernelInfo(f"custom_set0_{num}"))


class TestScheduleCache(unittest.TestCase):
  def test_bound_variable_reuses_cache(self):
    schedule_cache.clear()
    v = Variable('v', 1, 100)
    x = Tensor.ones(10).contiguous().realize()

    # first run with v=5
    t1 = (x + Tensor(v.bind(5))).sum()
    self.assertEqual(t1.item(), 60.0)
    cache_size_after_first = len(schedule_cache)

    # second run with v=10 should reuse cache
    t2 = (x + Tensor(v.bind(10))).sum()
    self.assertEqual(t2.item(), 110.0)
    self.assertEqual(len(schedule_cache), cache_size_after_first)

  def test_custom_kernel(self):
    for i in range(4):
      a = Tensor.empty(1)
      a = Tensor.custom_kernel(a, fxn=functools.partial(custom_set0_kernel, num=i))[0]
      a.realize()
      self.assertEqual(a.item(), i)

  def test_post_sched_cache_lunique_const_fallback(self):
    c = UOp.unique_const(dtypes.int, 7, device='CPU')
    c_lunique = c.replace(src=(UOp(Ops.LUNIQUE, arg=0), c.src[1]))
    c_restored = graph_rewrite(c_lunique, pm_post_sched_cache, ctx=({}, ()))
    self.assertIs(c_restored.op, Ops.CONST)
    self.assertEqual(c_restored.arg, 7)
    self.assertIs(c_restored.src[0].op, Ops.UNIQUE)
    self.assertEqual(c_restored.src[1].arg, 'CPU')

  def test_same_custom_function_reuses_cache(self):
    schedule_cache.clear()
    fxn = functools.partial(custom_set0_kernel, num=10)

    # first run
    a = Tensor.empty(1)
    a = Tensor.custom_kernel(a, fxn=fxn)[0]
    a.realize()
    self.assertEqual(a.item(), 10)
    cache_size_after_first = len(schedule_cache)

    # second run with same function should reuse cache
    b = Tensor.empty(1)
    b = Tensor.custom_kernel(b, fxn=fxn)[0]
    b.realize()
    self.assertEqual(b.item(), 10)
    self.assertEqual(len(schedule_cache), cache_size_after_first)

  def test_simple(self):
    a = Tensor.ones(10).contiguous()
    b = Tensor.ones(10).contiguous()
    Tensor.realize(a, b)

    # warm up
    for _ in range(2):
      num = (a.sum().contiguous() + b.sum().contiguous()).item()
      print(num)

    # confirm schedule cache doesn't grow
    start_len_schedule_cache = len(schedule_cache)
    for _ in range(3):
      num = (a.sum().contiguous() + b.sum().contiguous()).item()
      print(num)
    self.assertEqual(len(schedule_cache), start_len_schedule_cache)


if __name__ == "__main__":
  unittest.main()
