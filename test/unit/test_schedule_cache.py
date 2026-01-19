import unittest
from tinygrad import Tensor, Variable
from tinygrad.engine.schedule import pm_post_sched_cache, schedule_cache
from tinygrad.uop.ops import UOp, Ops, dtypes, graph_rewrite

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

  def test_bound_variable_var_vals(self):
    v = Variable('pos', 1, 100)
    x = Tensor.ones(10).contiguous().realize()

    t = x + Tensor(v.bind(42))
    _, var_vals = t.schedule_with_vars()
    self.assertEqual(var_vals, {'pos': 42})

  def test_post_sched_cache_lunique_const_fallback(self):
    c = UOp.unique_const(dtypes.int, 7, device='CPU')
    c_lunique = c.replace(src=(c.src[0], UOp(Ops.LUNIQUE, arg=0)))
    c_restored = graph_rewrite(c_lunique, pm_post_sched_cache, ctx={})
    self.assertIs(c_restored.op, Ops.CONST)
    self.assertEqual(c_restored.arg, 7)
    self.assertEqual(c_restored.src[0].arg, 'CPU')
    self.assertIs(c_restored.src[1].op, Ops.UNIQUE)

  def test_simple(self):
    a = Tensor.ones(10).contiguous()
    b = Tensor.ones(10).contiguous()
    Tensor.realize(a, b)

    # warm up
    for _ in range(2):
      num = (a.sum().contiguous()+b.sum().contiguous()).item()
      print(num)

    # confirm schedule cache doesn't grow
    start_len_schedule_cache = len(schedule_cache)
    for _ in range(3):
      num = (a.sum().contiguous()+b.sum().contiguous()).item()
      print(num)
    self.assertEqual(len(schedule_cache), start_len_schedule_cache)

if __name__ == "__main__":
  unittest.main()
