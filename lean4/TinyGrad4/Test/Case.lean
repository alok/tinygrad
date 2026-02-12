import LSpec
import TinyGrad4.Test.Profiles

namespace TinyGrad4.Test

structure TestCase where
  name : String
  group : String
  minProfile : TestProfile
  tags : List String := ["cpu"]
  suite : RunConfig → LSpec.TestSeq

end TinyGrad4.Test
