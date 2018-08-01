import torch
import math

# f(x) = |x-10|
class FunctionCocobExample(torch.autograd.Function):

  def forward(self, input):
    self.save_for_backward(input)
    value = (input-1).abs()
    return value

  def backward(self, _):

    input, = self.saved_tensors
    grad   = (input-1).sign()
    return grad


# f(x) = x^4 + 6x^2 + 12(x-4)e^(x-1)
class WierdFunctionExample(torch.autograd.Function):

  def forward(self, input):

    self.save_for_backward(input)
    e = math.e*torch.ones(input.size())
    value = input.pow(4) + 6*input.pow(2) + 12*(input-4)*e.pow(input-1)
    return value

  def backward(self, _):

    input, = self.saved_tensors
    e = math.e*torch.ones(input.size())
    grad = 4*input.pow(3) + 12*input + 12*(input-3)*e.pow(input-1)
    return grad