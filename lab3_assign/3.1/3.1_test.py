import torch
from typing import Tuple
import os
from copy import deepcopy

def _calculate_scale_and_zeropoint(
    min_val: float, max_val: float, num_bits: int = 8) -> Tuple[float, int]:
    # Define the quantized range for uint8
    q_min = 0        
    q_max = (2 ** num_bits) - 1  # 255 for 8-bit

    # Calculate the scale (S)
    scale = (max_val - min_val) / (q_max - q_min)
    
    # Calculate the zero point (Z)
    zero_point = round(q_min - min_val.item() / scale.item())  # Convert to scalar
    
    # Ensure the zero point is within the range of [q_min, q_max]
    zero_point = max(q_min, min(q_max, zero_point))

    if min_val == max_val:
        scale = 1.0  # or some small value to prevent division by zero
        zero_point = q_min

    return scale.item(), zero_point

# Quantize function to convert torch.Tensor to torch.uint8
def quantize(x: torch.Tensor, scale: float, zero_point: int, dtype=torch.uint8):
    return torch.round(x / scale + zero_point).to(dtype)

def dequantize(x: torch.Tensor, scale: float, zero_point: int):
    return (x.to(torch.float32) - zero_point) * scale

def test_case_0():
  torch.manual_seed(999)
  test_input = torch.randn((4,4))

  min_val, max_val = torch.min(test_input), torch.max(test_input)
  scale, zero_point = _calculate_scale_and_zeropoint(min_val, max_val, 8)

  your_quant = quantize(test_input, scale, zero_point)
  your_dequant = dequantize(your_quant, scale, zero_point)

  test_case_0 = torch.Tensor([
      [-0.2623,  1.3991,  0.2842,  1.0275],
      [-0.9838, -3.4104,  1.4866,  0.2405],
      [ 1.4866, -0.3716,  0.0874,  2.1424],
      [ 0.6340, -1.1587, -0.7870,  0.0656]])

#   assert torch.allclose(your_dequant, test_case_0, atol=1e-4)
#   assert torch.allclose(your_dequant, test_input, atol=5e-2)

  return test_input, your_dequant, your_quant

### Test Case 1
def test_case_1():
  torch.manual_seed(999)
  test_input = torch.randn((8,8))

  min_val, max_val = torch.min(test_input), torch.max(test_input)
  scale, zero_point = _calculate_scale_and_zeropoint(min_val, max_val, 8)

  your_quant = quantize(test_input, scale, zero_point)
  your_dequant = dequantize(your_quant, scale, zero_point)


  test_case_1 = torch.Tensor(
      [[-0.2623,  1.3991,  0.2842,  1.0275, -0.9838, -3.4104,  1.4866,  0.2405],
      [ 1.4866, -0.3716,  0.0874,  2.1424,  0.6340, -1.1587, -0.7870,  0.0656],
      [ 0.0000, -0.6558, -1.0056,  0.3061,  0.6340, -1.0931, -1.6178,  1.5740],
      [-1.7927,  0.6121, -0.7214,  0.6121,  0.3279, -1.5959, -0.5247,  0.3498],
      [-1.3773,  1.1149, -0.7870,  0.2842,  0.9182, -1.1805, -0.7433, -1.5522],
      [ 1.0056, -0.1093,  1.3991, -0.9182, -1.1805, -0.6777, -0.3061,  0.9838],
      [ 0.2186,  1.6396,  1.0712,  1.7489,  0.0874,  0.3498,  0.9838,  1.2024],
      [-0.3935, -0.6340,  1.9238,  1.2898,  0.0219,  0.3935,  1.4866, -0.9401]])

#   assert torch.allclose(your_dequant, test_case_1, atol=1e-4)
#   assert torch.allclose(your_dequant, test_input, atol=5e-2)

  return test_input, your_dequant, your_quant


# Function to calculate the quantization error
def calculate_quantization_error(original: torch.Tensor, dequantized: torch.Tensor):
    error = original - dequantized
    avg_error = torch.mean(torch.abs(error)).item()
    max_error = torch.max(torch.abs(error)).item()
    return avg_error, max_error

# Function to save tensors and report file size
def save_tensors_and_report_size(original: torch.Tensor, quantized: torch.Tensor):
    torch.save(original, 'original_fp32.pt')
    torch.save(quantized, 'quantized_uint8.pt')

    original_size = os.path.getsize('original_fp32.pt')
    quantized_size = os.path.getsize('quantized_uint8.pt')

    print(f"Original Tensor Size (fp32): {original_size / 1024:.2f} KB")
    print(f"Quantized Tensor Size (uint8): {quantized_size / 1024:.2f} KB")

    return original_size, quantized_size

# Run Test Case 0
test_input_0, dequant_0, quant_0 = test_case_0()
avg_error_0, max_error_0 = calculate_quantization_error(test_input_0, dequant_0)
print(f"Test Case 0 - Average Error: {avg_error_0}, Maximum Error: {max_error_0}")
save_tensors_and_report_size(test_input_0, quant_0)

# Run Test Case 1
test_input_1, dequant_1, quant_1 = test_case_1()
avg_error_1, max_error_1 = calculate_quantization_error(test_input_1, dequant_1)
print(f"Test Case 1 - Average Error: {avg_error_1}, Maximum Error: {max_error_1}")
save_tensors_and_report_size(test_input_1, quant_1)  