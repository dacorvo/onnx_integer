import numpy as np

def to_fixed_point(x, bitwidth, signed=True):
    """Convert a number to a FixedPoint representation

    The representation is composed of a mantissa and an implicit exponent expressed as
    a number of fractional bits, so that:

    x ~= mantissa . 2 ** -frac_bits

    The mantissa is an integer whose bitwidth and signedness are specified as parameters.

    Args:
        x: the source number or array


    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    # Evaluate the number of bits available for the mantissa
    mantissa_bits = bitwidth - 1 if signed else bitwidth
    # Evaluate the number of bits required to represent the whole part of x
    # as the power of two enclosing the absolute value of x
    # Note that it can be negative if x < 0.5
    whole_bits = np.ceil(np.log2(np.abs(x))).astype(np.int32)
    # Deduce the number of bits required for the fractional part of x
    # Note that it can be negative if the whole part exceeds the mantissa
    frac_bits = mantissa_bits - whole_bits
    # Evaluate the 'scale', which is the smallest value that can be represented (as 1)
    scale = 2. ** -frac_bits
    # Evaluate the minimum and maximum values for the mantissa
    mantissa_min = -2 ** mantissa_bits if signed else 0
    mantissa_max = 2 ** mantissa_bits - 1
    # Evaluate the mantissa by quantizing x with the scale, clipping to the min and max
    mantissa = np.clip(np.round(x / scale), mantissa_min, mantissa_max).astype(np.int32)
    return mantissa, frac_bits


def to_float(mantissa, frac_bits):
    return mantissa * 2. ** -frac_bits


if __name__ == '__main__':

    def test(x, bitwidth):
        signed = np.any(x < 0)
        mantissa, frac_bits = to_fixed_point(x, bitwidth, signed)
        # Check mantissa is correct
        value_bits = bitwidth - 1 if signed else bitwidth
        min_value = -2 ** value_bits if signed else 0
        max_value = 2 ** value_bits - 1
        assert mantissa >= min_value
        assert mantissa <= max_value
        # The conversion error is equal at most to the scale
        atol = 2. ** -frac_bits
        x_q = to_float(mantissa, frac_bits)
        np.testing.assert_allclose(x, x_q, atol=atol)
        print(f"{x} -> ({mantissa}, {frac_bits}) = {x_q}")

    test(4., 8)
    test(-4., 8)
    test(.666, 8)
    test(.042, 8)
    test(743, 8)
    test(np.pi, 8)
    test(np.pi, 7)
    test(np.pi, 6)
    test(np.pi, 5)
    test(np.pi, 4)
    test(np.pi, 3)