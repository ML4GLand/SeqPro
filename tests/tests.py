# def test_ascii_encode_seqs():
#     # Test 1 - Check if the returned array has the expected shape and datatype
#     seqs = ["abc", "def", "ghi"]
#     pad = 2
#     expected_output = np.array(
#         [[97, 98, 99, 0, 0], [100, 101, 102, 0, 0], [103, 104, 105, 0, 0]]
#     )
#     output = ascii_encode_seqs(seqs, pad=pad)
#     assert (
#         output.shape == expected_output.shape
#     ), f"Expected shape {expected_output.shape}, but got {output.shape}"
#     assert (
#         output.dtype == expected_output.dtype
#     ), f"Expected dtype {expected_output.dtype}, but got {output.dtype}"
#     assert np.array_equal(
#         output, expected_output
#     ), f"Expected {expected_output}, but got {output}"

#     # Test 2 - Check if the input sequence is not modified
#     seqs = ["abc", "def", "ghi"]
#     pad = 2
#     input_copy = seqs.copy()
#     _ = ascii_encode_seqs(seqs, pad=pad)
#     assert input_copy == seqs, f"Expected {input_copy}, but got {seqs}"

#     # Test 3 - Check if the function returns empty array for empty input sequence
#     seqs = []
#     pad = 0
#     expected_output = np.array([])
#     output = ascii_encode_seqs(seqs, pad=pad)
#     assert np.array_equal(
#         output, expected_output
#     ), f"Expected {expected_output}, but got {output}"
