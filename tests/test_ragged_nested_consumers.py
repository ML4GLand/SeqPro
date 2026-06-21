"""Consumer-case exit tests for Spec C: nested R=2 Ragged + string-under-axis.

Three real-world use-case shapes drawn from the shape survey
(genoray / GenVarLoader / genvarformer):

1. test_consumer_alleles_string_under_axis
   (ploidy=2 flattened, ~variants) opaque allele strings -> chars (R=2).

2. test_consumer_flat_variant_windows
   (batch, ploidy, ~variants, ~window) flat variant windows -> dense tensor.

3. test_consumer_codon_annotations_record
   (regions, ~genes, ~codons) two-field R=2 record -> pack + shared inner offsets.
"""

from __future__ import annotations

import numpy as np

from seqpro.rag._core import Ragged


def test_consumer_alleles_string_under_axis():
    # (ploidy=2 flattened, ~variants) opaque allele strings -> chars (R=2)
    data = np.frombuffer(b"ACGTT", "S1")
    o0 = np.array([0, 2, 3], dtype=np.int64)  # 2 rows: 2,1 variants
    str_off = np.array([0, 1, 2, 5], dtype=np.int64)  # 3 variants: A, C, GTT
    rag = Ragged.from_offsets(data, (2, None), o0, str_offsets=str_off)
    chars = rag.to_chars()
    assert chars.shape == (2, None, None) and chars.dtype == np.dtype("S1")
    # chars[1, 0]: row 1 has 1 variant (var-index 2), its 0th string = GTT
    assert chars[1, 0].tobytes() == b"GTT"


def test_consumer_flat_variant_windows():
    data = np.arange(10, dtype=np.int32)
    # o0=[0,2,3]: 2 outer rows; row0→2 middles (0:2), row1→1 middle (2:3)
    # o1=[0,3,5,10]: mid0=data[0:3]=[0,1,2], mid1=data[3:5]=[3,4], mid2=data[5:10]=[5,6,7,8,9]
    rag = Ragged.from_offsets(
        data,
        (2, None, None),
        [np.array([0, 2, 3]), np.array([0, 3, 5, 10])],
    )
    dense = rag.to_padded(-1)  # model-input tensor (both-dense)
    # outer M=max(2,1)=2, inner K=max(3,2,5)=5
    assert dense.shape == (2, 2, 5)


def test_consumer_codon_annotations_record():
    offs = [np.array([0, 2, 3]), np.array([0, 3, 5, 10])]
    # o0=[0,2,3]: row0 has middles 0:2 (2 middles), row1 has middles 2:3 (1 middle)
    # o1=[0,3,5,10]: mid0=data[0:3], mid1=data[3:5], mid2=data[5:10]
    pos = Ragged.from_offsets(np.arange(10, dtype=np.int32), (2, None, None), offs)
    strand = Ragged.from_offsets(np.ones(10, dtype=np.int8), (2, None, None), offs)
    rec = Ragged.from_fields({"codon_pos": pos, "strand": strand})
    assert rec.fields == ["codon_pos", "strand"]
    # STRONG field-access checks: assert real per-field values that fail on regression.
    # rec["codon_pos"][0, 1]: pos[0] → row0 of R=2 → R=1 with 2 middles (mid0,mid1);
    #   then [1] → mid1 = pos.data[3:5] = [3, 4]
    np.testing.assert_array_equal(
        rec["codon_pos"][0, 1], np.array([3, 4], dtype=np.int32)
    )
    # rec["strand"][0, 1]: same indexing on ones array → strand.data[3:5] = [1, 1]
    np.testing.assert_array_equal(rec["strand"][0, 1], np.array([1, 1], dtype=np.int8))
    packed = rec.to_packed()
    # shared inner offsets across fields after pack: _to_packed_record_r2 binds all
    # fields to the same shared_packed list, so offsets[1] is the identical object
    assert packed["codon_pos"]._layout.offsets[1] is packed["strand"]._layout.offsets[1]
