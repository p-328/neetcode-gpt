import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        position_vec = np.arange(seq_len)[:, None]        # (seq_len, 1)
        two_i = np.arange(0, d_model, 2)                  # (d_model//2,) -> this IS "2i"
        div_term = np.power(10000, two_i / d_model)        # (d_model//2,)

        pos_en = np.zeros((seq_len, d_model))
        angles = position_vec / div_term                   # (seq_len, d_model//2) via broadcasting

        pos_en[:, 0::2] = np.sin(angles)
        pos_en[:, 1::2] = np.cos(angles[:, : pos_en[:, 1::2].shape[1]])  # handles odd d_model safely

        return np.round(pos_en, 5)
