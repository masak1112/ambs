__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Michael Langguth"
__date__ = "2021-20-05"

"""
Some auxiliary functions that can be used by any video prediction model
"""


def set_and_check_pred_frames(seq_length, context_frames):
    """
    Checks if sequence length and context_frames are set properly and returns number of frames to be predicted.
    :param seq_length: number of frames/images per sequences
    :param context_frames: number of context frames/images
    :return: number of predicted frames
    """

    method = set_and_check_pred_frames.__name__

    # sanity checks
    assert isinstance(seq_length, int), "%{0}: Sequence length (seq_length) must be an integer".format(method)
    assert isinstance(context_frames, int), "%{0}: Number of context frames must be an integer".format(method)

    if seq_length > context_frames:
        return seq_length - context_frames
    else:
        raise ValueError("%{0}: Sequence length ({1}) must be larger than context frames ({2})."
                         .format(method, seq_length, context_frames))