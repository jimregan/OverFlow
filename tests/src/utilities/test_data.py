import torch

from src.utilities.data import TextMelMotionCollate


def test_collate_function(dummy_data_uncollated, hparams):
    text_padded, input_lengths, mel_padded, motion_padded, output_lengths = TextMelMotionCollate(
        hparams.n_frames_per_step
    )(dummy_data_uncollated)
    assert text_padded.shape[1] == torch.max(input_lengths).item()
    assert mel_padded.shape[2] == torch.max(output_lengths).item()
    assert motion_padded.shape[2] == torch.max(output_lengths).item()
