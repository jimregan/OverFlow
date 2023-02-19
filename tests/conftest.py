import pytest
import torch

from src.hparams import create_hparams
from src.utilities.data import TextMelMotionCollate
from tests.test_utilities import get_a_text_mel_motion_pair


@pytest.fixture
def hparams():
    hparams = create_hparams()
    hparams.checkpoint_path = None
    return hparams


@pytest.fixture
def test_batch_size():
    return 3


@pytest.fixture
def dummy_data_uncollated(test_batch_size):
    return [get_a_text_mel_motion_pair() for _ in range(test_batch_size)]


@pytest.fixture
def dummy_data(dummy_data_uncollated, hparams):
    (
        text_padded,
        input_lengths,
        mel_padded,
        motion_padded,
        output_lengths,
    ) = TextMelMotionCollate(
        hparams.n_frames_per_step
    )(dummy_data_uncollated)
    return text_padded, input_lengths, mel_padded, motion_padded, output_lengths


@pytest.fixture
def dummy_embedded_data(dummy_data, hparams):
    emb_dim = hparams.encoder_params[hparams.encoder_type]["hidden_channels"]
    text_padded, input_lengths, mel_padded, motion_padded, output_lengths = dummy_data
    embedded_input = torch.nn.Embedding(hparams.n_symbols, emb_dim)(text_padded)
    return (embedded_input, input_lengths, mel_padded, motion_padded, output_lengths)
