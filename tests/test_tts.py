from server.tts import _wav_bytes


def test_wav_bytes_wraps_pcm_data():
    audio = _wav_bytes(b"\x01\x02\x03\x04")

    assert audio.startswith(b"RIFF")
    assert audio[8:12] == b"WAVE"
    assert audio[-4:] == b"\x01\x02\x03\x04"
