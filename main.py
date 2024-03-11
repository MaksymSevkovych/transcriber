import os
from pathlib import Path

import moviepy.editor as editor
import torch
from faster_whisper import WhisperModel

VIDEO_DIR_PATH = "./videos"
AUDIO_DIR_PATH = "./audios"
TEXT_DIR_PATH = "./texts"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def MP4ToMP3(mp4: str, mp3: str) -> None:
    file_to_convert = editor.AudioFileClip(mp4)
    file_to_convert.write_audiofile(mp3)
    file_to_convert.close()


def transcribe_audio(model, mp3, text_file) -> None:
    segments, info = model.transcribe(mp3, beam_size=5)

    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )
    text = []
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        text.append(segment.text)

    with open(text_file, "w") as writer:
        text_to_write = "".join(text)
        writer.write(text_to_write)


def convert_mp4_to_mp3():
    for file in os.listdir(VIDEO_DIR_PATH):
        if not file.endswith(".mp4"):
            continue
        filename = Path(file).stem
        print(f"converting file: {filename}")

        mp4_filepath = os.path.join(VIDEO_DIR_PATH, file)
        mp3_filepath = os.path.join(AUDIO_DIR_PATH, f"{Path(file).stem}.mp3")

        MP4ToMP3(mp4_filepath, mp3_filepath)


def main() -> None:
    # convert videos
    convert_mp4_to_mp3()

    # if converted, transcribe audio:
    model = WhisperModel("large-v2", device=DEVICE, compute_type="int8")
    print("Whisper loaded")

    for file in os.listdir(AUDIO_DIR_PATH):
        if not file.endswith(".mp3"):
            continue

        mp3_filepath = os.path.join(AUDIO_DIR_PATH, file)
        stem = Path(file).stem
        text_filepath = os.path.join(TEXT_DIR_PATH, f"{stem}.txt")
        print(f"transcribing file: {stem}")
        transcribe_audio(model, mp3_filepath, text_filepath)


if __name__ == "__main__":
    main()
