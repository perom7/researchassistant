from __future__ import annotations

from pathlib import Path


def overlay_audio_on_video(video_path: str | Path, audio_path: str | Path, out_path: str | Path) -> str:
    """Overlay audio on a video; loops/cuts video to match audio."""

    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

    video_path = Path(video_path)
    audio_path = Path(audio_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video = VideoFileClip(str(video_path))
    audio = AudioFileClip(str(audio_path))

    if video.duration < audio.duration:
        loops = int(audio.duration / video.duration) + 1
        video = video.loop(n=loops)

    video = video.subclip(0, audio.duration)

    if video.audio is not None:
        original = video.audio.volumex(0.2)
        final_audio = CompositeAudioClip([original, audio])
    else:
        final_audio = audio

    final = video.set_audio(final_audio)
    final.write_videofile(str(out_path), codec="libx264", audio_codec="aac", temp_audiofile="temp_audio.m4a", remove_temp=True)

    video.close()
    audio.close()
    final.close()

    return str(out_path)
