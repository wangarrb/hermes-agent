import json
from pathlib import Path
from unittest.mock import patch

from tools.media_tools import (
    _transcribe_media_path,
    audio_transcribe_tool,
    video_analyze_tool,
)


class TestAudioTranscribeTool:
    def test_transcribes_small_local_audio(self, tmp_path):
        audio_path = tmp_path / "sample.ogg"
        audio_path.write_bytes(b"fake-audio")
        out_dir = tmp_path / "out"

        with patch(
            "tools.media_tools.transcribe_audio",
            return_value={"success": True, "transcript": "hello world", "provider": "local"},
        ) as mock_transcribe:
            result = json.loads(
                audio_transcribe_tool(
                    source=str(audio_path),
                    model="small",
                    output_dir=str(out_dir),
                )
            )

        assert result["success"] is True
        assert result["transcript"] == "hello world"
        assert Path(result["transcript_path"]).exists()
        assert Path(result["metadata_path"]).exists()
        mock_transcribe.assert_called_once_with(str(audio_path), model="small")


class TestTranscribeMediaPath:
    def test_segments_large_media_and_combines_offsets(self, tmp_path):
        media_path = tmp_path / "movie.mp4"
        media_path.write_bytes(b"video")
        work_dir = tmp_path / "segments"
        seg1 = work_dir / "segment_000.wav"
        seg2 = work_dir / "segment_001.wav"
        work_dir.mkdir()
        seg1.write_bytes(b"a")
        seg2.write_bytes(b"b")

        with patch(
            "tools.media_tools._segment_media_for_transcription",
            return_value=([seg1, seg2], None),
        ), patch(
            "tools.media_tools.transcribe_audio",
            side_effect=[
                {"success": True, "transcript": "first part", "provider": "local"},
                {"success": True, "transcript": "second part", "provider": "local"},
            ],
        ):
            result = _transcribe_media_path(
                media_path=media_path,
                model=None,
                segment_seconds=300,
                work_dir=work_dir,
                is_video=True,
            )

        assert result["success"] is True
        assert result["chunk_count"] == 2
        assert "[00:00:00] first part" in result["transcript"]
        assert "[00:05:00] second part" in result["transcript"]
        assert result["provider"] == "local"


class TestVideoAnalyzeTool:
    def test_deep_video_analysis_writes_report(self, tmp_path):
        video_path = tmp_path / "launch.mp4"
        video_path.write_bytes(b"video")
        out_dir = tmp_path / "report"
        frame1 = out_dir / "frames" / "frame_001.jpg"
        frame2 = out_dir / "frames" / "frame_002.jpg"
        frame1.parent.mkdir(parents=True)
        frame1.write_bytes(b"img1")
        frame2.write_bytes(b"img2")

        with patch(
            "tools.media_tools._transcribe_media_path",
            return_value={
                "success": True,
                "transcript": "product launch transcript",
                "provider": "local",
                "providers": ["local"],
                "chunk_count": 1,
            },
        ), patch(
            "tools.media_tools._check_vision_available",
            return_value=True,
        ), patch(
            "tools.media_tools._extract_video_frames",
            return_value=(
                [
                    {
                        "index": 0,
                        "path": str(frame1),
                        "timestamp_seconds": 0,
                        "timestamp": "00:00:00",
                    },
                    {
                        "index": 1,
                        "path": str(frame2),
                        "timestamp_seconds": 30,
                        "timestamp": "00:00:30",
                    },
                ],
                None,
            ),
        ), patch(
            "tools.media_tools.registry.dispatch",
            side_effect=[
                json.dumps({"success": True, "analysis": "speaker on stage"}),
                json.dumps({"success": True, "analysis": "pricing slide shown"}),
            ],
        ):
            result = json.loads(
                video_analyze_tool(
                    source=str(video_path),
                    output_dir=str(out_dir),
                    deep=True,
                    max_frames=2,
                )
            )

        assert result["success"] is True
        assert result["frame_count"] == 2
        assert result["frame_analyses"][0]["analysis"] == "speaker on stage"
        assert Path(result["transcript_path"]).exists()
        assert Path(result["report_path"]).exists()
