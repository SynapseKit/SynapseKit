"""Tests for VoiceAgent and its backends."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.agents.voice import VoiceAgent, VoiceResult
from synapsekit.agents.voice_backends import (
    EnergyVAD,
    OpenAITTSBackend,
    Pyttsx3TTSBackend,
    WhisperAPIBackend,
    WhisperLocalBackend,
)


class DummyLLM:
    """Mock LLM for agent testing."""

    async def call_with_tools(self, messages, tool_schemas):
        return {"content": "Mocked LLM response", "tool_calls": []}


class TestEnergyVAD:
    def test_silence_detected(self):
        vad = EnergyVAD(threshold=0.01)
        # Generate silence (zeros)
        silence = b"\x00" * 3200
        assert not vad.is_speech(silence)

    def test_speech_detected(self):
        vad = EnergyVAD(threshold=0.01)
        # Generate loud noise
        noise = b"\xff\x7f" * 1600
        assert vad.is_speech(noise)

    def test_empty_audio(self):
        vad = EnergyVAD()
        assert not vad.is_speech(b"")


class TestSTTBackends:
    @pytest.mark.asyncio
    async def test_whisper_api_missing_dep(self):
        with patch.dict("sys.modules", {"openai": None}):
            backend = WhisperAPIBackend()
            with pytest.raises(ImportError, match="openai"):
                await backend.transcribe(b"dummy")

    @pytest.mark.asyncio
    async def test_whisper_api_transcribes(self):
        mock_openai = MagicMock()
        mock_transcript = MagicMock()
        mock_transcript.text = "Hello world"
        mock_openai.OpenAI().audio.transcriptions.create.return_value = mock_transcript

        with patch.dict("sys.modules", {"openai": mock_openai}):
            backend = WhisperAPIBackend(api_key="sk-test")
            result = await backend.transcribe(b"dummy")
            assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_whisper_local_missing_dep(self):
        with patch.dict("sys.modules", {"faster_whisper": None, "whisper": None}):
            backend = WhisperLocalBackend()
            with pytest.raises(ImportError):
                await backend.transcribe(b"dummy")


class TestTTSBackends:
    @pytest.mark.asyncio
    async def test_openai_tts_missing_dep(self):
        with patch.dict("sys.modules", {"openai": None}):
            backend = OpenAITTSBackend()
            with pytest.raises(ImportError, match="openai"):
                await backend.synthesize("hello")

    @pytest.mark.asyncio
    async def test_openai_tts_synthesizes(self):
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.read.return_value = b"mp3data"
        mock_openai.OpenAI().audio.speech.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai}):
            backend = OpenAITTSBackend(api_key="sk-test")
            result = await backend.synthesize("Hello world")
            assert result == b"mp3data"

    @pytest.mark.asyncio
    async def test_pyttsx3_missing_dep(self):
        with patch.dict("sys.modules", {"pyttsx3": None}):
            backend = Pyttsx3TTSBackend()
            with pytest.raises(ImportError, match="pyttsx3"):
                await backend.synthesize("hello")

    @pytest.mark.asyncio
    async def test_empty_text(self):
        backend = OpenAITTSBackend()
        assert await backend.synthesize("   ") == b""


class MockSTTBackend:
    def __init__(self, text="Transcribed text"):
        self.text = text

    async def transcribe(self, audio, **kwargs):
        return self.text


class MockTTSBackend:
    def __init__(self, audio=b"mock-audio"):
        self.audio = audio

    async def synthesize(self, text, **kwargs):
        return self.audio


class TestVoiceAgentConstruction:
    def test_default_backends(self):
        agent = VoiceAgent(llm=DummyLLM())
        assert isinstance(agent._stt, WhisperAPIBackend)
        assert isinstance(agent._tts, OpenAITTSBackend)

    def test_local_whisper_auto(self):
        agent = VoiceAgent(llm=DummyLLM(), stt_model="base")
        assert isinstance(agent._stt, WhisperLocalBackend)

    def test_pyttsx3_auto(self):
        agent = VoiceAgent(llm=DummyLLM(), tts_model="pyttsx3")
        assert isinstance(agent._tts, Pyttsx3TTSBackend)

    def test_custom_agent(self):
        from synapsekit.agents.function_calling import FunctionCallingAgent

        custom_agent = FunctionCallingAgent(llm=DummyLLM(), tools=[])
        agent = VoiceAgent(agent=custom_agent)
        assert agent._agent is custom_agent

    def test_missing_agent_and_llm(self):
        with pytest.raises(ValueError):
            VoiceAgent()


class TestVoiceAgentFileMode:
    @pytest.mark.asyncio
    async def test_file_mode_end_to_end(self, tmp_path):
        agent = VoiceAgent(
            llm=DummyLLM(),
            stt_backend=MockSTTBackend("User said something"),
            tts_backend=MockTTSBackend(b"Agent audio response"),
        )
        input_audio = tmp_path / "input.wav"
        input_audio.write_bytes(b"dummy input")

        out_path = tmp_path / "output.mp3"

        result = await agent.run_file(input_path=input_audio, output=out_path)

        assert isinstance(result, VoiceResult)
        assert result.transcript == "User said something"
        assert result.response == "Mocked LLM response"
        assert result.audio == b"Agent audio response"
        assert result.output_path == str(out_path)
        assert out_path.exists()
        assert out_path.read_bytes() == b"Agent audio response"

    @pytest.mark.asyncio
    async def test_file_mode_no_output(self, tmp_path):
        agent = VoiceAgent(
            llm=DummyLLM(),
            stt_backend=MockSTTBackend("User said something"),
            tts_backend=MockTTSBackend(b"Agent audio response"),
        )
        input_audio = tmp_path / "input.wav"
        input_audio.write_bytes(b"dummy input")

        result = await agent.run_file(input_path=input_audio)
        assert result.output_path is None
        assert result.audio == b"Agent audio response"

    @pytest.mark.asyncio
    async def test_file_mode_no_speech(self, tmp_path):
        agent = VoiceAgent(
            llm=DummyLLM(),
            stt_backend=MockSTTBackend("   "),
            tts_backend=MockTTSBackend(b""),
        )
        input_audio = tmp_path / "input.wav"
        input_audio.write_bytes(b"dummy input")

        result = await agent.run_file(input_path=input_audio)
        assert result.transcript == ""
        assert "No speech detected" in result.response
        assert result.audio is None


class TestVoiceAgentWebSocket:
    @pytest.mark.asyncio
    async def test_ws_handler(self):
        agent = VoiceAgent(
            llm=DummyLLM(),
            stt_backend=MockSTTBackend("WS transcript"),
            tts_backend=MockTTSBackend(b"WS audio"),
            vad=False,  # Simplifies testing
        )

        class MockWS:
            def __init__(self):
                self.sent_json = None
                self.sent_bytes = None

            def __aiter__(self):
                return self

            async def __anext__(self):
                if getattr(self, "_stopped", False):
                    raise StopAsyncIteration
                self._stopped = True
                return b"fake-audio-chunk" * 1000

            async def send_json(self, data):
                self.sent_json = data

            async def send_bytes(self, data):
                self.sent_bytes = data

        ws = MockWS()
        await agent.ws_handler(ws)

        assert ws.sent_json == {
            "transcript": "WS transcript",
            "response": "Mocked LLM response",
        }
        assert ws.sent_bytes == b"WS audio"
