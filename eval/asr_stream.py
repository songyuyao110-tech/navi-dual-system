"""Real streaming ASR adapter (Volcengine / Doubao bigasr sauc, v3 binary protocol).

stream_asr(source) yields (t_ms, text, final) updates in the same shape the TurnSession consumes.
Credentials come from CLOUD_ASR_* env vars, falling back to ~/.zshrc exports; values are never printed.
`simple-websocket` (already installed as a Flask-SocketIO dependency) provides the sync WS client.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

DEFAULT_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
PROTOCOL_VERSION, HEADER_SIZE = 0x1, 0x1
MESSAGE_TYPE_FULL_CLIENT_REQUEST, MESSAGE_TYPE_AUDIO_ONLY_REQUEST, MESSAGE_TYPE_FULL_SERVER_RESPONSE = 0x1, 0x2, 0x9
MESSAGE_TYPE_ERROR = 0xF
SERIALIZATION_NONE, SERIALIZATION_JSON, COMPRESSION_GZIP = 0x0, 0x1, 0x1
FLAG_NONE, FLAG_HAS_SEQUENCE, FLAG_LAST_PACKET_HAS_SEQUENCE = 0x0, 0x1, 0x3


def credentials() -> dict:
    names = ("CLOUD_ASR_APP_ID", "CLOUD_ASR_ACCESS_TOKEN", "CLOUD_ASR_RESOURCE_ID", "CLOUD_ASR_URL")
    values = {name: (os.environ.get(name) or "").strip() for name in names}
    if not values["CLOUD_ASR_ACCESS_TOKEN"]:
        rc = Path.home() / ".zshrc"
        if rc.is_file():
            exports = dict(re.findall(r"^export (CLOUD_ASR_[A-Z_]+)=(.*)$", rc.read_text(), re.M))
            for name, value in exports.items():
                if name in values and not values[name]:
                    values[name] = value.strip().strip('"').strip("'")
    if not (values["CLOUD_ASR_APP_ID"] and values["CLOUD_ASR_ACCESS_TOKEN"]):
        raise RuntimeError("Streaming ASR credentials unavailable (CLOUD_ASR_APP_ID / CLOUD_ASR_ACCESS_TOKEN)")
    values["CLOUD_ASR_RESOURCE_ID"] = values["CLOUD_ASR_RESOURCE_ID"] or "volc.bigasr.sauc.duration"
    values["CLOUD_ASR_URL"] = values["CLOUD_ASR_URL"] or DEFAULT_URL
    return values


def read_pcm(source, sample_rate=16000) -> bytes:
    """16 kHz mono 16-bit PCM from a wav file; `mic` reads the same format from the default input device."""
    if str(source) == "mic":
        import sounddevice  # optional: only needed for live microphone input
        frames = int(sample_rate * 0.1)
        with sounddevice.RawInputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
            raise RuntimeError("mic mode requires an explicit stop condition; use a wav file for the PoC")
    with wave.open(str(source), "rb") as handle:
        if handle.getframerate() != sample_rate or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"Requires {sample_rate} Hz mono 16-bit PCM wav")
        return handle.readframes(handle.getnframes())


def _header(message_type, flags, serialization, compression) -> bytes:
    return bytes([(PROTOCOL_VERSION << 4) | HEADER_SIZE, (message_type << 4) | flags,
                  (serialization << 4) | compression, 0x00])


def _frame(header, payload, sequence=None) -> bytes:
    if sequence is not None:
        return header + sequence.to_bytes(4, "big", signed=True) + len(payload).to_bytes(4, "big") + payload
    return header + len(payload).to_bytes(4, "big") + payload


def handshake_frame(sample_rate=16000, uid="poc-streaming-eot") -> bytes:
    payload = {"user": {"uid": uid},
               "audio": {"format": "pcm", "codec": "raw", "rate": sample_rate, "bits": 16,
                         "channel": 1, "language": "zh-CN"},
               "request": {"model_name": "bigmodel", "enable_itn": True, "enable_punc": False,
                           "result_type": "full"}}
    return _frame(_header(MESSAGE_TYPE_FULL_CLIENT_REQUEST, FLAG_NONE, SERIALIZATION_JSON, COMPRESSION_GZIP),
                  gzip.compress(json.dumps(payload, ensure_ascii=False).encode()))


def audio_frame(chunk, sequence, last=False) -> bytes:
    flags = FLAG_LAST_PACKET_HAS_SEQUENCE if last else FLAG_HAS_SEQUENCE
    return _frame(_header(MESSAGE_TYPE_AUDIO_ONLY_REQUEST, flags, SERIALIZATION_NONE, COMPRESSION_GZIP),
                  gzip.compress(chunk), sequence=-abs(sequence) if last else sequence)


def parse_message(message) -> list:
    """Return [(text, is_final)] for one server frame; raises on an error frame."""
    if isinstance(message, str):
        message = message.encode()
    if len(message) < 4:
        raise ValueError("ASR frame too short")
    message_type, flags = (message[1] >> 4) & 0x0F, message[1] & 0x0F
    serialization, compression = (message[2] >> 4) & 0x0F, message[2] & 0x0F
    if message_type == MESSAGE_TYPE_ERROR:
        code = int.from_bytes(message[4:8], "big")
        size = int.from_bytes(message[8:12], "big")
        payload = message[12:12 + size]
        if compression == COMPRESSION_GZIP:
            payload = gzip.decompress(payload)
        raise RuntimeError(f"ASR error {code}: {payload.decode('utf-8', errors='replace')}")
    if message_type != MESSAGE_TYPE_FULL_SERVER_RESPONSE:
        return []
    offset = 8 if flags in {FLAG_HAS_SEQUENCE, FLAG_LAST_PACKET_HAS_SEQUENCE} else 4
    size = int.from_bytes(message[offset:offset + 4], "big")
    payload = message[offset + 4:offset + 4 + size]
    if compression == COMPRESSION_GZIP and payload:
        payload = gzip.decompress(payload)
    if serialization != SERIALIZATION_JSON or not payload:
        return []
    body = json.loads(payload.decode("utf-8"))
    result = body.get("result")
    if isinstance(result, list):
        result = result[0] if result else None
    if not isinstance(result, dict):
        return []
    text = str(result.get("text", "")).strip()
    utterances = [item for item in (result.get("utterances") or []) if isinstance(item, dict)]
    # The sauc protocol reports the turn boundary on the sentence, not on the top-level result.
    is_final = bool(result.get("is_final") or result.get("definite")) or any(
        item.get("definite") for item in utterances)
    # end_time = end of the detected speech segment in ms from stream start; the closest thing the
    # ASR gives us to "where the audio VAD ended" (display only, it does not gate the state machine).
    ends = [item["end_time"] for item in utterances if isinstance(item.get("end_time"), (int, float))]
    if isinstance(result.get("end_time"), (int, float)):
        ends.append(result["end_time"])
    end_ms = max(ends) if ends else None
    return [(text, is_final, end_ms)] if text else []


def _drain(ws, budget_s):
    """Collect (text, is_final) updates for up to budget_s; also report whether the socket closed."""
    from simple_websocket.errors import ConnectionClosed
    updates = []
    deadline = time.monotonic() + budget_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return updates, False
        try:
            message = ws.receive(timeout=remaining)
        except TimeoutError:
            return updates, False
        except ConnectionClosed:
            return updates, True
        if message is None:
            return updates, False
        updates.extend(parse_message(message))


class ASRStream:
    """Push-based session for live audio (microphone): push 20 ms PCM frames, poll partials.

    Same v3 protocol as the file replay; the handshake consumes sequence 1, audio starts at 2.
    """

    def __init__(self, sample_rate=16000, *, connect_timeout_s=5.0):
        from simple_websocket import Client
        config = credentials()
        self.sample_rate = sample_rate
        self.url = config["CLOUD_ASR_URL"]
        headers = {"X-Api-Resource-Id": config["CLOUD_ASR_RESOURCE_ID"],
                   "X-Api-Request-Id": str(uuid.uuid4()), "X-Api-Sequence": "-1",
                   "X-Api-Connect-Id": str(uuid.uuid4()),
                   "X-Api-App-Key": config["CLOUD_ASR_APP_ID"],
                   "X-Api-Access-Key": config["CLOUD_ASR_ACCESS_TOKEN"]}
        self._ws = Client.connect(self.url, headers=headers, receive_bytes=65536, max_message_size=1 << 22)
        self._sequence = 2
        self._seen = ""
        self._closed = False
        self._ws.send(handshake_frame(sample_rate))
        updates, self._closed = _drain(self._ws, 0.3)
        self._consume(updates)

    def _consume(self, updates):
        fresh = []
        for text, final, end_ms in updates:
            if text != self._seen or final:
                self._seen = text
                fresh.append((text, final, end_ms))
        return fresh

    def push(self, pcm, *, last=False):
        if not pcm or self._closed:
            return
        if last:
            self._ws.send(audio_frame(pcm, self._sequence, last=True))
        else:
            self._ws.send(audio_frame(pcm, self._sequence))
        self._sequence += 1

    def poll(self):
        """New (text, is_final) updates since the last call; never blocks longer than ~20 ms."""
        updates, self._closed = _drain(self._ws, 0.001)
        return self._consume(updates)

    def finish(self, timeout_s=5.0):
        """Signal end of audio with a silent last packet and collect the final result."""
        collected = []
        try:
            self.push(b"\x00" * int(self.sample_rate * 0.02) * 2, last=True)
        except Exception:
            return collected
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            updates, closed = _drain(self._ws, 0.2)
            fresh = self._consume(updates)
            collected.extend(fresh)
            if any(final for _text, final, _end in fresh):
                break
            if closed:
                break
        return collected

    def close(self):
        self._closed = True
        try:
            self._ws.close()
        except Exception:
            pass


def stream_asr(source, *, sample_rate=16000, chunk_ms=100, realtime=True, final_timeout_s=8.0):
    """Yield (t_ms, text, final) for a wav path (or "mic"), paced in real time like a live stream."""
    pcm = read_pcm(source, sample_rate)
    session = ASRStream(sample_rate)
    started = time.monotonic()

    def now_ms():
        return int((time.monotonic() - started) * 1000)

    step = int(sample_rate * 2 * chunk_ms / 1000)
    try:
        for offset in range(0, len(pcm), step):
            chunk = pcm[offset:offset + step]
            session.push(chunk)          # finish() sends the single "last packet"; never send two
            if realtime:
                time.sleep(chunk_ms / 1000.0)
            for text, final, _end_ms in session.poll():
                yield now_ms(), text, final
            if session._closed:
                return
        for text, final, _end_ms in session.finish(final_timeout_s):
            yield now_ms(), text, final
    finally:
        session.close()


if __name__ == "__main__":
    for update in stream_asr(sys.argv[1]):
        print(json.dumps(update, ensure_ascii=False))