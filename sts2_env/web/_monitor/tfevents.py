from __future__ import annotations

import struct
from datetime import datetime, timezone
from pathlib import Path

from sts2_env.web._monitor.types import ObservedFact, TfEventsCursor

_CRC_MASK = 0xFFFFFFFF
# Castagnoli CRC-32C, reflected. TensorFlow TFRecords use this, not zlib CRC-32.
_CRC32C_POLY = 0x82F63B78
_TF_CRC_MASK = 0xA282EAD8


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ _CRC32C_POLY if crc & 1 else crc >> 1
    return crc ^ 0xFFFFFFFF


def _masked_crc32(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + _TF_CRC_MASK) & _CRC_MASK


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while pos < len(buf):
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
    raise ValueError("truncated varint")


def _read_fixed32(buf: bytes, pos: int) -> tuple[float, int]:
    if pos + 4 > len(buf):
        raise ValueError("truncated fixed32")
    (raw,) = struct.unpack("<f", buf[pos : pos + 4])
    return raw, pos + 4


def _decode_summary_value(payload: bytes) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    pos = 0
    while pos < len(payload):
        tag, pos = _read_varint(payload, pos)
        field = tag >> 3
        wire = tag & 7
        if wire == 2:
            length, pos = _read_varint(payload, pos)
            chunk = payload[pos : pos + length]
            pos += length
            if field == 1:
                inner_pos = 0
                tag_name: str | None = None
                value: float | None = None
                while inner_pos < len(chunk):
                    itag, inner_pos = _read_varint(chunk, inner_pos)
                    ifield = itag >> 3
                    iwire = itag & 7
                    if iwire == 2 and ifield == 1:
                        ln, inner_pos = _read_varint(chunk, inner_pos)
                        tag_name = chunk[inner_pos : inner_pos + ln].decode("utf-8", "replace")
                        inner_pos += ln
                    elif iwire == 5 and ifield == 2:
                        value, inner_pos = _read_fixed32(chunk, inner_pos)
                    elif iwire == 2:
                        ln, inner_pos = _read_varint(chunk, inner_pos)
                        inner_pos += ln
                    elif iwire == 0:
                        _, inner_pos = _read_varint(chunk, inner_pos)
                    else:
                        break
                if tag_name is not None and value is not None:
                    out.append((tag_name, value))
        elif wire == 0:
            _, pos = _read_varint(payload, pos)
        else:
            break
    return out


def decode_event_payload(payload: bytes) -> tuple[int | None, list[tuple[str, float]]]:
    step: int | None = None
    scalars: list[tuple[str, float]] = []
    pos = 0
    while pos < len(payload):
        tag, pos = _read_varint(payload, pos)
        field = tag >> 3
        wire = tag & 7
        if wire == 0 and field == 2:
            step, pos = _read_varint(payload, pos)
        elif wire == 2 and field == 3:
            length, pos = _read_varint(payload, pos)
            summary = payload[pos : pos + length]
            pos += length
            scalars.extend(_decode_summary_value(summary))
        elif wire == 2:
            length, pos = _read_varint(payload, pos)
            pos += length
        elif wire == 0:
            _, pos = _read_varint(payload, pos)
        elif wire == 1:
            pos += 8
        elif wire == 5:
            pos += 4
        else:
            break
    return step, scalars


def parse_tfevents(
    path: Path, cursor: TfEventsCursor | None
) -> tuple[TfEventsCursor, tuple[ObservedFact, ...], bool]:
    data = path.read_bytes()
    size = len(data)
    offset = 0 if cursor is None else cursor.byte_offset
    max_step = 0 if cursor is None else cursor.max_step
    # Rotation replaces the file with a shorter one. The old offset is past
    # EOF, and keeping max_step would report a step the new file never had.
    if cursor is not None and offset > size:
        offset = 0
        max_step = 0
    facts: list[ObservedFact] = []
    delivered = False
    at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

    while offset + 12 <= size:
        length = struct.unpack("<Q", data[offset : offset + 8])[0]
        len_crc = struct.unpack("<I", data[offset + 8 : offset + 12])[0]
        if _masked_crc32(data[offset : offset + 8]) != len_crc:
            break
        record_start = offset + 12
        record_end = record_start + length
        if record_end + 4 > size:
            break
        payload = data[record_start:record_end]
        data_crc = struct.unpack("<I", data[record_end : record_end + 4])[0]
        if _masked_crc32(payload) != data_crc:
            break
        try:
            step, scalars = decode_event_payload(payload)
        except ValueError:
            break
        offset = record_end + 4
        if step is not None:
            max_step = max(max_step, step)
        for tag, value in scalars:
            kind: str = "tb_fps" if tag == "time/fps" else "tb_scalar"
            facts.append(
                ObservedFact(
                    kind=kind,  # type: ignore[arg-type]
                    tag=tag,
                    step=step,
                    value=value,
                    at=at,
                )
            )
            delivered = True
        if step is not None and not scalars:
            facts.append(ObservedFact(kind="tb_step", step=step, at=at))
            delivered = True

    new_cursor = TfEventsCursor(str(path), size, offset, max_step)
    return new_cursor, tuple(facts), delivered


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)


def _encode_summary_value(tag: str, value: float) -> bytes:
    tag_b = tag.encode("utf-8")
    inner = b"\x0a" + _encode_varint(len(tag_b)) + tag_b
    inner += b"\x15" + struct.pack("<f", value)
    return b"\x0a" + _encode_varint(len(inner)) + inner


def build_event_payload(step: int, tag: str, value: float) -> bytes:
    summary = _encode_summary_value(tag, value)
    out = b"\x10" + _encode_varint(step)
    out += b"\x1a" + _encode_varint(len(summary)) + summary
    return out


def wrap_tfrecord(payload: bytes) -> bytes:
    length = len(payload)
    len_bytes = struct.pack("<Q", length)
    len_crc = struct.pack("<I", _masked_crc32(len_bytes))
    data_crc = struct.pack("<I", _masked_crc32(payload))
    return len_bytes + len_crc + payload + data_crc
