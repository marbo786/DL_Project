"""Bounded in-memory multipart upload for the local comparison API."""

from io import BytesIO

import anyio
from fastapi import HTTPException, Request
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header
from starlette.requests import ClientDisconnect

OVERHEAD = 8192
UPLOAD_SECONDS = 20


async def comparison_files(request: Request, max_bytes: int) -> tuple[bytes, bytes]:
    media, options = parse_options_header(request.headers.get("content-type", ""))
    if media.lower() != b"multipart/form-data":
        raise HTTPException(415, "Upload a multipart reference and candidate.")
    boundary = options.get(b"boundary", b"")
    if not 0 < len(boundary) <= 70 or any(c < 32 or c > 126 for c in boundary):
        raise HTTPException(400, "Invalid multipart boundary.")
    max_body = 2 * max_bytes + OVERHEAD
    length = request.headers.get("content-length")
    if length:
        if not length.isascii() or not length.isdecimal() or len(length) > 10:
            raise HTTPException(400, "Invalid upload length.")
        if int(length) > max_body:
            raise HTTPException(413, "Use two images no larger than 8 MiB each.")

    field, value = bytearray(), bytearray()
    disposition = b""
    current = None
    complete = False
    received = 0
    header_bytes = 0
    parts = {}

    def append_header(buffer, data, start, end):
        nonlocal header_bytes
        header_bytes += end - start
        if header_bytes > OVERHEAD:
            raise HTTPException(400, "Multipart headers are too large.")
        buffer.extend(data[start:end])

    def header_end():
        nonlocal disposition
        if field.lower() == b"content-disposition":
            disposition = bytes(value)
        field.clear()
        value.clear()

    def part_begin():
        nonlocal current, disposition
        current = None
        disposition = b""

    def headers_finished():
        nonlocal current
        kind, parameters = parse_options_header(disposition)
        name = parameters.get(b"name")
        if kind != b"form-data" or name not in {b"reference", b"candidate"} or b"filename" not in parameters:
            raise HTTPException(400, "Provide exactly one reference and one candidate file.")
        if name in parts:
            raise HTTPException(400, "Each image field must appear once.")
        current = parts[name] = BytesIO()

    def part_data(data, start, end):
        if current is None:
            raise HTTPException(400, "Invalid multipart image part.")
        if current.tell() + end - start > max_bytes:
            raise HTTPException(413, "Use images no larger than 8 MiB each.")
        current.write(memoryview(data)[start:end])

    def end():
        nonlocal complete
        complete = True

    parser = MultipartParser(boundary, {
        "on_part_begin": part_begin,
        "on_header_field": lambda *args: append_header(field, *args),
        "on_header_value": lambda *args: append_header(value, *args),
        "on_header_end": header_end,
        "on_headers_finished": headers_finished,
        "on_part_data": part_data,
        "on_end": end,
    })
    try:
        with anyio.fail_after(UPLOAD_SECONDS):
            async for chunk in request.stream():
                received += len(chunk)
                if received > max_body:
                    raise HTTPException(413, "Use two images no larger than 8 MiB each.")
                parser.write(chunk)
        parser.finalize()
        if not complete or set(parts) != {b"reference", b"candidate"}:
            raise HTTPException(400, "Provide a complete reference and candidate upload.")
        if not all(part.tell() for part in parts.values()):
            raise HTTPException(400, "Both image files must contain data.")
        return parts[b"reference"].getvalue(), parts[b"candidate"].getvalue()
    except (ClientDisconnect, MultipartParseError) as error:
        raise HTTPException(400, "The image upload was incomplete or malformed.") from error
    except TimeoutError as error:
        raise HTTPException(408, "The image upload timed out.") from error
    finally:
        for part in parts.values():
            part.close()
