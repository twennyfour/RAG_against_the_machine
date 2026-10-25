def split_by_lines(text: str, start: int, end: int,
                   max_size: int) -> list[tuple[int, int]]:
    if end - start <= max_size:
        return [(start, end)]
    pieces: list[tuple[int, int]] = []
    piece_start = start
    position = start
    for line in text[start:end].splitlines(keepends=True):
        line_end = position + len(line)
        if line_end - piece_start > max_size and position > piece_start:
            pieces.append((piece_start, position))
            piece_start = position
        while line_end - piece_start > max_size:
            pieces.append((piece_start, piece_start + max_size))
            piece_start += max_size
        position = line_end
    if piece_start < end:
        pieces.append((piece_start, end))
    return pieces
