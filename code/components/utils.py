def parse_text_input(text_input):
    return [line.strip() for line in text_input.strip().splitlines() if line.strip()]
