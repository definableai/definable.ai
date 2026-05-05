from definable.media import File
from definable.reader import BaseReader


reader = BaseReader()
file = File(content=b"hello", filename="note.txt", mime_type="text/plain")
result = reader.read(file)
parser = reader.get_parser(file)

assert result.content == "hello"
assert type(parser).__name__ == "TextParser"
