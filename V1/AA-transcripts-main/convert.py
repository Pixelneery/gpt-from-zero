from pathlib import Path

folder = Path("C:\dev\gpt-from-zero\AA-transcripts-main\Phoenix Wright Transcripts")
output = Path("C:\dev\gpt-from-zero\script.txt")

with open(output, "a", encoding="UTF-8") as out:
    for file in folder.rglob("*.txt"):
        with open(file, "r", encoding="UTF-8") as cur:
            out.write(cur.read())