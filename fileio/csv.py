import csv

from typing import Generator

def load_csv_rows(path: str) -> Generator[dict[str, str], None, None]:
    with open(path, newline='', encoding='utf-8-sig') as csvfile: # encoding as per source file
        reader = csv.DictReader(csvfile)
        for row in reader:
            yield row