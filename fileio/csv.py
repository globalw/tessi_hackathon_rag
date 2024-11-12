import csv

from typing import Generator

def load_csv_rows(path: str) -> Generator[dict[str, str], None, None]:
    with open(path, newline='', encoding='utf-8-sig') as csvfile: # encoding as per source file
        reader = csv.DictReader(csvfile)
        for row in reader:
            yield row

def write_dict_to_csv(path: str, data: dict[str,str]):
    with open(path, mode='w', newline='') as csvfile:
        fieldnames = data[0].keys()
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for row in data:
            writer.writerow(row)