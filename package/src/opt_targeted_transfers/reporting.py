import csv
import os
from pathlib import Path


def write_result(results_file, result):
    """Writes results to a csv file."""
    Path(results_file).parent.mkdir(exist_ok=True, parents=True)
    with open(results_file, "a+", newline="") as csvfile:
        field_names = result.keys()
        dict_writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if os.stat(results_file).st_size == 0:
            dict_writer.writeheader()
        dict_writer.writerow(result)
