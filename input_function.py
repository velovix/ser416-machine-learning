import re
from pathlib import Path
from os.path import basename, getsize
from collections import namedtuple
import math
import numpy as np
from random import shuffle

import numpy as np
from pandas import read_csv, read_excel


CAR_DATA_DIR = Path("Car Data")


OUTPUT_FILE_PATH = CAR_DATA_DIR / Path("Vh_Output_Data.xlsx")


STUDENT_DIR_PATTERN = re.compile("[Ss]t(?:udent)?(\d*)[Mm]ode(\d*)")


SAS_PATTERN = re.compile("[Ss]t(?:udent)?s?(\d*)[Mm]ode(\d*).*\.sas")

INPUT_SIZE = 9
OUTPUT_SIZE = 5


CarDataInput = namedtuple("CarDataInput", [
    "velocity",
    "lane_pos",
    "speed_limit",
    "steer",
    "acceleration",
    "brake",
    "long_acceleration",
    "headway_time",
    "headway_dist"])


CarDataOutput = namedtuple("CarDataOutput", [
    "mode",
    "speed",
    "error_count",
    "response_time",
    "step_count"])


def conv_to_int(val):
    if val == "-":
        return 0
    return float(val)


class CarData:

    def __init__(self):
        self._input = {}
        self._output = {}

    def input_from_row(self, student, mode, input_row):
        if (student, mode) not in self._input:
            self._input[(student, mode)] = []

        self._input[(student, mode)].append(
            CarDataInput(
                velocity=conv_to_int(input_row["Velocity"]),
                lane_pos=conv_to_int(input_row["LanePos"]),
                speed_limit=conv_to_int(input_row["SpeedLimit"]),
                steer=conv_to_int(input_row["Steer"]),
                acceleration=conv_to_int(input_row["Accel"]),
                brake=conv_to_int(input_row["Brake"]),
                long_acceleration=conv_to_int(input_row["LongAccel"]),
                headway_time=conv_to_int(input_row["HeadwayTime"]),
                headway_dist=conv_to_int(input_row["HeadwayDist"])))

    def output_from_row(self, student, mode, output_row):
        if (student, mode) not in self._output:
            self._output[(student, mode)] = []

        self._output[(student, mode)].append(
            CarDataOutput(
                mode=conv_to_int(output_row["Mode"]),
                speed=conv_to_int(output_row["Speed"]),
                error_count=conv_to_int(output_row["Number Of Errors"]),
                response_time=conv_to_int(output_row["Response Time"]),
                step_count=conv_to_int(output_row["Number Of Steps"])))

    def max_time(self):
        """Returns the largest sequence of related inputs in the dataset.
        Named max_time because that's what Tensorflow calls it.

        :return: max time
        """
        return max([len(value) for key, value in self._input.items()])

    def training_data(self):
        inputs = []
        outputs = []

        for key, output_items in self._output.items():
            if key not in self._input:
                print("Warning: No corresponding input for output", key)
                continue

            input_sequence = []
            output_sequence = []

            for output_i, output_item in enumerate(output_items):
                input_i = int((output_i / len(output_items)) *
                              len(self._input[key]))
                input_item = self._input[key][input_i]

                input_raw =  [val for val in input_item._asdict().values()]
                output_raw = [val for val in output_item._asdict().values()]

                #fake_input_raw = np.full_like(input_raw, 30)
                #fake_output_raw = np.full_like(output_raw, 60)

                input_sequence.append(input_raw)
                output_sequence.append(output_raw)
                #input_sequence.append(fake_input_raw)
                #output_sequence.append(fake_output_raw)

            input_sequence = input_sequence[:4]
            output_sequence = output_sequence[:4]

            inputs.append(input_sequence)
            outputs.append(output_sequence)

        return np.array(inputs), np.array(outputs)


def _find_delineators(csv_path):
    with open(csv_path) as csv:
        csv_data = csv.read()

    delineators = {
        "\t": csv_data.find("\t"),
        " ": csv_data.find(" "),
        ",": csv_data.find(",")
    }
    delineators = {key: value for (key, value) in delineators.items()
                   if value != -1}

    if len(delineators) == 0:
        raise RuntimeError("Could not find a delineator")

    lowest_delin = None
    for key, value in delineators.items():
        if lowest_delin is None or value < delineators[lowest_delin]:
            lowest_delin = key

    return lowest_delin



def create_car_data():
    car_data = CarData()

    mode_dirs = [x for x in CAR_DATA_DIR.glob("mode*")
                 if x.is_dir()]

    for mode_dir in mode_dirs:
        student_dirs = [x for x in mode_dir.iterdir()
                        if STUDENT_DIR_PATTERN.match(basename(x))]

        for student_dir in student_dirs:
            # Find the largest output file, which is the complete one
            largest_sas = None
            for sas in student_dir.glob("*.sas"):
                if largest_sas is None or getsize(sas) > getsize(largest_sas):
                    largest_sas = sas

            delineator = _find_delineators(largest_sas)

            # Parse the filename for student and mode info
            result = SAS_PATTERN.match(basename(largest_sas))
            if result is None:
                print("Warning: Did not match to folder", largest_sas)
            else:
                student = result.group(1)
                mode = result.group(2)

                #try:
                reader = read_csv(largest_sas, sep=delineator)
                for index, row in reader.iterrows():
                    car_data.input_from_row(int(student), int(mode), row)
                """except:
                    raise RuntimeError(
                        f"Error parsing file {result} at index {index}")"""

    last_user = None
    reader = read_excel(OUTPUT_FILE_PATH)
    for index, row in reader.iterrows():
        if ("User" in row
                and row["User"] not in ["", None]
                and not np.isnan(row["User"])):
            last_user = row["User"]

        if last_user in ["", None] or np.isnan(last_user):
            raise RuntimeError(f"Warning: Unknown student row, got "
                               f"invalid value '{last_user}'")

        mode = row["Mode"]

        car_data.output_from_row(int(last_user), int(mode), row)

    count = 0
    for key, val in car_data._output.items():
        for v in val:
            assert type(v) == CarDataOutput
            count += 1

    print("Total outputs:", count)

    print("created some car data", len(car_data._input))
    return car_data


if __name__ == "__main__":
    input_function()
