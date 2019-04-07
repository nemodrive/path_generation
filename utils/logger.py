import os
import utils
import logging
import csv
import collections
import numpy as np
import sys


def get_log_path(model_dir: str):
    return os.path.join(model_dir, "log.txt")


def get_logger(model_dir: str):
    path = get_log_path(model_dir)
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def get_csv_path(model_dir: str):
    return os.path.join(model_dir, "log.csv")


def get_csv_writer(model_dir: str):
    csv_path = get_csv_path(model_dir)
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = np.mean(array)
    d["std"] = np.std(array)
    d["min"] = np.amin(array)
    d["max"] = np.amax(array)
    return d


SYNTHESIS = {
    "μ": lambda x: [np.mean(x)],
    "+": lambda x: [np.sum(x)],
    ".": lambda x: [x[-1]],
    ":": lambda x: [x],
    "μσmM": lambda x: list(synthesize(x).values())
}


class MultiLogger:
    def __init__(self, model_dir: str, tb: bool, print_extra_keys: list = None):
        self.logger = get_logger(model_dir)
        self.csv_file, self.csv_writer = get_csv_writer(model_dir)
        self.tb_writer = None
        self.tb = tb
        self.tb_writer = None
        self.print_extra_keys = print_extra_keys

        if tb:
            from tensorboardX import SummaryWriter
            self.tb_writer = SummaryWriter(model_dir)

        self.iter_no = 0
        self.header = None
        self.header_processing = None
        self.log_format = None

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def set_header(self, header: collections.OrderedDict):
        """
        :param header: {key: [synthesis_type, print_name, print_format]
        """

        assert self.header is None, "Header already set"
        start = collections.OrderedDict({"iter_no": [":", "i", "{}"]})
        start.update(header)
        header = start

        self.csv_file.flush()
        self.csv_writer.writerow(header.keys())
        self.header = header

        self.header_processing = header_processing = dict({})

        log_format = ""
        for k, v in header.items():
            do_print = False
            synth = SYNTHESIS[v[0]]
            if isinstance(v[1], str):
                log_format += f"{v[1]}:{v[0]}"

                # Add multiple log elements
                test_synth = synth(np.random.rand(10))
                for _ in range(len(test_synth)):
                    log_format += f" {v[2]}"

                log_format += " | "

                do_print = True
            header_processing[k] = (synth, do_print)

        self.log_format = log_format

    def write(self, source_data: collections.OrderedDict):
        header_prep = self.header_processing

        print_list = [self.iter_no]
        data = [self.iter_no]
        for k, v in source_data.items():
            prep, do_print = header_prep[k]
            wv = prep(v)
            data += wv
            if do_print:
                print_list += wv

        self.csv_writer.writerow(data)
        self.csv_file.flush()

        if self.tb:
            for field, value in zip(self.header, data):
                self.tb_writer.add_scalar(field, value, self.iter_no)
        self.iter_no += 1

        # add log fields that are not in the standard log format (for example value_int)
        self.logger.info(self.log_format.format(*print_list))


