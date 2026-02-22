# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import xmltodict
import json
import os

from pathlib import Path
import pandas as pd


def parse_xml_dict(d):
    """takes the catch2 xml dict, performs clean-up and returns a
    list of records"""
    data = d["Catch2TestRun"]["TestCase"]
    if isinstance(data, dict):
        data = [data]
    records = []
    for cases in data:
        test_case_raw_name = cases["@name"].split(" - ")
        test_case = test_case_raw_name[0]
        value_type = test_case_raw_name[-1] if len(test_case_raw_name) > 1 else ""
        name_parts = test_case.split("::")
        benchmark_name = name_parts[0]
        dimension = name_parts[1] if len(name_parts) > 1 else ""

        if "Section" not in cases:
            continue
        sections = cases["Section"]
        if isinstance(sections, dict):
            sections = [sections]
        for section in sections:
            section_parts = section["@name"].split(" - ", 1)
            size = section_parts[0]
            variant = section_parts[1] if len(section_parts) > 1 else ""
            mean = ""
            standard_deviation = ""
            executor = ""
            for k, v in section["BenchmarkResults"].items():
                if k == "@name":
                    executor = v
                    continue
                if k == "mean":
                    mean = v["@value"]
                    continue
                if k == "standardDeviation":
                    standard_deviation = v["@value"]
                    continue
            records.append({
                "benchmark_name": benchmark_name,
                "dimension": dimension,
                "variant": variant,
                "executor": executor,
                "value_type": value_type,
                "size": size,
                "mean": mean,
                "standard_deviation": standard_deviation,
            })
    return records

def convert(root, fn):
    xml_file = Path(root) / fn
    try:
        with open(xml_file, "r") as fh:
            d = xmltodict.parse(fh.read())
            res = parse_xml_dict(d)
        with open(Path(root)/fn.replace("xml", "json"), "w") as outfile:
            json.dump(res, outfile)
    except Exception as e:
        print(e)

def display(root, fn):
    fn = fn.replace("xml", "json")
    df = pd.read_json(Path(root)/fn)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width',1000)
    try:
        print(f"\n{fn}")
        print(df.pivot(columns=["test_case", "executor"], values="mean", index=["size"]))
    except:
        pass

def main():
    mode = sys.argv[1]
    modes= ["convert", "display"]
    if not mode in  modes:
        print(f"{mode} not a valid modes, {modes}")

    src = sys.argv[2]
    root, _, files = next(os.walk(src))
    for xml_file in files:
        if not xml_file.endswith(".xml"):
            continue
        try:
            with open(xml_file, "r") as fh:
                print(f"reading {xml_file}")
                content = fh.read()
            # Strip any non-XML preamble (e.g. "Initializing NeoN") that
            # appears before the XML declaration so the parser sees valid XML.
            xml_start = content.find("<?xml")
            if xml_start > 0:
                content = content[xml_start:]
            # Truncate at the XML root closing tag so that trailing stdout
            # noise (e.g. "Finalizing NeoN" from NeoN::finalize() when spdlog
            # is disabled) does not cause the XML parser to fail.
            root_close = "</Catch2TestRun>"
            idx = content.rfind(root_close)
            if idx != -1:
                content = content[: idx + len(root_close)]
            else:
                # Root closing tag is absent — the benchmark crashed mid-run
                # (e.g. MPI_ERR_TRUNCATE). Salvage every complete <TestCase>
                # by truncating after the last </TestCase> and appending the
                # missing root close.  If there are no complete test cases the
                # parse below will fail and the file is skipped as usual.
                tc_close = "</TestCase>"
                idx = content.rfind(tc_close)
                if idx != -1:
                    content = content[: idx + len(tc_close)] + "\n</Catch2TestRun>"
            print(f"parsing {xml_file}")
            d = xmltodict.parse(content)
            res = parse_xml_dict(d)
            with open(xml_file.replace(".xml", ".json"), "w") as outfile:
                json.dump(res, outfile)
        except Exception as e:
            print("skipping", xml_file, e)


if __name__ == "__main__":
    main()
