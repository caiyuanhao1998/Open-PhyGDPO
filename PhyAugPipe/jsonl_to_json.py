import json, argparse, pathlib

# ------------------------- CLI ------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--jsonl_path", required=True, help="input JSONL path")
parser.add_argument("--json_path", required=True, help="output JSON path")
args = parser.parse_args()

jsonl_path = pathlib.Path(args.jsonl_path)
json_path  = pathlib.Path(args.json_path)

# ------------------------- 主逻辑 ------------------------- #
records = []
with jsonl_path.open("r", encoding="utf-8") as fin:
    for line in fin:
        records.append(json.loads(line))

with json_path.open("w", encoding="utf-8") as fjson:
    json.dump(records, fjson, ensure_ascii=False, indent=2)