
import json, random, argparse, os

def split_jsonl(input_file, train_file, val_file, test_file, ratios=(0.8,0.1,0.1), seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    random.seed(seed)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    total = len(lines)
    random.shuffle(lines)

    n_train = int(total * ratios[0])
    n_val   = int(total * ratios[1])
    n_test  = total - n_train - n_val

    train_data = lines[:n_train]
    val_data   = lines[n_train:n_train + n_val]
    test_data  = lines[n_train + n_val:]

    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    for path, subset in [(train_file,train_data),(val_file,val_data),(test_file,test_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for e in subset: f.write(json.dumps(e) + "\n")

    print(f"✅ Split complete")
    print(f"  Total: {total}")
    print(f"  Train: {n_train} ({ratios[0]*100:.1f}%)")
    print(f"  Val  : {n_val} ({ratios[1]*100:.1f}%)")
    print(f"  Test : {n_test} ({ratios[2]*100:.1f}%)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input JSONL file")
    p.add_argument("--train", required=True, help="Output train JSONL path")
    p.add_argument("--val", required=True, help="Output validation JSONL path")
    p.add_argument("--test", required=True, help="Output test JSONL path")
    p.add_argument("--ratio", nargs=3, type=float, default=[0.8,0.1,0.1],
                   help="Ratios for train val test (default 0.8 0.1 0.1)")
    args = p.parse_args()

    split_jsonl(args.input, args.train, args.val, args.test, tuple(args.ratio))