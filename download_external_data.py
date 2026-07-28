import json
import os
from datasets import load_dataset

def convert_and_save():
    print("Downloading chhatramani/nepal_5_law_RAG_QA...")
    dataset = load_dataset("chhatramani/nepal_5_law_RAG_QA")
    
    output_path = "data/external_nepal_5_law_qa.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total_written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for split in ["train", "test"]:
            if split not in dataset:
                continue
            for row in dataset[split]:
                meta = row.get("metadata", {})
                new_row = {
                    "instruction": row.get("instruction", ""),
                    "input": row.get("input", ""),
                    "output": row.get("output", ""),
                    "language": meta.get("language", "unknown"),
                    "source_act": meta.get("law_name", "unknown"),
                    "section": meta.get("section_id", ""),
                    "doc_type": "statute",
                    "example_type": meta.get("example_type", "unknown")
                }
                f.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                total_written += 1
                
    print(f"Successfully converted and saved {total_written} rows to {output_path}")

    print("\nExamples:")
    with open(output_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 3:
                print(json.dumps(json.loads(line), ensure_ascii=False, indent=2))
            else:
                break

if __name__ == "__main__":
    convert_and_save()
