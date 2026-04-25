from datasets import load_dataset
import os

if __name__ == "__main__":
    dataset = load_dataset("BeIR/fiqa", "corpus")
    save_path = "data/raw/fiqa_corpus"
    dataset.save_to_disk(save_path)

    print("Current working dir:", os.getcwd())
    print("Absolute save path:", os.path.abspath(save_path))