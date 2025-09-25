import pickle
import json

def convert_pickle_to_json(pickle_path, json_path):
    """
    Loads vocabularies from a pickle file and saves them to a JSON file.
    This makes the data accessible to other languages like C#.
    """
    try:
        with open(pickle_path, "rb") as f:
            letter2id, ph2id, id2ph = pickle.load(f)

        # The third dictionary, id2ph, has integer keys.
        # JSON keys must be strings. Convert them before saving.
        id2ph_str_keys = {str(k): v for k, v in id2ph.items()}

        vocab_data = {
            "letter2id": letter2id,
            "ph2id": ph2id,
            "id2ph": id2ph_str_keys
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=4)

        print(f"✅ Successfully converted '{pickle_path}' to '{json_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{pickle_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    convert_pickle_to_json("vocabs.pkl", "vocabs.json")
