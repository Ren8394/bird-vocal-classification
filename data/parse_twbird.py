from pathlib import Path

import numpy as np

NON_LABELED_AUDIO_DIR = "~/Desktop/Audio_data/pretrain_Audio"
LABELED_AUDIO_DIR = "~/Desktop/Audio_data/finetune_Audio"
OUTPUT_BASE_DIR = "./data"

def parse_twbird(folder_path: str, save_dir: str = None) -> None:
    """
    Create train, val, and test lists of filepath.

    Args:
        folder_path (str): Path to the folder containing the audio files.
        labeled (bool): Whether the audio files are labeled or not.
        save_dir (str): Path to the folder to save the train, val, and test lists.
    """
    folder_path = Path(folder_path).expanduser()
    file_paths = list(folder_path.glob("*.wav"))
    np.random.shuffle(file_paths)
    # Split the file paths into train, val, and test lists with 80%, 10%, and 10% respectively
    train_list, val_list, test_list = np.split(file_paths, [int(.8*len(file_paths)), int(.9*len(file_paths))])
    # Save the lists
    if save_dir is None:
        save_dir = OUTPUT_BASE_DIR
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(f"{save_dir}/train.txt", train_list, fmt="%s")
    np.savetxt(f"{save_dir}/val.txt", val_list, fmt="%s")
    np.savetxt(f"{save_dir}/test.txt", test_list, fmt="%s")

if __name__ == "__main__":
    parse_twbird(NON_LABELED_AUDIO_DIR, save_dir="./data/pretrain")
    parse_twbird(LABELED_AUDIO_DIR, save_dir="./data/finetune")