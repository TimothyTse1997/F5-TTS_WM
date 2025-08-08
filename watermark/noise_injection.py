import json
from pathlib import Path

import torch
from f5_tts.api import F5TTS

def get_audio_and_text_fname_from_path(speaker_path):
    audio_path = list(speaker_path.glob("**/*.wav"))[0]
    audio_id = audio_path.name.split(".")[0]
    text_path = audio_path.parent / f"{audio_id}.normalized.txt"

    return {"audio_path": audio_path, "text_path": text_path}

def get_normalize_text(text_path):
    text = open(text_path, "r").readline()
    return text

def main(
    save_dir,
    test_text="I go to school by bus, how about you?",
    #num_trial=10,
    tts_dataset_path="/gpfs/fs3c/nrc/dt/tst000/LibriTTS/dev-clean/"
):

    f5tts = F5TTS(model="F5TTS_Base")
    tts_dataset_path = Path(tts_dataset_path)

    reference_dicts = {
        speaker_path.name: get_audio_and_text_fname_from_path(speaker_path)  for speaker_path in tts_dataset_path.glob("*")
    }
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()

    for speaker, speaker_ref_dict in reference_dicts.items():
        ref_file = str(speaker_ref_dict["audio_path"])
        ref_text = get_normalize_text(speaker_ref_dict["text_path"])

        speaker_save_dir = (save_dir / speaker)
        
        if not speaker_save_dir.exists():
            speaker_save_dir.mkdir()
        
        meta_data = {
            "ref_text": ref_text,
            "ref_file": ref_file,
            "gen_text": test_text
        }

        json.dump(meta_data, open(speaker_save_dir / "metadata.json", 'w'))

        _ = f5tts.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=test_text,
            file_wave=str((speaker_save_dir / "no_aug.wav")),
            file_spec=str((speaker_save_dir / "no_aug_mel.png")),
            seed=None,
            sway_sampling_coef=-1.0,
            add_extra_noise_step=None
        )

        for noise_insertion_step in range(1, 30):
            _ = f5tts.infer(
                ref_file=ref_file,
                ref_text=ref_text,
                gen_text=test_text,
                file_wave=str((speaker_save_dir / f"noised_step_{noise_insertion_step}.wav")),
                file_spec=str((speaker_save_dir / f"noised_step_{noise_insertion_step}_mel.png")),
                seed=None,
                sway_sampling_coef=-1.0,
                add_extra_noise_step=noise_insertion_step
            )

def generate_trajectory(
    save_dir,
    test_text="I go to school by bus, how about you?",
    #num_trial=10,
    tts_dataset_path="/gpfs/fs3c/nrc/dt/tst000/LibriTTS/dev-clean/"
):
    f5tts = F5TTS(model="F5TTS_Base")
    tts_dataset_path = Path(tts_dataset_path)

    reference_dicts = {
        speaker_path.name: get_audio_and_text_fname_from_path(speaker_path)  for speaker_path in tts_dataset_path.glob("*")
    }
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()

    for speaker, speaker_ref_dict in reference_dicts.items():
        ref_file = str(speaker_ref_dict["audio_path"])
        ref_text = get_normalize_text(speaker_ref_dict["text_path"])

        speaker_save_dir = (save_dir / speaker)
        
        if not speaker_save_dir.exists():
            speaker_save_dir.mkdir()
        
        meta_data = {
            "ref_text": ref_text,
            "ref_file": ref_file,
            "gen_text": test_text
        }

        json.dump(meta_data, open(speaker_save_dir / "metadata.json", 'w'))
        trajectory_dir = speaker_save_dir / "forward_trajectory/"

        if not trajectory_dir.exists():
            trajectory_dir.mkdir()

        _ = f5tts.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=test_text,
            file_wave=None,
            file_spec=None,
            seed=None,
            sway_sampling_coef=-1.0,
            add_extra_noise_step=None,
            trajectory_dir=str(trajectory_dir)
        )
    
if __name__ == "__main__":
    #save_dir = "/home/tst000/projects/tst000/f5tts_examples/noise_injection"
    #main(save_dir)
    save_dir = "/home/tst000/projects/tst000/f5tts_examples/reversal"
    generate_trajectory(save_dir)
    pass
    




