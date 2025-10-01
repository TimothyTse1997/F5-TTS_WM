import random
import sys
from importlib.resources import files

import soundfile as sf
import tqdm
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    transcribe,
)
from f5_tts.model.utils import seed_everything

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.cm import viridis
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

FFT_WATER_MARK_LINES = [5, 10, 15, 20, 25]

def plot_kde(A, B, fname):
    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot A (fixed color and label)
    sns.kdeplot(A, label='original distribution', color='red', linewidth=2)

    # Normalize the keys for colormap
    keys = sorted(B.keys())
    norm = Normalize(vmin=min(keys), vmax=max(keys))
    cmap = plt.get_cmap('viridis')

    # Plot each distribution in B with a color from the continuous colormap
    for key in keys:
        color = cmap(norm(key))
        sns.kdeplot(B[key], label=str(key), color=color, linewidth=2, alpha=0.8)

    # Create colorbar as legend for B's keys
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required for colorbar
    cbar = plt.colorbar(sm, pad=0.01, ax=plt.gca())
    cbar.set_label('EC loops')

    plt.title('original distribution vs reverse distribution')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend(title='Legend')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


class F5TTS:
    def __init__(
        self,
        model="F5TTS_v1_Base",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_local_path=None,
        device=None,
        hf_cache_dir=None,
    ):
        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch

        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.target_sample_rate = model_cfg.model.mel_spec.target_sample_rate

        self.ode_method = ode_method
        self.use_ema = use_ema

        if device is not None:
            self.device = device
        else:
            import torch

            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "xpu"
                if torch.xpu.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        # Load models
        self.vocoder = load_vocoder(
            self.mel_spec_type, vocoder_local_path is not None, vocoder_local_path, self.device, hf_cache_dir
        )

        repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

        # override for previous models
        if model == "F5TTS_Base":
            if self.mel_spec_type == "vocos":
                ckpt_step = 1200000
            elif self.mel_spec_type == "bigvgan":
                model = "F5TTS_Base_bigvgan"
                ckpt_type = "pt"
        elif model == "E2TTS_Base":
            repo_name = "E2-TTS"
            ckpt_step = 1200000

        if not ckpt_file:
            ckpt_file = str(
                cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}", cache_dir=hf_cache_dir)
            )
        self.ema_model = load_model(
            model_cls, model_arc, ckpt_file, self.mel_spec_type, vocab_file, self.ode_method, self.use_ema, self.device
        )

    def transcribe(self, ref_audio, language=None):
        return transcribe(ref_audio, language)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spec, file_spec):
        save_spectrogram(spec, file_spec)

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spec=None,
        seed=None,
    ):
        if seed is None:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)

        wav, sr, spec = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        if file_spec is not None:
            self.export_spectrogram(spec, file_spec)
<<<<<<< HEAD
        
        if dir_traj is not None:

            dir_traj = Path(dir_traj)
            if not dir_traj.exists():
                dir_traj.mkdir()
            for batch_id, trajectory in enumerate(trajectories):
                batch_dir = dir_traj / str(batch_id)
                if not batch_dir.exists(): batch_dir.mkdir()
                for i, t in enumerate(trajectory):
                    torch.save(t.cpu(), batch_dir / f"{i}.pt")
                    self.export_spectrogram(t[0].cpu(), batch_dir / f"{i}_mel.png")

        return wav, sr, spec, trajectories
    
    def inverse(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        dir_traj=None,
        seed=None,
        gen_audio_mel=None,
        gen_audio=None,
        forward_backward_step=None
    ):
        if seed is None:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)
        ref_audio, sr = torchaudio.load(ref_file)
        if gen_audio_mel is not None:
            gen_audio_mel = torch.load(gen_audio_mel)
            #gen_audio_mel = gen_audio_mel.permute(0, 2, 1)
        if gen_audio is not None:
            gen_audio, source_sample_rate = torchaudio.load(gen_audio)
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                gen_audio = resampler(gen_audio)
        else: source_sample_rate = None

        trajectory, masked_trajectory = single_inverse_batch_process(
            (ref_audio, sr),
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
            gen_audio_mel=gen_audio_mel,
            gen_audio=(gen_audio, source_sample_rate),
            forward_backward_step=forward_backward_step
        )

        if dir_traj is not None:
            dir_traj = Path(dir_traj)
            if not dir_traj.exists():
                dir_traj.mkdir()
            for i, t in enumerate(trajectory):
                torch.save(t.cpu(), dir_traj / f"{i}.pt")
                self.export_spectrogram(t[0].cpu(), dir_traj / f"{i}_mel.png")
                #self.export_spectrogram(masked_trajectory[i][0].cpu(), dir_traj / f"{i}_masked_mel.png")
                #mask_traj = masked_trajectory[i]
                #full_shape = mask_traj.shape[1]
                #non_watermark_dimensions = [i for i in range(full_shape) if (i not in FFT_WATER_MARK_LINES)]

                #watermark_dimensions = {
                #    i: mask_traj[:, i, :].squeeze(0).cpu().numpy() for i in FFT_WATER_MARK_LINES
                #}
                #all_values = mask_traj[:, non_watermark_dimensions, :].flatten().cpu().numpy()

                #plot_kde(all_values, watermark_dimensions, dir_traj / f"{i}_distribution.png")
=======
>>>>>>> parent of 9bf9990 (clean inversion code, wip using audio file as input (so we can attack))

        return wav, sr, spec


if __name__ == "__main__":
    f5tts = F5TTS()
<<<<<<< HEAD
    # for i in range(10):
    #     wav, sr, spec, _ = f5tts.infer(
    #         ref_file="/home/tst000/projects/tst000/LibriTTS/dev-clean/1272/128104/1272_128104_000006_000008.wav",
    #         ref_text="On the whole, the book will not do.",
    #         gen_text="""I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences.""",
    #         file_wave=f"./api_out_{i}.wav",
    #         file_spec=f"./api_out_{i}.png", #str(files("f5_tts").joinpath("../../tests/api_out.png")),
    #         seed=None,
    #         #dir_traj="./api_traj/",
    #         #cfg_strength=0,
    #     )
    for forward_backward_step in range(2, 33, 4):
        _ = f5tts.inverse(
            ref_file="/home/tst000/projects/tst000/LibriTTS/dev-clean/1272/128104/1272_128104_000006_000008.wav",
            ref_text="On the whole, the book will not do.",
            gen_text="""I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences.""",
            seed=None,
            dir_traj=f"./api_traj_forward_backward_step_{forward_backward_step}/",
            gen_audio_mel="/home/tst000/projects/tst000/F5-TTS_WM/api_traj/0/31.pt",
            forward_backward_step=forward_backward_step
            #gen_audio="./api_out.wav"
            #cfg_strength=0,
        )
=======

    wav, sr, spec = f5tts.infer(
        ref_file=str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
        ref_text="some call me nature, others call me mother nature.",
        gen_text="""I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences.""",
        file_wave=str(files("f5_tts").joinpath("../../tests/api_out.wav")),
        file_spec=str(files("f5_tts").joinpath("../../tests/api_out.png")),
        seed=None,
    )
>>>>>>> parent of 9bf9990 (clean inversion code, wip using audio file as input (so we can attack))

    print("seed :", f5tts.seed)
