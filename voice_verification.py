from speechbrain.pretrained import (
    SepformerSeparation as separator,
    SpectralMaskEnhancement as enhancement,
    SpeakerRecognition as verification,
)
import sys
import time
import torchaudio
import torch


CHUNK_LENGTH = 10  # second
SAMPLE_RATE = 8000  # model.hparams.sample_rate
TOTAL_VOICES = 2
PATH = "wav/Thy_Trim.wav"  # TODO replace
GROUNDTRUTH_PATH = "vocal_thy.wav"  # TODO replace

model_separator = separator.from_hparams(
    source="speechbrain/sepformer-wsj02mix", 
    savedir='sepformer-wsj02mix',
    )

model_enhancement_metricgan = enhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="metricGan",
    )

model_verification = verification.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="Verification",
    )


def read_audio(path):
    batch, fs_file = torchaudio.load(path)

    batch = batch.to(model_separator.device)  
    fs_model = model_separator.hparams.sample_rate

    # resample the data if needed
    if fs_file != fs_model:
        print(
            "Resampling the audio from {} Hz to {} Hz".format(
                fs_file, fs_model
            )
        )
        tf = torchaudio.transforms.Resample(
            orig_freq=fs_file, new_freq=fs_model
        ).to(model_separator.device)
        batch = batch.mean(dim=0, keepdim=True)
        batch = tf(batch)

    return batch


def count_speaking_time(batch, threshold = 10e-4):
    count = 0
    for x in batch[0]:
        if(abs(x) > threshold):
            count += 1
    return count / SAMPLE_RATE


def separate_batch(batch): 
    est_sources = model_separator.separate_batch(batch)
    est_sources = (
        est_sources / est_sources.abs().max(dim=1, keepdim=True)[0]
    )
    return est_sources


def enhance_metricgan(batch):
    model = model_enhancement_metricgan
    #batch = batch.unsqueeze(0)
    enhanced = model.enhance_batch(batch, lengths=torch.tensor([1.]))
    return enhanced.cpu()


def main():
    start_time = time.time()

    # Reading audio file
    batch = read_audio(PATH)

    # Split audio into n seconds (CHUNK_LENGTH)
    chunk_len = int(SAMPLE_RATE * CHUNK_LENGTH ) # frames
    batches = torch.split(batch, chunk_len, 1)

    # Separate voices in each part

    groundtruth = read_audio(GROUNDTRUTH_PATH)
    voice = torch.tensor([])
    for batch in batches:
        est_sources = separate_batch(batch) 
        # concat the voices
        for i in range(TOTAL_VOICES):
            batch_enhance = est_sources[:, :, i].detach().cpu()
            batch_enhance = enhance_metricgan(batch_enhance)
            # print(type(batch_enhance))
            _, decision = model_verification.verify_batch(batch_enhance, groundtruth)
            if decision:
                voice = torch.cat((voice, batch_enhance), 1)
        
        """
        print("saving voices...")
        torchaudio.save("source1hat_" + str(part) + ".wav", est_sources[:, :, 0].detach().cpu(), SAMPLE_RATE)
        torchaudio.save("source2hat_" + str(part) + ".wav", est_sources[:, :, 1].detach().cpu(), SAMPLE_RATE)
        

   
     for i in range(TOTAL_VOICES):
         # Enhance voice
        voices[i] = enhance_metricgan(voices[i])

         #torchaudio.save('thy_1.wav', voices[i], 16000) # TODO: change sample rate 8000 -> 16000?
        
        # Load groundtruth voice
        groundtruth = model_verification.load_audio(GROUNDTRUTH_PATH).unsqueeze(0)
     _, decision = model_verification.verify_batch(voices[i], groundtruth)

         if decision == True:
     print(voice)
    """
    print(f"teacher voice id: {i} -> talk {count_speaking_time(voice)} seconds")

    print(f"processing time: {time.time() - start_time} sec")


if __name__ == "__main__":
    sys.exit(main() or 0)