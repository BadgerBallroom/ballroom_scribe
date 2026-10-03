# Ballroom Scribe

This AI pipeline generates timestamps in videos of ballroom competitions. It takes the videos as inputs and outputs a
CSV file of the starting timestamp of individual events, along with other details like the proficiency levels and types
of dance.

## Architecture

The process is orchestrated by [timestamp_pipeline.py](ballroom_scribe/timestamp_pipeline.py) in distinct stages.

```mermaid
flowchart TD
    InputVideo["Input Video (.mp4)"]
    AudioChannel["Audio Channel Selection"]

    ExtractAudio["Audio Extractor (FFmpeg)"]

    subgraph Stage1["VRAM Stage: Music Valley Detection"]
        AudioLoader["Audio Loader"]
        AudioClassifier["Audio Classifier"]
        MusicValleyDetector["Music Valley Detector"]
    end

    subgraph Stage2["VRAM Stage: Speech Transcription"]
        SpeechTranscriber["Speech-to-Text (Faster Whisper)"]
    end

    subgraph Stage3["VRAM Stage: Event Detection & Export"]
        EventDetector["Event Detector (Ollama/Qwen)"]
        EventRecorder["Event Recorder"]
    end

    InputVideo -->|"File Path"| ExtractAudio
    AudioChannel -->|"Index of Audio Channel"| ExtractAudio
    ExtractAudio -->|"File Path"| AudioLoader
    AudioLoader -->|"Unclassified Audio Chunks"| AudioClassifier
    AudioClassifier -->|"Classified Audio Chunks"| MusicValleyDetector
    MusicValleyDetector -->|"Music Valleys"| SpeechTranscriber
    SpeechTranscriber -->|"Transcript Chunks"| EventDetector
    InputVideo -->|"File Path"| EventRecorder
    EventDetector -->|"Events"| EventRecorder
    EventRecorder -->|"CSV Rows"| OutputCSV["Output CSV (.csv)"]

    CacheDir[("Cache Directory")]
    SpeechTranscriber <-->|"Speech (.csv)"| CacheDir
    AudioClassifier <-->|"Probabilities (.npy)"| CacheDir
    ExtractAudio <-->|"Mono Audio (.wav)"| CacheDir
    EventDetector <-->|"Events (.json)"| CacheDir
```

1.  **Audio Extraction** ([extract_audio.py](ballroom_scribe/utils/extract_audio.py)): Extracts the audio from the video
    file using [FFmpeg](https://ffmpeg.org/).
2.  **Audio Classification** ([audio_classifier.py](ballroom_scribe/audio_classifier.py),
    [pann_inferencer.py](ballroom_scribe/pann_inferencer.py)): Classifies audio as music and speech using
    [PANNs Inference](https://github.com/qiuqiangkong/panns_inference).
    *   VRAM is freed upon completion.
3.  **Music Valley Detection** ([music_valley_detector.py](ballroom_scribe/music_valley_detector.py)): Finds spans of
    non-music between sustained spans of music.
4.  **Speech Transcription** ([speech_transcriber.py](ballroom_scribe/speech_transcriber.py)): Transcribes the audio
    that was classified as speech into timestamped words using
    [Faster Whisper](https://github.com/SYSTRAN/faster-whisper).
    *   VRAM is freed upon completion.
5.  **Event Detection** ([event_detector.py](ballroom_scribe/event_detector.py)): Parses the emcee's announcements for
    each event's proficiency level, dance type, etc. using **Qwen 2.5 7B** via [Ollama](https://ollama.com/). Finds the
    starting timestamps of the emcee's announcements.
    *   Ollama handles freeing VRAM.
6.  **Event Recording** ([event_recorder.py](ballroom_scribe/event_recorder.py)): Writes event details to a CSV file.

Each stage that writes to the cache directory skips its work if its output file already exists, so rerunning an
interrupted run resumes where it left off. To make this safe, cache files are written atomically
([atomic_path.py](ballroom_scribe/utils/atomic_path.py)): each file is first written under a temporary name (e.g.,
`Audio-Channel-0.partial.wav`) and renamed to its final name only once it is complete.

## Requirements

Although more hardware may be supported in the future, Scribe currently requires an NVIDIA GPU with CUDA support and a
minimum of **8 GB of VRAM**. These models are used:

*   **PANNs Inference** uses ~6-8 GB of VRAM when classifying audio.
    *   You can reduce this usage with the `--pann-chunk-minutes` flag.
*   **Faster Whisper (`large-v3`)** requires ~3–4 GB of VRAM.
*   **Qwen 2.5 7B (`qwen2.5:7b-instruct-q6_K`)** requires ~6-7 GB of VRAM.
    *   You can use a different model via the `--ollama-model` flag.

## Setup

Follow these steps to set up and run Scribe on your machine:

### Install System Dependencies

Install Python, FFmpeg (for extracting audio from video), and Ollama (for running the local LLM). On Windows, you can
install all three using `winget`:

```powershell
# Run as Administrator.
winget install -e --id Gyan.FFmpeg
winget install -e --id Python.Python.3.14
winget install -e --id Ollama.Ollama
```

After installation completes, restart your terminal so the new command line tools are added to your system `PATH`.

### Download the Language Model (Ollama)

Download the model:

```powershell
ollama pull qwen2.5:7b-instruct-q6_K
```

### Install Python Dependencies

Install the required packages:

```powershell
pip install -r requirements.txt
```

### Workaround for PANNs Inference

The PANNs Inference library, which is used to discern between dance music and the emcee speaking, will try to download
[a CSV of class labels](https://github.com/qiuqiangkong/panns_inference/blob/f673f604ec6f4805a61c5b3be087e24776ec5fda/panns_inference/config.py#L11-L13)
and
[a PyTorch model checkpoint](https://github.com/qiuqiangkong/panns_inference/blob/f673f604ec6f4805a61c5b3be087e24776ec5fda/panns_inference/inference.py#L34-L37).
Unfortunately, it does it with `wget`, which most Windows machines do not have. You can download it with:

```powershell
$dir = "$HOME\panns_data"
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir }
Invoke-WebRequest -Uri "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv" -OutFile "$dir\class_labels_indices.csv"
Invoke-WebRequest -Uri "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1" -OutFile "$dir\Cnn14_DecisionLevelMax.pth"
```

## Usage

Invoke the main script, [find_timestamps.py](find_timestamps.py), from a PowerShell terminal. The basic usage is:

```powershell
python find_timestamps.py C:\path\to\video.mp4 {0 or 1} --output-csv 'C:\path\to\output.csv'
```

`{0 or 1}` refers to the audio channel to use. 0 usually means left, and 1 usually means right.

You can find more options with:

```powershell
python find_timestamps.py --help
```

The first time that you run the script, you may see the following message:

> Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits
> and faster downloads.

This is normal; it appears when a model is downloaded from HuggingFace. After the first time, the models will be cached
on your local system, and you should not see the message again.

## Tuning PANN Thresholds

The `--pann-threshold-speech` and `--pann-threshold-music` thresholds apply to the probabilities that the audio neural
network has outputted that a given frame of audio contains those types of sounds. They take precedence in that order;
for example, if the probabilities of both speech and music for a given frame exceed their thresholds, the frame will be
labeled as speech.

You can use the [tune_pann_thresholds.py](tune_pann_thresholds.py) script to find the best thresholds for your specific
audio. You must install [Streamlit](https://streamlit.io/):

```powershell
pip install streamlit
```

Then, you can launch the tool:

```powershell
streamlit run tune_pann_thresholds.py -- C:\path\to\video.mp4 {0 or 1}
```
