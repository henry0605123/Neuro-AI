---
license: cc-by-nc-sa-3.0
viewer: false
task_categories:
- automatic-speech-recognition
- audio-classification
- text-classification
tags:
- audio
- child speech
- speaker diarization
- asr
- speech acts
- dpics
language:
- en
language_bcp47:
- en-US
annotations_creators:
- expert-generated
source_datasets:
- CHILDES
extra_gated_prompt: >-
  Before you are granted access to the Playlogue dataset, you must confirm that
  you have read and agree to the CHILDES ground rules for data use and sharing
  at https://talkbank.org/share/rules.html. These rules apply to all data
  included in Playlogue.
extra_gated_fields:
  Full Name: text
  Institution: text
  Country: country
  I have read and agree to the ground rules at the link above: checkbox
pretty_name: Playlogue
---
# Playlogue: Dataset and Benchmarks for Analyzing Adult-Child Conversations During Play

Playlogue is a first-of-its-kind dataset of naturalistic adult-child conversations with transcripts, speaker information, and speech acts. It is designed to develop and evaluate audio and language models on child-centered speech involving preschool-aged children. For more details, please refer to [our paper](https://doi.org/10.1145/3699775).

## Dataset Details

### Dataset Description

Playlogue is a *curated* dataset containing over 33 hours of adult-child interaction audio, sourced from three play-based corpora and one non-play corpus in the [CHILDES system](https://childes.talkbank.org/). We apply extensive manual filtering and automated forced-alignment using the [NVIDIA NeMo Forced Aligner](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html) to enable researchers to use the dataset for applications such as speaker diarization and automatic speech recognition.

We also annotate a subset of this curated dataset using the clinically validated [Dyadic Parent-Child Interaction Coding System (DPICS)](https://www.researchgate.net/profile/Sheila-Eyberg/publication/242667283_Manual_for_the_Dyadic_Parent-Child_Interaction_Coding_System/links/5429fac80cf277d58e87038f/Manual-for-the-Dyadic-Parent-Child-Interaction-Coding-System.pdf), which is widely used in parent-child interaction therapy to assess the quality of parent-child communication.
Playlogue contains 4773 labeled parent utterances and 3895 labeled child utterances with full conversation audio and context. 

### Dataset Sources

The audio data in Playlogue is sourced from [CHILDES](https://childes.talkbank.org/) and must be manually downloaded from the original source. The DPICS labels were generated through manual annotation by trained graduate students (details in the [paper](https://doi.org/10.1145/3699775)). Please read the [ground rules for data sharing and usage](https://talkbank.org/share/rules.html) and ensure you abide by these when using any data from CHILDES and/or Playlogue.

## Uses

Playlogue has been designed to enable researchers working on child-centered speech and language applications to develop and evaluate machine learning models for analyzing various levels of adult-child interactions during play. These include low-level vocal behaviors (e.g., tone, pitch), diarization-based metrics such as turn durations and overlaps, ASR-based measures like word complexity and diversity, and higher-level interaction markers such as DPICS codes. 
The availability of full conversation audio, timestamped and speaker-attributed transcripts, and DPICS codes enables researchers to study conversation dynamics in addition to sentence-level measures.

## Benchmark Tasks

We use Playlogue to evaluate the performance of state-of-the-art models on child-centered speech. We consider three benchmark tasks:

- Adult-child Speaker Diarization (Diarization Error Rate)
  - NeMo Neural Diarizer: 67.6%
  - NeMo Clustering Diarizer: 64.5%
  - PyAnnote Powerset Diarizer: 53.4%
    
- Automatic Speech Recognition (Word Error Rate)
  - NVIDIA Canary-1B: 51.22%
  - OpenAI Whisper-medium-en: 49.44%
  - OpenAI Whisper-large-v3: 40.84%
    
- DPICS Labeling (Accuracy)
  - OpenAI GPT-4: 77.70% for 8-class parent DPICS, 87.35% for 4-class child DPICS

As evidenced above, the performance of state-of-the-art models deteriorates significantly from their reported performance on standard benchmark datasets containing adult speech. Researchers could utilize the Playlogue dataset to fill this gap by training or fine-tuning models that are better suited to child-centered conversational settings.
For more results and details, including the performance of models fine-tuned on Playlogue as well as generalizability to other contexts, [see our paper](https://doi.org/10.1145/3699775).

## Dataset Structure

```txt
data
└── audio:
  └── README.md: Contains instructions for downloading audio from CHILDES.
└── dpics_labels: Contains DPICS labels for each parent and child utterance in all audio clips from `ew_42pc`. Clips that were labeled by multiple coders contain their independent ratings as well as the final consolidated rating in the `final_rating` column. Clips labeled by a single coder contain a single rating in the column named `final_rating`.
└── speaker_diarization: Contains ground-truth RTTM files for adult-child diarization. Generated based on the ground-truth transcripts from CHILDES and the segment-level alignment information in `data/speech_alignment/segments`.
└── speech_alignment: Contains the CTM outputs of the NeMo forced aligner at the segment-, token-, and word-level.
  └── segments
  └── tokens
  └── words
metadata
└── clip_timings.csv: Contains the curated start and end times to trim each audio clip from CHILDES for inclusion in Playlogue.
└── splits.csv: Contains metadata on whether each audio clip belongs to the train, val, or test split. Splits contain mutually exclusive participants.
```

Explanation of file formats: [RTTM](https://github.com/nryant/dscore?tab=readme-ov-file#rttm) for speaker diarization, [CTM](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html#output-ctm-file-format) for speech alignment.

Note that all data in `speaker_diarization`, `speech_alignment`, and `dpics_labels` correspond to the *trimmed* audio clips generated using the start and end timestamps in `metadata/clip_timings.csv` and NOT the *original* audio from CHILDES.

## Limitations

Playlogue only focuses on English-speaking children in North America with typically developing speech. As such, the analysis and results using this dataset only apply to this target demographic and should not be assumed to generalize to other contexts. We encourage future researchers to curate similar datasets that span more diverse subgroups within the early childhood population.

## Citation

If you use Playlogue in your work, please cite the [original publication](https://doi.org/10.1145/3699775):

```bibtex
@article{kalanadhabhatta2024playlogue,
  title={Playlogue: Dataset and Benchmarks for Analyzing Adult-Child Conversations During Play},
  author={Kalanadhabhatta, Manasa and Rastikerdar, Mohammad Mehdi and Rahman, Tauhidur and Grabell, Adam S. and Ganesan, Deepak},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={8},
  number={4},
  year={2024},
  publisher={ACM New York, NY, USA},
  url={https://doi.org/10.1145/3699775}
}
```