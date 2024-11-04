# AutoSort
**Multimodal deep learning for real-time stable decoding of month-long neural activities from the same cells**

<p align="center">
  <img src="/img/figure1.png" >
</p>

AutoSort is designed to tackle two significant challenges in long-term stable recording. 
- First, it efficiently aligns neurons over the course of long-term recordings to ensure consistent tracking of the same neurons each day. 
- Second, it accurately sorts spikes while maintaining the precision throughout the recordings, ensuring that the performance achieved at the first of the recordings is sustained throughout the later days.


<p align="center">
  <img src="/img/figure2.png" >
</p>

AutoSort innovatively leverages multimodal features as inputs. We extract single-channel waveform, multi-channel waveform, and the inferred spatial location for any potential spike that exceeds a certain threshold on any particular channel to be sorted.

For more details, please check out our publication.


## Manuscript code and data
### Reproducibity
Code used in the study is uploaded to figshare and will be public with the manuscript publication. 

## System Requirements
### Hardware requirements
`AutoSort` package requires a standard computer with GPU to support the in-memory operations.

### Software requirements
#### OS Requirements
This package is supported for *Linux*. The package has been tested on the following system:
+ Linux: Ubuntu 20.04

#### Python Dependencies
`AutoSort` mainly depends on the Python scientific stack.

```
spikeinterface
scipy
scikit-learn
pandas
pytorch
seaborn
```

## Installation
```
pip install autosort-neuron==0.0.1.4
```


## Tutorial
- Read our [tutorials](https://autosort.readthedocs.io/en/latest/index.html) with [provided datasets](https://drive.google.com/drive/folders/18kMTfZKlN7vw04xEVZ5SC2fDD-N9XYxr?usp=drive_link) .


## Citation

If you find AutoSort useful for your work, please cite our paper: 

> Multimodal deep learning for real-time stable decoding of month-long neural activities from the same cells.
Yichun He#, Arnau Marin-Llobet#, Hao Sheng, Ren Liu, Jia Liu*. Preprint at bioRxiv ? (2024).
