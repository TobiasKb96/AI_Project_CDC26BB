# AI Project 2025  

Authors:  Tobias Kaltenbrunner
          Anna Zierlinger
          Stefanie Biber
          
## Documentation

https://github.com/githubharald/SimpleHTR

### Motivation/State of the Art


#### Motivation
    The motivation for our Project is to find out existing methods of Handrwritten text recognision and apply them to a real World use case.
    The final longterm goal would be to find specially marked TODOs in a PDF,translate them to text and then port them to calendar entries
    This project will focus more on the second step of translating handwritten text to computational txt files. 
    
#### State of the Art




### Methods

1. Preprocess images with `src/preprocessor.py`.
2. Load data splits using `src/dataloader_iam.py`.
3. Train the CNN+LSTM network defined in `src/model.py` via `src/main.py`.
4. Evaluate with different CTC decoders (best path, beam search, word beam search).

#### AI Algorithms and Architecture

The core neural network is implemented with TensorFlow and consists of convolutional layers followed by bidirectional LSTM layers. 
The network is trained using Connectionist Temporal Classification (CTC) loss. 
A variety of decoders are available including best path, beam search and word beam search.

---
CRNN (CNN + RNN + CTC) 
    Pros : Lightweight, less accurate
    Outdated but easier to train

#### Hyperparameters

Training is typically performed with a batch size of 100 and early stopping after 25 epochs without improvement. 
The images are resized to a height of 32 pixels while the width depends on whether single words or text lines are processed.

### Results

The pretrained model from the original repository recognizes around three quarters of the words from the IAM validation set correctly with a character error rate of roughly 10%. 
Using the same configuration we achieve comparable results on our experiments.

#### Performance Comparison

When comparing the different decoders, word beam search tends to improve word accuracy when a dictionary is available, 
while best path decoding is faster but slightly less accurate.

#### Discussion

Our results show that even a relatively small CNN+LSTM architecture can achieve usable accuracy for HTR. 
Further improvements could be gained by augmenting the training data or experimenting with deeper networks.

## Implementation

### Data pre-processing (plot some data, data cleaning, data augmentation, divide data in train/validation/test sets)

The `Preprocessor` class handles resizing and augmentation of the images. 
It can optionally simulate text lines by stitching several word images together and performs photometric and geometric distortions during training. 
The IAM dataset is split into 95% training data and 5% validation data by the `DataLoaderIAM` loader.

### Model selection and Training (Use at least 2 Models and/or different hyperparameters (network size, number of nodes etc.)

Validation runs after every epoch using the same script with `--mode validate`. 
Character error rate and word accuracy are reported and written to `model/summary.json` for later analysis.

### Model evaluation (Overfitting/Underfitting, use Regularization if needed)

For single image inference (`--mode infer`) a GradCAM heatmap can be generated with the `--gradcam` option. 
The resulting `gradcam.png` highlights the areas that contributed most strongly to the predicted characters.

### Explanation i.e. GradCam (which input pixels were most relevant for the output decision)


