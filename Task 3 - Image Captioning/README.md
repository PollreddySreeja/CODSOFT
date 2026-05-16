# Image Captioning using Deep Learning

**CodSoft AI Internship — Task 3**
> **CodSoft AI Internship | Sreeja Pollreddy | BY25RY287818**

This project builds an image captioning system that takes an image as input and generates a natural language description of what's happening in it. It combines computer vision (CNN) for understanding the image with NLP (RNN) for generating the text.

## How it Works

The model has two main parts:

1. **Encoder (ResNet-50)**: A pre-trained CNN that looks at the image and extracts meaningful features. Instead of using the final classification layer, I removed the last two layers so we get spatial feature maps (7x7 grid of 2048-dimensional vectors). This way the decoder can "attend" to different parts of the image.

2. **Decoder (LSTM + Attention)**: An LSTM network that generates the caption one word at a time. At each step, the Bahdanau attention mechanism lets the decoder focus on relevant regions of the image. For example, when generating the word "dog", it attends to the part of the image where the dog is.

```
Image → ResNet-50 → Spatial Features (7×7×2048) → Attention + LSTM → Words
```

## Project Structure

```
├── src/
│   ├── encoder.py         - ResNet-50 feature extractor
│   ├── attention.py       - Bahdanau attention module
│   ├── decoder.py         - LSTM decoder with attention
│   ├── caption_model.py   - combines encoder + decoder
│   ├── vocabulary.py      - word <-> index mapping
│   ├── dataset.py         - Flickr8k dataloader
│   └── beam_search.py     - beam search decoding
├── train.py               - training script
├── evaluate.py            - BLEU score evaluation
├── app.py                 - Streamlit web app
├── config.py              - hyperparameters
└── requirements.txt
```

## Setup & Run

```bash
# install dependencies
pip install -r requirements.txt

# run the web app (uses BLIP pre-trained model, works out of the box)
streamlit run app.py

# to train custom model on Flickr8k (optional, needs GPU ideally)
python train.py --epochs 25 --batch_size 32
```

The Streamlit app supports two modes:
- **BLIP model** — works immediately without any training, good for quick demo
- **Custom model** — uses the ResNet+LSTM architecture I built, needs training first

## Key Features

- Custom encoder-decoder architecture with attention (not just a library wrapper)
- Bahdanau attention with gating mechanism
- Beam search with length penalty for better captions
- Attention heatmap visualization — shows which parts of the image the model looks at for each word
- BLEU-1 to BLEU-4 evaluation metrics
- Interactive Streamlit UI for testing

## Some Implementation Details

- Used **teacher forcing** during training (feed ground truth word instead of predicted word with 75% probability)
- **Doubly stochastic attention regularization** — adds a penalty so that the model attends to every part of the image roughly equally over time. Without this the model tends to fixate on the same region
- Separate learning rates for encoder (1e-4) and decoder (4e-4) since the encoder is pre-trained
- Gradient clipping at 5.0 to avoid exploding gradients in LSTM

## Dataset

Trained on [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) — 8,091 images with 5 captions each. The training script downloads it automatically if you have Kaggle API set up.

## References

- Xu et al., "Show, Attend and Tell" (2015) — the main paper this is based on
- He et al., "Deep Residual Learning" (2016) — ResNet architecture
- Li et al., "BLIP" (2022) — used for the pre-trained comparison model

## Tech Stack

- PyTorch, torchvision
- HuggingFace Transformers (for BLIP)
- Streamlit
- NLTK (for BLEU scores)
- Matplotlib (for attention visualizations)
