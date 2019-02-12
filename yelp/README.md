# ARAE for language style transfer on Yelp

- Training: `python main.py --mode=train [--load_models]`
- CLI for style transfer and interpolation: `python main.py (--mode=transfer|--mode=interpolate) --load_models`
- HTTP API for style transfer and interpolation: `python main.py --mode=serve --load_models`. 
    
## Requirements
- Python 3.6
- PyTorch 1.0.1, numpy, nltk, flask

You made need to download the Punkt tokenizer models by running `python -c 'import nltk; nltk.download("punkt")'`.
