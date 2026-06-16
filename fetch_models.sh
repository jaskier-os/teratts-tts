#!/usr/bin/env bash
#
# Pre-download the TeraTTS model and the RUAccent accentizer data.
#
# This is OPTIONAL: server.py downloads both on first startup automatically.
# Run this if you want to pre-cache them (e.g. for offline deployment or to
# warm a Docker layer) before starting the server.
#
# The model named by TTS_MODEL (default: TeraTTS/glados2-g2p-vits) is a standard
# pretrained model published on the HuggingFace Hub. It is NOT bundled in this
# repo because of its size (~200 MB). The TeraTTS Python library fetches it.
#
set -e
cd "$(dirname "$0")"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

MODEL="${TTS_MODEL:-TeraTTS/glados2-g2p-vits}"

echo "[teratts-tts] Pre-downloading TeraTTS model: ${MODEL}"
python -c "from TeraTTS import TTS; TTS('${MODEL}', add_time_to_end=0.5, tokenizer_load_dict=True)"

echo "[teratts-tts] Pre-downloading RUAccent accentizer data"
python -c "from ruaccent import RUAccent; a = RUAccent(); a.load(omograph_model_size='turbo', use_dictionary=True)"

echo "[teratts-tts] Pre-downloading g2p_en NLTK data"
python -c "import nltk; [nltk.download(r, quiet=True) for r in ['averaged_perceptron_tagger','averaged_perceptron_tagger_eng','cmudict','punkt','punkt_tab']]"
python -c "from g2p_en import G2p; G2p()('warmup test')"

echo "[teratts-tts] Done."
