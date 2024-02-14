# Whisper-training-template
Generic whisper trainer template works in windows.

https://huggingface.co/sin2piusc/whisper-med-JP

Works in windows with this version of bitsandbytes :
python.exe -m pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl

Includes code for merging peft/lora adapter and model (at end) which can then be converted with ctranslate2 and run with faster-whisper cli or placed into subtitle edit by replacing faster-whisper files .. etc. etc. (Safetensors can be converted with ctranslate2)
