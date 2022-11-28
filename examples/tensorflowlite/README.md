# Converting to TensorFlowLite

To convert pretrained text2mel and vocoder models to TensorFlowLite, you can run the script

```py
python TensorFlowTTS/examples/tensorflowlite/convert_tflite.py \
    --text2mel_path="bookbot/lightspeech-mfa-id-v3" \
    --text2mel_savename="lightspeech_quant.tflite" \
    --vocoder_path="bookbot/mb-melgan-hifi-postnets-id-v10" \
    --vocoder_savename="mbmelgan.tflite" \
    --use_auth_token
```

[Inference code](./inference.py) is also provided.