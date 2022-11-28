import tensorflow as tf
from tensorflow_tts.inference import TFAutoModel
import argparse


def convert_text2mel_tflite(
    model_path: str, save_name: str, use_auth_token: bool = False
) -> float:
    """Convert text2mel model to TFLite.

    Args:
        model_path (str): Pretrained model checkpoint in HuggingFace Hub.
        save_name (str): TFLite file savename.
        use_auth_token (bool, optional): Use HF Hub Token. Defaults to False.

    Returns:
        float: Model size in Megabytes.
    """
    # load pretrained model
    model = TFAutoModel.from_pretrained(
        model_path, enable_tflite_convertible=True, use_auth_token=use_auth_token
    )

    # setup model concrete function
    concrete_function = model.inference_tflite.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])

    # specify optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # quantize
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    # convert and save model to TensorFlowLite
    tflite_model = converter.convert()
    with open(save_name, "wb") as f:
        f.write(tflite_model)

    size = len(tflite_model) / 1024 / 1024.0
    return size


def convert_vocoder_tflite(
    model_path: str, save_name: str, use_auth_token: bool = False
) -> float:
    """Convert vocoder model to TFLite.

    Args:
        model_path (str): Pretrained model checkpoint in HuggingFace Hub.
        save_name (str): TFLite file savename.
        use_auth_token (bool, optional): Use HF Hub Token. Defaults to False.

    Returns:
        float: Model size in Megabytes.
    """
    # load pretrained model
    model = TFAutoModel.from_pretrained(model_path, use_auth_token=use_auth_token)

    # setup model concrete function
    concrete_function = model.inference_tflite.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])

    # specify optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    converter.target_spec.supported_types = [tf.float16]  # fp16 ops

    # convert and save model to TensorFlowLite
    tflite_model = converter.convert()
    with open(save_name, "wb") as f:
        f.write(tflite_model)

    size = len(tflite_model) / 1024 / 1024.0
    return size


def main():
    parser = argparse.ArgumentParser(description="Convert text2mel and vocoder.")

    parser.add_argument(
        "--text2mel_path", type=str, required=True, help="Path to text2mel model."
    )
    parser.add_argument(
        "--text2mel_savename", type=str, required=True, help="Text2mel file savename."
    )
    parser.add_argument(
        "--vocoder_path", type=str, required=True, help="Path to vocoder model."
    )
    parser.add_argument(
        "--vocoder_savename", type=str, required=True, help="Vocoder file savename."
    )
    parser.add_argument(
        "--use_auth_token", action="store_true", default=False, help="Use HF Hub Token."
    )

    args = parser.parse_args()

    text2mel = convert_text2mel_tflite(
        model_path=args.text2mel_path,
        save_name=args.text2mel_savename,
        use_auth_token=args.use_auth_token,
    )

    vocoder = convert_vocoder_tflite(
        model_path=args.vocoder_path,
        save_name=args.vocoder_savename,
        use_auth_token=args.use_auth_token,
    )

    print(f"Text2mel: {text2mel} MBs\nVocoder: {vocoder} MBs")


if __name__ == "__main__":
    main()
