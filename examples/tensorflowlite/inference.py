import tensorflow as tf
from tensorflow_tts.inference import AutoProcessor
from typing import List, Tuple
import soundfile as sf


def tokenize(text: str, processor: AutoProcessor) -> List[int]:
    """Tokenize text to input ids.

    Args:
        text (str): Input text to tokenize.
        processor (AutoProcessor): Processor for tokenization.

    Returns:
        List[int]: List of input (token) ids.
    """
    return processor.text_to_sequence(text)


def prepare_input(
    input_ids: List[str], speaker: int
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Prepares input for LightSpeech TFLite inference.

    Args:
        input_ids (List[str]): Phoneme input ids according to processor.
        speaker (int): Speaker ID.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            Tuple of tensors consisting of:
                1. Input IDs
                2. Speaker ID
                3. Speed Ratio
                4. Pitch Ratio
                5. Energy Ratio
    """
    input_ids = tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0)
    return (
        input_ids,
        tf.convert_to_tensor([speaker], tf.int32),
        tf.convert_to_tensor([1.0], dtype=tf.float32),
        tf.convert_to_tensor([1.0], dtype=tf.float32),
        tf.convert_to_tensor([1.0], dtype=tf.float32),
    )


def ls_infer(
    input_ids: List[str], speaker: int, lightspeech_path: str
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Performs LightSpeech inference.

    Args:
        input_ids (List[str]): Phoneme input ids according to processor.
        speaker (int): Speaker ID.
        lightspeech_path (str): Path to LightSpeech weights.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            Tuple of tensors consisting of:
                1. Mel-spectrogram output
                2. Durations array
    """
    # load model to Interpreter
    lightspeech = tf.lite.Interpreter(model_path=lightspeech_path)
    input_details = lightspeech.get_input_details()
    output_details = lightspeech.get_output_details()

    # resize input tensors according to actual shape
    lightspeech.resize_tensor_input(input_details[0]["index"], [1, len(input_ids)])
    lightspeech.resize_tensor_input(input_details[1]["index"], [1])
    lightspeech.resize_tensor_input(input_details[2]["index"], [1])
    lightspeech.resize_tensor_input(input_details[3]["index"], [1])
    lightspeech.resize_tensor_input(input_details[4]["index"], [1])

    # allocate tensors
    lightspeech.allocate_tensors()

    input_data = prepare_input(input_ids, speaker)

    # set input tensors
    for i, detail in enumerate(input_details):
        lightspeech.set_tensor(detail["index"], input_data[i])

    # invoke interpreter
    lightspeech.invoke()

    # return outputs
    return (
        lightspeech.get_tensor(output_details[0]["index"]),
        lightspeech.get_tensor(output_details[1]["index"]),
    )


def melgan_infer(melspectrogram: tf.Tensor, mb_melgan_path: str) -> tf.Tensor:
    """Performs MB-MelGAN inference.

    Args:
        melspectrogram (tf.Tensor): Mel-spectrogram to synthesize.
        mb_melgan_path (str): Path to MB-MelGAN weights.

    Returns:
        tf.Tensor: Synthesized audio output tensor.
    """
    # load model to Interpreter
    mb_melgan = tf.lite.Interpreter(model_path=mb_melgan_path)
    input_details = mb_melgan.get_input_details()
    output_details = mb_melgan.get_output_details()

    # resize input tensors according to actual shape
    mb_melgan.resize_tensor_input(
        input_details[0]["index"],
        [1, melspectrogram.shape[1], melspectrogram.shape[2]],
        strict=True,
    )

    # allocate tensors
    mb_melgan.allocate_tensors()

    # set input tensors
    mb_melgan.set_tensor(input_details[0]["index"], melspectrogram)

    # invoke interpreter
    mb_melgan.invoke()

    # return output
    return mb_melgan.get_tensor(output_details[0]["index"])


def main():
    processor = AutoProcessor.from_pretrained(
        "bookbot/lightspeech-mfa-id-v3", use_auth_token=True
    )
    processor.mode = "eval"  # change processor from train to eval mode

    text = "Halo, bagaimana kabar mu?"
    input_ids = tokenize(text, processor)

    mel_output_tflite, *_ = ls_infer(
        input_ids, speaker=0, lightspeech_path="lightspeech_quant.tflite"
    )

    audio_tflite = melgan_infer(mel_output_tflite, mb_melgan_path="mbmelgan.tflite")[
        0, :, 0
    ]

    sf.write("./audio.wav", audio_tflite, 32000, "PCM_16")


if __name__ == "__main__":
    main()
