import sys
import tensorflow as tf
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(inputs.tokens(), result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Retorna el índice del token [MASK] en la secuencia.
    """
    # inputs["input_ids"] es un tensor de forma (1, n_tokens)
    input_ids = inputs["input_ids"][0].numpy()
    for index, token_id in enumerate(input_ids):
        if token_id == mask_token_id:
            return index
    return None


def get_color_for_attention_score(attention_score):
    """
    Convierte un score (0-1) en una tupla RGB de gris (0-255).
    """
    # Escala lineal: score 1.0 -> 255 (blanco), 0.0 -> 0 (negro)
    value = int(attention_score * 255)
    return (value, value, value)


def visualize_attentions(tokens, attentions):
    """
    Genera diagramas para todas las capas (12) y cabezales (12).
    """
    # attentions es una tupla de 12 capas
    for layer_idx, layer in enumerate(attentions):
        # Cada capa es un tensor (batch_size, num_heads, sequence_length, sequence_length)
        num_heads = layer.shape[1]
        for head_idx in range(num_heads):
            # generate_diagram usa numeración basada en 1
            generate_diagram(
                layer_idx + 1,
                head_idx + 1,
                tokens,
                layer[0][head_idx] # Extraemos la matriz de atención para este cabezal
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Genera un diagrama de atención para un cabezal específico.
    """
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    for i, token in enumerate(tokens):
        # Dibujar columnas de tokens
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Dibujar filas de tokens
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()
