import torch
import torch.nn as nn
import pickle
import argparse

import sys, os
sys.path.append(os.path.dirname(__file__))

from train import Encoder, Decoder, Seq2Seq
from infer import load_model

def export_to_onnx(model_path, vocab_path, enc_onnx_path, dec_onnx_path):
    """
    Exports the trained Encoder and Decoder PyTorch models to separate ONNX files.
    """
    # 1. Load the trained PyTorch model and vocabularies
    print("⏳ Loading PyTorch model...")
    model, letter2id, ph2id, id2ph, device = load_model(model_path, vocab_path)
    print("✅ PyTorch model loaded successfully.")

    # 2. Export the Encoder model
    print("⏳ Exporting Encoder to ONNX...")
    enc = model.enc
    
    # Create dummy input for the Encoder
    # The Encoder takes 'src' (padded sequence of letter IDs) and 'src_len' (actual lengths).
    # We'll use a batch size of 1 for simplicity.
    dummy_src = torch.randint(low=1, high=len(letter2id), size=(1, 10), device=device)
    dummy_src_len = torch.tensor([10], device=device)
    
    # We must trace the forward pass, so we get the outputs
    enc_outputs, hidden, cell = enc(dummy_src, dummy_src_len)

    # ONNX export for the encoder
    # The inputs are 'src' and 'src_len'.
    # The outputs are 'enc_outputs', 'hidden', and 'cell'.
    torch.onnx.export(
        enc,
        (dummy_src, dummy_src_len),
        enc_onnx_path,
        export_params=True,
        opset_version=11, # Recommended opset
        do_constant_folding=True,
        input_names=['src_input', 'src_len_input'],
        output_names=['encoder_outputs', 'encoder_hidden', 'encoder_cell'],
        dynamic_axes={
            'src_input': {0: 'batch_size', 1: 'sequence_length'},
            'src_len_input': {0: 'batch_size'},
            'encoder_outputs': {0: 'batch_size', 1: 'sequence_length'},
            'encoder_hidden': {1: 'batch_size'},
            'encoder_cell': {1: 'batch_size'}
        }
    )
    print(f"✅ Encoder exported to {enc_onnx_path}")

    # 3. Export the Decoder model
    print("⏳ Exporting Decoder to ONNX...")
    dec = model.dec
    
    # Create dummy input for the Decoder
    # The Decoder's forward method takes 'input' (a single token ID), 'hidden', and 'cell'.
    dummy_input = torch.tensor([1], device=device) # <sos> token
    # Get the shapes of hidden and cell states from the Encoder's output
    dummy_hidden = torch.randn(hidden.shape, device=device)
    dummy_cell = torch.randn(cell.shape, device=device)

    # ONNX export for the decoder
    # The inputs are 'input_token', 'hidden_state', and 'cell_state'.
    # The outputs are 'output_logits', 'hidden_state_out', and 'cell_state_out'.
    torch.onnx.export(
        dec,
        (dummy_input, dummy_hidden, dummy_cell),
        dec_onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_token', 'decoder_hidden_in', 'decoder_cell_in'],
        output_names=['output_logits', 'decoder_hidden_out', 'decoder_cell_out'],
        dynamic_axes={
            'input_token': {0: 'batch_size'},
            'decoder_hidden_in': {1: 'batch_size'},
            'decoder_cell_in': {1: 'batch_size'},
            'output_logits': {0: 'batch_size'},
            'decoder_hidden_out': {1: 'batch_size'},
            'decoder_cell_out': {1: 'batch_size'}
        }
    )
    print(f"✅ Decoder exported to {dec_onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Seq2Seq model to ONNX")
    parser.add_argument("--model", default="seq2seq.pt", help="Trained model checkpoint")
    parser.add_argument("--vocab", default="vocabs.pkl", help="Pickle with vocabularies")
    parser.add_argument("--enc_output", default="encoder.onnx", help="Output path for encoder ONNX model")
    parser.add_argument("--dec_output", default="decoder.onnx", help="Output path for decoder ONNX model")
    args = parser.parse_args()

    # Call the export function
    export_to_onnx(args.model, args.vocab, args.enc_output, args.dec_output)
