import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from modules import AudioEncoder

class BartCaptionModel(nn.Module):
    def __init__(self, n_mels=128, num_of_conv=6, sr=16000, duration=10, max_length=128, 
                 label_smoothing=0.1, bart_type="facebook/bart-base", audio_dim=768):
        super(BartCaptionModel, self).__init__()
        # Initialize BART model and tokenizer
        bart_config = BartConfig.from_pretrained(bart_type)
        self.tokenizer = BartTokenizer.from_pretrained(bart_type)
        self.bart = BartForConditionalGeneration(bart_config)
        
        # Audio processing parameters
        self.n_sample = sr * duration
        self.hop_length = int(0.01 * sr)  # hard coding hop_size
        self.n_frames = int(self.n_sample // self.hop_length)
        self.num_of_stride_conv = num_of_conv - 1
        self.n_ctx = int(self.n_frames // 2**self.num_of_stride_conv) + 1
        
        # Audio encoder
        self.audio_encoder = AudioEncoder(
            n_mels=n_mels,  # hard coding n_mel
            n_ctx=self.n_ctx, 
            audio_dim=audio_dim, 
            text_dim=self.bart.config.hidden_size,
            num_of_stride_conv=self.num_of_stride_conv
        )

        self.max_length = max_length
        self.loss_fct = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)

    @property
    def device(self):
        return next(self.parameters()).device

    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def forward_encoder(self, audio):
        audio_embs = self.audio_encoder(audio)
        encoder_outputs = self.bart.model.encoder(
            input_ids=None,
            inputs_embeds=audio_embs,
            return_dict=True
        )
        return encoder_outputs, audio_embs

    def forward_decoder(self, text, encoder_outputs):
        text = self.tokenizer(text,
                              padding='longest',
                              truncation=True,
                              max_length=self.max_length,
                              return_tensors="pt")
        input_ids = text["input_ids"].to(self.device)
        attention_mask = text["attention_mask"].to(self.device)

        decoder_targets = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100
        )

        decoder_input_ids = self.shift_tokens_right(
            decoder_targets, self.bart.config.pad_token_id, self.bart.config.decoder_start_token_id
        )

        decoder_outputs = self.bart(
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=attention_mask,
            inputs_embeds=None,
            labels=None,
            encoder_outputs=encoder_outputs,
            return_dict=True
        )
        lm_logits = decoder_outputs["logits"]
        loss = self.loss_fct(lm_logits.view(-1, self.tokenizer.vocab_size), decoder_targets.view(-1))
        return loss

    def forward(self, audio, text):
        encoder_outputs, _ = self.forward_encoder(audio)
        loss = self.forward_decoder(text, encoder_outputs)
        return loss

    def generate(self,
                samples,
                use_nucleus_sampling=False,
                num_beams=5,
                max_length=128,
                min_length=2,
                top_p=0.9,
                repetition_penalty=1.0):
        """
        Generate captions for audio samples
        """
        # Generate audio embeddings
        audio_embs = self.audio_encoder(samples)

        # Run encoder to get hidden states
        encoder_outputs = self.bart.model.encoder(
            input_ids=None,
            inputs_embeds=audio_embs,
            return_dict=True
        )

        # Set up initial decoder inputs
        batch_size = samples.size(0)
        decoder_input_ids = torch.tensor(
            [[self.bart.config.decoder_start_token_id]] * batch_size,
            device=self.device
        )

        # Prepare attention mask for encoder outputs
        encoder_attention_mask = torch.ones(
            encoder_outputs.last_hidden_state.shape[0],
            encoder_outputs.last_hidden_state.shape[1],
            dtype=torch.long,
            device=self.device
        )

        # Choose generation strategy
        if use_nucleus_sampling:
            generation_kwargs = {
                "do_sample": True,
                "top_p": top_p,
                "max_length": max_length,
                "min_length": min_length,
                "repetition_penalty": repetition_penalty,
            }
        else:
            generation_kwargs = {
                "num_beams": num_beams,
                "max_length": max_length,
                "min_length": min_length,
                "repetition_penalty": repetition_penalty,
            }

        # Generate output
        with torch.no_grad():
            output_sequences = self.bart.generate(
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                attention_mask=encoder_attention_mask,
                **generation_kwargs
            )

        # Decode the generated sequences
        captions = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        return captions