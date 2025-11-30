#!/usr/bin/env python
# coding=utf-8
import torch
from transformers import AutoTokenizer
from diffusers import FluxPipeline
from typing import List, Dict, Optional, Union, Tuple

def add_custom_tokens(
    pipeline: FluxPipeline,
    tokens: List[str],
) -> Tuple[FluxPipeline, List[int]]:
    """
    Add custom tokens to the tokenizer.
    
    Args:
        pipeline: The FLUX pipeline
        tokens: List of tokens to add
        
    Returns:
        Updated pipeline and list of token IDs
    """
    # Add tokens to the tokenizer
    tokenizer = pipeline.tokenizer
    num_added_tokens = tokenizer.add_tokens(tokens)
    print(f"Added {num_added_tokens} tokens to the tokenizer")
    
    # Resize token embeddings for text encoder
    pipeline.text_encoder.resize_token_embeddings(len(tokenizer))
    
    # Get token IDs
    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
    print(f"Token IDs: {token_ids}")
    
    return pipeline, token_ids

def initialize_token_embeddings(
    pipeline: FluxPipeline,
    token_ids: List[int],
    init_words: List[str],
) -> FluxPipeline:
    """
    Initialize token embeddings with embeddings of specific words.
    
    Args:
        pipeline: The FLUX pipeline
        token_ids: List of token IDs to initialize
        init_words: List of words to use for initialization
        
    Returns:
        Updated pipeline
    """
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    
    # Get the word embeddings
    for token_id, word in zip(token_ids, init_words):
        # Tokenize the word
        word_tokens = tokenizer.encode(word, add_special_tokens=False)
        
        if len(word_tokens) == 0:
            print(f"Warning: '{word}' tokenized to empty list. Using random initialization for token ID {token_id}")
            continue
            
        # # Use the first token's embedding if the word is split into multiple tokens
        # word_token_id = word_tokens[0]
        # # Get the embedding
        # word_embedding = text_encoder.get_input_embeddings().weight[word_token_id].clone().detach()
        
        # 改进版本:使用所有token embeddings的平均值
        if len(word_tokens) > 1:
            # 获取所有token的embeddings
            all_embeddings = [text_encoder.get_input_embeddings().weight[id].clone().detach() for id in word_tokens]
            # 计算平均值
            word_embedding = torch.stack(all_embeddings).mean(dim=0)
        else:
            word_embedding = text_encoder.get_input_embeddings().weight[word_tokens[0]].clone().detach()

        # Set the new token's embedding
        with torch.no_grad():
            text_encoder.get_input_embeddings().weight[token_id] = word_embedding
            
        print(f"Initialized token ID {token_id} with embedding of '{word}'")
    
    return pipeline

def make_tokens_learnable(
    pipeline: FluxPipeline,
    token_ids: List[int],
    train_nonspecial_only: bool = False,
) -> FluxPipeline:
    """
    Make specific token embeddings learnable, while freezing others.
    
    Args:
        pipeline: The FLUX pipeline
        token_ids: List of token IDs to make learnable
        train_nonspecial_only: If True, only train non-special tokens
        
    Returns:
        Updated pipeline
    """
    text_encoder = pipeline.text_encoder
    
    # Freeze all token embeddings first
    for param in text_encoder.get_input_embeddings().parameters():
        param.requires_grad = False
    
    # Make the specified token embeddings learnable
    for token_id in token_ids:
        text_encoder.get_input_embeddings().weight[token_id].requires_grad = True
        
    # Count trainable tokens
    trainable_tokens = len(token_ids)
    print(f"Made {trainable_tokens} token embeddings learnable")
    
    return pipeline

def setup_custom_tokens(
    pipeline: FluxPipeline,
    tokens: List[str],
    init_words: List[str],
    make_learnable: bool = True,
) -> Tuple[FluxPipeline, List[int]]:
    """
    Complete workflow to add custom tokens, initialize them, and make them learnable.
    
    Args:
        pipeline: The FLUX pipeline
        tokens: List of tokens to add
        init_words: List of words to use for initialization
        make_learnable: Whether to make the tokens learnable
        
    Returns:
        Updated pipeline and list of token IDs
    """
    assert len(tokens) == len(init_words), "Number of tokens must match number of initialization words"
    
    # Add tokens
    pipeline, token_ids = add_custom_tokens(pipeline, tokens)
    
    # Initialize embeddings
    pipeline = initialize_token_embeddings(pipeline, token_ids, init_words)
    
    # Make tokens learnable
    if make_learnable:
        pipeline = make_tokens_learnable(pipeline, token_ids)
    
    return pipeline, token_ids

def save_token_embeddings(
    pipeline: FluxPipeline,
    token_ids: List[int],
    output_file: str,
) -> None:
    """
    Save token embeddings to a file.
    
    Args:
        pipeline: The FLUX pipeline
        token_ids: List of token IDs to save
        output_file: Path to output file
    """
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    
    # Get tokens from IDs
    tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in token_ids]
    
    # Get embeddings
    embeddings = {
        token: text_encoder.get_input_embeddings().weight[token_id].clone().detach().cpu()
        for token, token_id in zip(tokens, token_ids)
    }
    
    # Save
    torch.save(embeddings, output_file)
    print(f"Saved token embeddings to {output_file}")

def load_token_embeddings(
    pipeline: FluxPipeline,
    input_file: str,
) -> Tuple[FluxPipeline, List[int]]:
    """
    Load token embeddings from a file.
    
    Args:
        pipeline: The FLUX pipeline
        input_file: Path to input file
        
    Returns:
        Updated pipeline and list of token IDs
    """
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    
    # Load embeddings
    embeddings_dict = torch.load(input_file)
    
    # Add tokens to tokenizer
    tokens = list(embeddings_dict.keys())
    pipeline, token_ids = add_custom_tokens(pipeline, tokens)
    
    # Set embeddings
    for token, embedding in embeddings_dict.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        with torch.no_grad():
            text_encoder.get_input_embeddings().weight[token_id] = embedding.to(text_encoder.device)
    
    print(f"Loaded token embeddings from {input_file}")
    return pipeline, token_ids

