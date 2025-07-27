import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import login, whoami
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, LlamaForCausalLM


def setup_huggingface_auth():
    """
    Handle HuggingFace authentication with multiple fallback methods.
    This ensures models can be downloaded regardless of how authentication is set up.
    """
    # Method 1: Check if already logged in via CLI
    import logging

    try:
        user_info = whoami()
        print(f"Already authenticated as: {user_info['name']}")
        return True
    except Exception:
        logging.exception("Exception occurred during HuggingFace authentication check")
        raise

    # Method 2: Try environment variable
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        print("Using HuggingFace token from environment variable")
        login(token=hf_token)
        return True

    # Method 3: Interactive login prompt for development
    print(
        "No authentication found. Please run 'huggingface-cli login' or set HUGGINGFACE_HUB_TOKEN"
    )
    return False


class BaseSentimentAnalyzer(ABC):
    """
    Abstract base class for sentiment analyzers.

    This creates a contract that all sentiment models must follow,
    ensuring they provide the same interface regardless of their
    internal architecture differences.
    """

    @abstractmethod
    def analyze_sentiment(self, text: str) -> dict[str, Any]:
        """
        Analyze sentiment of input text.

        Returns:
            Dict with keys: 'sentiment', 'confidence', 'all_scores'
        """

    @abstractmethod
    def analyze_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Analyze sentiment for multiple texts efficiently.
        """


class FinBERTAnalyzer(BaseSentimentAnalyzer):
    """
    FinBERT-based sentiment analyzer.

    This handles the classification-based approach where the model
    directly outputs probabilities for sentiment classes.
    """

    def __init__(self, cache_dir: Path | str, force_download: bool = False):
        self.model_name = "ProsusAI/finbert"
        cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        self.cache_dir = cache_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading FinBERT with cache directory: {cache_dir}")

        # Create cache directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Load tokenizer and model with explicit caching
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            force_download=force_download,
            trust_remote_code=True,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            force_download=force_download,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Move model to appropriate device
        self.model = self.model.to(self.device)
        self.model.eval()

        # FinBERT outputs: [positive, negative, neutral]
        self.labels = ["positive", "negative", "neutral"]

        print(f"FinBERT loaded successfully on {self.device}")

    def analyze_sentiment(self, text: str) -> dict[str, Any]:
        """
        Analyze sentiment using FinBERT's classification approach.

        This method tokenizes the input, runs it through the model,
        and converts the logits to probabilities.
        """
        # Tokenize input text with proper truncation
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        # Run inference without gradient computation for efficiency
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Convert logits to probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Extract predictions and confidence scores
        predicted_class = probabilities[0].argmax().item()
        confidence = probabilities[0][int(predicted_class)].item()

        return {
            "sentiment": self.labels[int(predicted_class)],
            "confidence": confidence,
            "all_scores": {
                "positive": probabilities[0][0].item(),
                "negative": probabilities[0][1].item(),
                "neutral": probabilities[0][2].item(),
            },
        }

    def analyze_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Efficiently process multiple texts in a single forward pass.

        This is much faster than calling analyze_sentiment multiple times
        because it batches the computation.
        """
        # Tokenize all texts together for batch processing
        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Process each result in the batch
        results = []
        for i in range(len(texts)):
            predicted_class = int(probabilities[i].argmax().item())
            confidence = probabilities[i][int(predicted_class)].item()

            results.append(
                {
                    "sentiment": self.labels[int(predicted_class)],
                    "confidence": confidence,
                    "all_scores": {
                        "positive": probabilities[i][0].item(),
                        "negative": probabilities[i][1].item(),
                        "neutral": probabilities[i][2].item(),
                    },
                }
            )

        return results


class FinGPTAnalyzer(BaseSentimentAnalyzer):
    """
    FinGPT-based sentiment analyzer using instruction following.

    This handles the generative approach where we provide instructions
    to the model and parse its text response.
    """

    def __init__(self, cache_dir: Path | str, force_download: bool = False):
        self.base_model_name = "meta-llama/Meta-Llama-3-8B"
        self.peft_model_name = "FinGPT/fingpt-mt_llama3-8b_lora"
        cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        self.cache_dir = cache_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure authentication for Llama model
        if not setup_huggingface_auth():
            raise ValueError("Authentication required for FinGPT (uses gated Llama model)")

        print(f"Loading FinGPT with cache directory: {cache_dir}")

        # Create cache directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Load the base Llama tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            cache_dir=cache_dir,
            force_download=force_download,
            trust_remote_code=True,
        )
        # Set padding token for batch processing
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the base Llama model
        self.model = LlamaForCausalLM.from_pretrained(
            self.base_model_name,
            cache_dir=cache_dir,
            force_download=force_download,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",  # Automatically handle multi-GPU if available
        )

        # Apply the FinGPT LoRA adapters on top of the base model
        print("Loading FinGPT LoRA adapters...")
        self.model = PeftModel.from_pretrained(self.model, self.peft_model_name)
        self.model.eval()

        print("FinGPT loaded successfully")

    def _create_prompt(self, text: str) -> str:
        """
        Create the instruction prompt format that FinGPT expects.

        FinGPT was trained with specific instruction formatting,
        so we need to match that pattern exactly.
        """
        return f"""Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}
Input: {text}
Answer: """

    def _parse_response(self, generated_text: str, original_prompt: str) -> str:
        """
        Extract the sentiment answer from the generated text.

        The model generates the full prompt + its answer, so we need
        to extract just the answer portion.
        """
        try:
            # Remove the original prompt to get just the generated answer
            if "Answer: " in generated_text:
                answer = generated_text.split("Answer: ")[-1].strip()
                # Clean up common variations in the response
                answer = answer.lower().strip()

                # Map variations to standard labels
                if "positive" in answer:
                    return "positive"
                elif "negative" in answer:
                    return "negative"
                elif "neutral" in answer:
                    return "neutral"
                else:
                    # If unclear, try to extract first word
                    first_word = answer.split()[0] if answer.split() else ""
                    if first_word in ["positive", "negative", "neutral"]:
                        return first_word
                    else:
                        return "neutral"  # Default fallback
            else:
                return "neutral"  # Fallback if parsing fails
        except Exception as exc:
            print(f"Error parsing FinGPT response: {exc}")
            raise

    def analyze_sentiment(self, text: str) -> dict[str, Any]:
        """
        Analyze sentiment using FinGPT's generative approach.

        This method creates an instruction prompt, generates a response,
        and parses the answer to extract sentiment.
        """
        prompt = self._create_prompt(text)

        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        # Generate response with controlled parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,  # We only need a few tokens for sentiment
                temperature=0.1,  # Low temperature for consistent responses
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the generated response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        sentiment = self._parse_response(generated_text, prompt)

        # FinGPT doesn't provide confidence scores like classification models,
        # so we'll use a placeholder confidence based on successful parsing
        confidence = 0.85 if sentiment in ["positive", "negative", "neutral"] else 0.5

        # Create simplified scores (since FinGPT doesn't provide probabilities)
        all_scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        all_scores[sentiment] = confidence

        return {"sentiment": sentiment, "confidence": confidence, "all_scores": all_scores}

    def analyze_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Process multiple texts by calling analyze_sentiment for each.

        Note: FinGPT's generative nature makes efficient batching more complex
        than classification models, so we process sequentially for reliability.
        """
        return [self.analyze_sentiment(text) for text in texts]


def load_sentiment_analyzer(
    model_name: str, cache_dir: Path | None = None, force_download: bool = False
) -> BaseSentimentAnalyzer:
    """
    Factory function to load either FinBERT or FinGPT sentiment analyzer.

    This is your main entry point - it handles all the complexity of loading
    different model architectures behind a simple interface.

    Args:
        model_name: Either 'finbert' or 'fingpt'
        cache_dir: Directory for model caching (uses config default if None)
        force_download: Whether to re-download cached models

    Returns:
        A sentiment analyzer instance that follows the BaseSentimentAnalyzer interface
    """
    # Import settings here to avoid circular imports
    try:
        from src.config import settings

        if cache_dir is None:
            cache_dir = settings.huggingface.cache_dir
    except ImportError:
        # Fallback if config not available
        if cache_dir is None:
            hf_home = os.getenv("HF_HOME")
            if hf_home is not None:
                cache_dir = Path(hf_home)
            else:
                cache_dir = Path.home() / ".cache" / "huggingface"

    # Normalize model name to handle variations
    model_name = model_name.lower().strip()

    if model_name == "finbert":
        print("Loading FinBERT sentiment analyzer...")
        return FinBERTAnalyzer(cache_dir=cache_dir, force_download=force_download)

    elif model_name == "fingpt":
        print("Loading FinGPT sentiment analyzer...")
        return FinGPTAnalyzer(cache_dir=cache_dir, force_download=force_download)

    else:
        available_models = ["finbert", "fingpt"]
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models}")


# Convenience function for quick usage
def create_sentiment_analyzer(model_name: str = "finbert") -> BaseSentimentAnalyzer:
    """
    Create a sentiment analyzer with default settings.

    This is a simpler interface for common usage patterns.
    """
    return load_sentiment_analyzer(
        model_name
    )  # Example usage that works identically for both models
