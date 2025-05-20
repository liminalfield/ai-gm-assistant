# hf_integration.py
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
import torch, logging

class LLMIntegrationHF:
    def __init__(self):
        model_id = "meta-llama/Llama-2-13b-chat-hf"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer (with auth)
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_id, use_fast=False, use_auth_token=True
        )

        # 4-bit quant config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load & quantize
        self.model = LlamaForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
            use_auth_token=True
        ).to(self.device)

        logging.info("Loaded Llama-2-13b-chat-hf in 4-bit on %s", self.device)

    def generate_response(self, query: str, context: str) -> str:
        # 1) Build a single system+user prompt
        system_msg = (
            "You are an expert RPG rules assistant. "
            "You are given a user question and some excerpts from official rulebooks. "
            "Your job is to answer the question directly, using only the information in the context provided. "
            "If the context includes tables, data, or rule definitions, extract them and use them to construct a complete and useful answer. "
            "Do not explain what the user is asking. Do not reflect on the question. Just answer it using rules from the context. "
            "If a table is mentioned but not shown, say so. Otherwise, summarize how the rules work in plain language, using actual numbers or examples where appropriate. "
            "Do not invent rules. Do not make assumptions outside the context. Use quotes and citations only if they clarify the rules."
        )

        user_msg = f"Question: {query}\n\nContext:\n{context}\n\nExtracted Rules:\n"
        prompt = f"{system_msg}\n\n{user_msg}"

        # 2) Truncate so prompt + output ≤ model window
        max_model_len = self.model.config.max_position_embeddings  # e.g. 4096
        max_new_tokens = 1024
        max_input_tokens = max_model_len - max_new_tokens

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(self.device)

        # 3) Greedy decode under no_grad
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 4) Slice off the prompt tokens and decode only the new ones
        gen_start = inputs["input_ids"].shape[-1]
        gen_tokens = outputs[0][gen_start:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    def generate_direct(self, query: str) -> str:
        """
        Free‐form answer with sampling.
        """
        prompt = (
            "You are a helpful RPG rules assistant.\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings - 128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,  # sampling enabled
                temperature=0.7,  # randomness
                top_p=0.9,  # nucleus sampling
                eos_token_id=self.tokenizer.eos_token_id,
            )

        start = inputs["input_ids"].shape[-1]
        gen = outputs[0][start:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
