import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"#change when needed
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
import simplejson as json
from accelerate import Accelerator, dispatch_model, infer_auto_device_map
from transformers import pipeline

# import bitsandbytes as bnb
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
token = "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"


# ===============================
# Custom Attention Module
# ===============================
class LlamaAttentionWeightsOnly(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.head_dim = original_attn.head_dim

        self.num_query_heads = self.q_proj.out_features // self.head_dim
        self.num_kv_heads = self.k_proj.out_features // self.head_dim
        self.expand_factor = self.num_query_heads // self.num_kv_heads

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states,
        rotary_embeddings,
        attn_mask,
        output_attentions=False,
        **kwargs,
    ):
        batch, seq_len, _ = hidden_states.shape

        # Compute Q and K projections.
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)

        # Reshape Q
        q = q.view(batch, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        # Reshape K
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.apply_rotary_pos_emb(
            q, k, rotary_embeddings[0], rotary_embeddings[1]
        )

        # Expand key heads
        k = k.repeat_interleave(self.expand_factor, dim=1)

        # Dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-1, -2))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        return attn_weights


# ===============================
# TokenPruner Module
# ===============================
class TokenPruner(nn.Module):
    def __init__(self, small_model_id, compression_ratio, device="cuda:0"):
        super().__init__()
        # Load the small model
        small_model = AutoModelForCausalLM.from_pretrained(
            small_model_id,
            trust_remote_code=True,
            token=token,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.embeddings = small_model.get_input_embeddings()
        self.rotary_embeddings = small_model.model.rotary_emb

        # Replace the first self-attention layer
        original_attn = small_model.model.layers[0].self_attn
        self.self_attention1 = LlamaAttentionWeightsOnly(original_attn)

        # Delete the rest of the small model to save memory
        del small_model

        self.compression_ratio = compression_ratio

    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        batch_size, seq_len, hidden_dim = embeddings.shape

        # Rotary embeddings
        pos_ids = torch.arange(seq_len).unsqueeze(0).to(input_ids.device)
        rotary_embeds = self.rotary_embeddings(input_ids, position_ids=pos_ids)

        attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=embeddings.device))
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        attention_weights = self.self_attention1(
            embeddings, rotary_embeds, attn_mask=attn_mask
        )

        # Average over heads and query dimension => token importance
        #################################DIFF###########################################
        # importance_scores = attention_weights.mean(dim=1).mean(dim=1) # [1, seq_len]
        importance_scores = attention_weights.mean(dim=1).mean(dim=1)[0] # [seq_len]
        #################################DIFF###########################################

        # Number of tokens to keep
        compressed_length = int(seq_len * self.compression_ratio)
        compressed_length = max(compressed_length, 2)

        # Top-k token indices
        _, topk_ids = torch.topk(
            importance_scores, compressed_length,  sorted=False
        ) # [seq_len]
        # Sort to preserve order
        # topk_ids = torch.sort(topk_ids, dim=1)[0] # [1, seq_len]
        topk_ids = torch.sort(topk_ids)[0] # [seq_len]

        if topk_ids[-1] != seq_len - 1:
            topk_ids = torch.cat(
                (topk_ids, torch.tensor([seq_len - 1], device=topk_ids.device)), dim=1
            )

        # Gather pruned tokens
        pruned_tokens = input_ids.gather(1, topk_ids)
        return pruned_tokens, topk_ids


# ===============================
# LlamaPrunedModel Module
# ===============================
class LlamaPrunedModel(nn.Module):
    def __init__(self, main_model_id, small_model_id, compression_ratio):
        super().__init__()
        self.main_model = AutoModelForCausalLM.from_pretrained(
            main_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=token,
            # quantization_config=bnb_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            main_model_id,
            trust_remote_code=True,
            token=token,
        )
        self.main_model_pipeline = pipeline(
            "text-generation",
            model=self.main_model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,  # force fp16 mode
            device_map="auto",
            return_full_text=False,
            use_cache=True,
            max_new_tokens=1,
        )
        self.embeddings = self.main_model.get_input_embeddings()
        self.token_pruner = TokenPruner(
            small_model_id, compression_ratio, device="cuda:0"
        )

        # Freeze main model
        for param in self.main_model.parameters():
            param.requires_grad = False
        self.embeddings.requires_grad = False

    def forward(self, input_ids):
        pruned_tokens_ids, position_ids = self.token_pruner(input_ids.to("cuda:0"))
        pruned_tokens = self.tokenizer.decode(pruned_tokens_ids, skip_special_tokens=True)['generated_text']
        # output = self.main_model.generate(
        #     input_ids=pruned_tokens,
        #     position_ids=position_ids,
        #     max_new_tokens=1,
        #     do_sample=False,
        # )
        output = self.main_model_pipeline(pruned_tokens)
        return output[-1]


def main():
    accelerator = Accelerator()

    small_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    main_model_id = "meta-llama/Llama-3.1-70B-Instruct"
    compression_ratio = 0.9

    tokenizer = AutoTokenizer.from_pretrained(main_model_id, token=token)

    model = LlamaPrunedModel(main_model_id, small_model_id, compression_ratio)
    # model = dispatch_model(
    #     model,
    #     device_map=infer_auto_device_map(
    #         model,
    #         max_memory_utilization=0.5,
    #         offload_buffers=True,
    #         dtype=torch.int8,
    #         fallback_allocation=True,
    #     ),
    #     offload_dir="./offload",
    # )
    model.eval()

    # List of languages
    all_langs = ["french", "german", "hindi", "italian", "portugese", "spanish", "thai"]

    for lang in all_langs:
        # Build config name for the dataset
        ds_config_name = f"Llama-3.2-1B-Instruct-evals__mmlu_{lang}_chat__details"

        # Load dataset
        dataset = datasets.load_dataset(
            "meta-llama/Llama-3.2-1B-Instruct-evals", ds_config_name, token=token
        )["latest"]

        # Preprocessing
        def mapping(example):
            outputs = tokenizer(example["input_final_prompts"])
            example["input_ids"] = outputs["input_ids"][0]
            example["mask"] = outputs["attention_mask"][0]
            return example

        dataset = dataset.map(
            mapping,
            remove_columns=[
                "task_type",
                "task_name",
                "input_question",
                "input_choice_list",
                "output_prediction_text",
                "output_choice_negative_log_likelihoods",
                "input_question_hash",
                "benchmark_label",
                "eval_config",
                "output_metrics",
            ],
        )

        local_data = []
        total_flops = 0

        with torch.no_grad():
            for sample in tqdm.tqdm(dataset, desc=f"Processing {lang}"):
                input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long).to(
                    model.main_model.device
                )

                    # Profile on main process only
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_flops=True,
                    profile_memory=True,
                ) as prof:
                    output = model(input_ids)

                # pred_token = tokenizer.decode(output).strip()
                sample["output"] = output

                # Compare predicted token with target token (example logic)
                target_token = tokenizer.decode(
                    tokenizer.encode(sample["input_correct_responses"][0])[-1]
                ).strip()

                current_flops = prof.key_averages().total_average().flops
                total_flops += current_flops
                sample["flops"] = current_flops

                local_data.append(sample)
                torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # Store each process's local_data in a JSON Lines file, appending lines.
        # Everyone uses the same file name for a given lang, opened in 'a' mode.
        # ---------------------------------------------------------------------
        filename = f"simplePruning_{lang}_183.jsonl"
        with open(filename, "a") as f:
            for item in local_data:
                f.write(json.dumps(item) + "\n")

        # Optionally, only rank 0 logs total FLOPS for that language
        if accelerator.is_main_process:
            print(f"Finished {lang}: appended results to {filename}")
            print(f"Total FLOPS for {lang}: {total_flops}")


if __name__ == "__main__":
    main()
