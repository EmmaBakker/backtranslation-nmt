from tqdm import tqdm
import torch

def translate_text(texts, model, tokenizer, max_length=128, batch_size=32, decoding="greedy"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    translations = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

        generation_params = {
            "max_length": max_length, 
            "early_stopping": True    
        }

        if decoding == "greedy":
            generation_params["num_beams"] = 1
            generation_params["do_sample"] = False
        elif decoding == "sampling":
            generation_params["do_sample"] = True
            generation_params["top_k"] = 50       
            generation_params["top_p"] = 0.95    
            generation_params["temperature"] = 0.9 
        else:
            raise ValueError(f"Unknown decoding strategy: {decoding}")

        outputs = model.generate(**inputs, **generation_params) 

        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(batch_translations)

    return translations


def pivot_translate_text(texts, model, tokenizer, src_lang, tgt_lang, decoding="greedy", max_length=128, batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    tokenizer.src_lang = src_lang
    results = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"{src_lang} → {tgt_lang}"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        forced_bos = tokenizer.get_lang_id(tgt_lang)

        if decoding == "greedy":
            out_ids = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=max_length)
        elif decoding == "sampling":
            out_ids = model.generate(**inputs, forced_bos_token_id=forced_bos, do_sample=True, top_k=50, temperature=0.9, max_length=max_length)
        else:
            raise ValueError(f"Unknown decoding strategy: {decoding}")

        results.extend(tokenizer.batch_decode(out_ids, skip_special_tokens=True))

    return results
