"""
vision.py
---------
Adapta um modelo BLIP de descrição de imagens com legendas de referência do treino
e gera descrições breves e scores de sequência, preservando a separação por imagem.
"""
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import pandas as pd
from PIL import Image

from src.data.processor import project_path, require_columns
from src.models.feature_common import FeatureStage, image_records, set_seed, device_for, download_pretrained, save_metadata
from src.data.validator import write_json

def caption_partitions(config: dict, frames: dict, root: Path) -> dict[str, pd.DataFrame]:
    """Associa legendas por hash e rejeita imagens compartilhadas entre partições.

    Cada conteúdo visual aparece uma única vez em cada partição. O CSV precisa de
    image_path, image_sha256 e caption.
    """
    base = project_path(config["preprocessing"]["base_dir"], root)

    column = config["features"][0]
    records = {part: image_records(frame, column, base).drop_duplicates("sha256")
               for part, frame in frames.items()}
    
    seen = set()
    for part, table in records.items():
        hashes = set(table["sha256"])
        if seen.intersection(hashes):
            raise ValueError(f"Imagens compartilhadas em {part}; use split por image_hash")
        
        seen.update(hashes)

    if not config["training"]["enabled"]:
        return records
    
    annotations = pd.read_csv(project_path(config["training"]["captions_file"], root), dtype="string")
    require_columns(annotations, ["image_path", "image_sha256", "caption"], "Legendas de referência")

    if annotations[["image_sha256", "caption"]].isna().any().any() or annotations["caption"].str.strip().eq("").any():
        raise ValueError("Referências contêm hashes/legendas vazios")
    
    actual = image_records(annotations, "image_path", base)
    if not actual["sha256"].eq(annotations["image_sha256"]).all():
        raise ValueError("Uma imagem foi alterada após a anotação; revise image_sha256")
    
    if (annotations.groupby("image_sha256")["caption"].nunique() > 1).any():
        raise ValueError("O mesmo conteúdo visual possui legendas conflitantes")
    
    captions = annotations.drop_duplicates("image_sha256").set_index("image_sha256")["caption"]
    for part in ("train", "validation"):
        records[part] = records[part].assign(caption=records[part]["sha256"].map(captions))

        if records[part].empty or records[part]["caption"].isna().any():
            raise ValueError(f"Forneça uma legenda de referência para cada imagem de {part}")
        
    return records

def sequence_confidence(log_probabilities, token_ids, special_ids) -> float:
    """Média geométrica das probabilidades dos tokens de conteúdo.

    Tokens especiais e padding são excluídos, a sequência deve conter ao menos
    um token de conteúdo, e os logaritmos devem corresponder aos tokens gerados.
    """
    scores, ids = np.asarray(log_probabilities, dtype=float), np.asarray(token_ids)

    if scores.shape != ids.shape:
        raise ValueError("Tokens e scores da legenda não estão alinhados")
    
    selected = scores[~np.isin(ids, list(special_ids))]
    if not len(selected) or not np.isfinite(selected).all() or (selected > 1e-5).any():
        raise ValueError("Legenda sem tokens válidos ou probabilidades inválidas")
    
    return float(np.exp(selected.mean()))

class ImageCaptioner(FeatureStage):
    """BLIP com backbone visual congelado e ajuste leve do final do decoder textual."""

    def _load(self, path: Path) -> None:
        """Carrega processor e pesos locais."""
        from torch import nn
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        self.processor = BlipProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
        self.model = BlipForConditionalGeneration.from_pretrained(path, local_files_only=True)

        cast(nn.Module, self.model).to(self.device)
        self.model.eval()

    def _batch(self, rows: pd.DataFrame) -> dict:
        """Lê pixels RGB e tokeniza as legendas, mascarando padding na loss."""
        from torch import Tensor
        
        images = []
        for path in rows["path"]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        encoded = self.processor(
            images=images,
            text=rows["caption"].astype(str).tolist(),
            text_kwargs={
                "padding": True,
                "truncation": True,
                "max_length": int(self.config["training"]["max_length"])
            },
            common_kwargs={
                "return_tensors": "pt"
            }
        )

        batch = cast(dict[str, Tensor], dict(encoded))
        bos_token_id = self.model.config.text_config.bos_token_id

        if bos_token_id is None:
            raise ValueError("O tokenizer do BLIP não define um token BOS")
        
        batch["input_ids"][:, 0] = bos_token_id
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        return {key: value.to(self.device) for key, value in batch.items()}

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> Self:
        """Ajusta só imagens únicas do treino; usa legendas de validação para escolher época."""
        import torch
        
        partitions = caption_partitions(self.config, {"train": train, "validation": validation}, self.root)
        settings = self.config["training"]
        set_seed(self.seed)

        self.device = device_for(self.config.get("device", "auto"))
        snapshot = download_pretrained(self.config["fonte"], self.root)

        self._load(snapshot)
        self.history = {"loss": [], "val_loss": []}
        self.details.update(pretrained_snapshot=str(snapshot), pretrained_commit=snapshot.name,
                            unique_images={p: len(t) for p, t in partitions.items()}, target_used=False,
                            score_definition="geometric_mean_content_token_probability", language="en")
        
        self.model.requires_grad_(False)
        if not settings["enabled"]:
            return self
        
        blocks = self.model.text_decoder.bert.encoder.layer
        count = settings["unfreeze_last_layers"]

        if not 1 <= count <= len(blocks):
            raise ValueError("unfreeze_last_layers excede o número de camadas do decoder")
        
        for block in blocks[-count:]:
            block.requires_grad_(True)
        self.model.text_decoder.cls.requires_grad_(True)

        parameters = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
        optimizer = torch.optim.AdamW(parameters.values(), lr=float(settings["learning_rate"]))

        self.details["trainable_parameters"] = sum(p.numel() for p in parameters.values())
        rng = np.random.default_rng(self.seed)
        best, stale, state = float("inf"), 0, None
        for _ in range(settings["epochs"]):
            # Mantém dropout dos blocos congelados desligado.
            self.model.eval()

            for block in blocks[-count:]:
                block.train()
                
            self.model.text_decoder.cls.train()
            table = partitions["train"].iloc[rng.permutation(len(partitions["train"]))]

            total, tokens = 0., 0
            for start in range(0, len(table), settings["batch_size"]):
                batch = self._batch(table.iloc[start:start + settings["batch_size"]])
                optimizer.zero_grad()
                loss = self.model(**batch).loss
                if not torch.isfinite(loss):
                    raise ValueError("Loss de captioning não finita")
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters.values(), 1.0)
                optimizer.step()
                n = int((batch["labels"][:, 1:] != -100).sum())
                total, tokens = total + float(loss.detach()) * n, tokens + n

            self.history["loss"].append(total / tokens)
            self.model.eval()
            total, tokens = 0., 0
            with torch.inference_mode():
                table = partitions["validation"]

                for start in range(0, len(table), settings["batch_size"]):
                    batch = self._batch(table.iloc[start:start + settings["batch_size"]])
                    loss = float(self.model(**batch).loss)
                    n = int((batch["labels"][:, 1:] != -100).sum())
                    total, tokens = total + loss * n, tokens + n

            current = total / tokens
            self.history["val_loss"].append(current)
            if current < best:
                best, stale = current, 0
                state = {name: value.detach().cpu().clone() for name, value in parameters.items()}

            else:
                stale += 1
                if stale >= settings["patience"]:
                    break

        if state is None:
            raise ValueError("Nenhuma época válida de captioning")
        
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(state[name].to(self.device))

        self.model.eval()
        self.details["best_validation_loss"] = best

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Descreve os pixels com geração determinística e reaproveita imagens repetidas."""
        import torch

        records = image_records(data, self.config["features"][0], project_path(self.config["preprocessing"]["base_dir"], self.root))
        cache = {}
        settings = self.config["generation"]
        
        self.model.eval()
        with torch.inference_mode():
            for row in records.drop_duplicates("sha256").itertuples():
                with Image.open(str(row.path)) as image:
                    inputs = self.processor(images=image.convert("RGB"), common_kwargs={"return_tensors":"pt"})

                pixel_values = cast(Any, inputs["pixel_values"])
                pixels = pixel_values.to(self.device)

                result = cast(Any, self.model.generate(
                    pixel_values=pixels, do_sample=False,
                    num_beams=settings["num_beams"],
                    max_new_tokens=settings["max_new_tokens"],
                    repetition_penalty=settings.get("repetition_penalty", 1.0),
                    no_repeat_ngram_size=settings.get("no_repeat_ngram_size", 0),
                    return_dict_in_generate=True,
                ))

                description = self.processor.decode(result.sequences[0], skip_special_tokens=True).strip()

                if not description:
                    raise ValueError("BLIP produziu uma descrição vazia")

                sequences = result.sequences
                outputs = self.model(pixel_values=pixels, input_ids=sequences)
                log_probs = torch.log_softmax(outputs.logits[:, :-1, :].float(), dim=-1)

                selected = log_probs.gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
                
                confidence = sequence_confidence(selected[0].cpu().numpy(), sequences[0, 1:].cpu().numpy(), cast(Any, self.processor).tokenizer.all_special_ids)

                cache[row.sha256] = (description, confidence)
                
        return pd.DataFrame([cache[h] for h in records["sha256"]], columns=["image_description", "image_score"], index=data.index)

    def save(self, directory: Path) -> None:
        """Persiste modelo ajustado, processor, histórico e metadados."""
        save_metadata(self, directory)
        self.model.save_pretrained(directory / "model", safe_serialization=True)
        self.processor.save_pretrained(directory / "model")
        write_json(directory / "history.json", self.history)

    @classmethod
    def restore(cls, directory: Path, config: dict, root: Path, seed: int = 42) -> Self:
        """Restaura os pesos ajustados locais sem repetir o fine-tuning."""
        instance = cls(config, root, seed)
        instance.device = device_for(config.get("device", "auto"))
        instance._load(directory / "model")
        instance.history = {}
        return instance
