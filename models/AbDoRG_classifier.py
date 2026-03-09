import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType

from arm.Finetuning.models_mamba import arm_base_pz16, arm_large_pz16

class MambaCTDownStream(pl.LightningModule):
    """
    MambaCTDownStream model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.json_path = args.annotation
        self.type = args.type

        print(f'Loading vision encoder:{args.vision_model}')
        if 'B' in args.vision_model:
            self.visual_encoder = arm_base_pz16(self.type)
        else:
            self.visual_encoder = arm_large_pz16(self.type)
        finetune = args.vision_model
        if finetune!='None':
            checkpoint = torch.load(finetune, map_location='cpu')
            print(f"Load arm pre-trained checkpoint from: {finetune}" )
            checkpoint_model = checkpoint['model']
            new_dict = {}
            for k, v in checkpoint_model.items():
                if "visual_encoder." in k:
                    new_dict[k.replace("visual_encoder.", "")] = v
            # load pre-trained model
            self.visual_encoder.load_state_dict(new_dict)

        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')
        # breakpoint()
        print(f'Loading LLM ...')
        
        if args.dataset == 'ct':
            llama_model="Qwen/Qwen1.5-1.8B-Chat"
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, trust_remote_code=True, use_fast=False)
            self.llama_tokenizer.pad_token_id = 0
            self.llama_tokenizer.bos_token_id = 0
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.float16,
                )
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')
        else:
            llama_model="meta-llama/Llama-2-7b-chat-hf"
            #self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, trust_remote_code=True, use_fast=False)
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, trust_remote_code=True, use_fast=False)
            self.llama_tokenizer.pad_token_id = 0
            if args.low_resource:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map="auto"
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.float16,
                )
            if args.llm_use_lora:
                self.embed_tokens = self.llama_model.get_input_embeddings()
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, 
                    inference_mode=False, 
                    r=args.llm_r, 
                    lora_alpha=args.llm_alpha, 
                    lora_dropout=args.lora_dropout,
                    target_modules=["q_proj", "v_proj"],
                )
                self.llama_model = get_peft_model(self.llama_model, peft_config)
                self.llama_model.print_trainable_parameters()
                llama_dict = {}
                for k, v in checkpoint_model.items():
                    if "text_encoder." in k:
                        llama_dict[k.replace("text_encoder.", "")] = v
                self.llama_model.load_state_dict(llama_dict, strict = False)
                for name, param in self.llama_model.named_parameters():
                    param.requires_grad = False
                print('Loading LLAMA LoRA Done')         
            else:
                self.embed_tokens = self.llama_model.get_input_embeddings()
                for name, param in self.llama_model.named_parameters():
                    param.requires_grad = False
                print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this abdomen CT scan image.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')


    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)

        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        )
        # print(empty_targets.size(),targets.size())
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def on_train_epoch_end(self):
        # Always save at the end of a training epoch, useful if validation is disabled
        if self.trainer.local_rank == 0:
            self.save_checkpoint()

    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)

        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref
    
    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w', encoding='utf-8'),ensure_ascii=False)
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w', encoding='utf-8'),ensure_ascii=False)
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            self.save_checkpoint(eval_res)
            if val_score > self.val_score:
                self.val_score = val_score

        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w', encoding='utf-8'),ensure_ascii=False)
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w', encoding='utf-8'),ensure_ascii=False)
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def save_checkpoint(self, eval_res=None):
        if eval_res is None:
            eval_res = {'Bleu_4': 0.0, 'CIDEr': 0.0}
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res.get('Bleu_4', 0.0), eval_res.get('CIDEr', 0.0)),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    def encode_img(self, images, segmentation=None):
        image_embeds = []
        for image in images:
            device = image.device
            image_embed = self.visual_encoder(image,segmentation)
            image_embeds.append(image_embed)
            
        image_embeds = torch.stack(image_embeds).mean(0)
        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)

        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img):
        prompt=f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        # print(p_before_embeds.size(),img_embeds.size(),p_after_embeds.size())
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        # print(wrapped_atts_img.size(),atts_img.size())
        return wrapped_img_embeds, wrapped_atts_img

    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # the model might output a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        if self.args.dataset == 'chinese':
            hypo = {k: [' '.join(vi) for vi in v] for k, v in hypo.items()}
            ref = {k: [' '.join(vi) for vi in v] for k, v in ref.items()}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores




class MambaCTClassifierGen(MambaCTDownStream):
    def __init__(self, args):
        super().__init__(args)
        
        if hasattr(args, 'condition_names') and args.condition_names:
            self.condition_names = args.condition_names
        else:
            # Fallback to default labels if not provided
            self.condition_names = [
                'Ascites', 'Fluid', 'Free Air', 'Pneumoperitoneum',
                'Lesion', 'Mass', 'Nodule', 'Cyst', 'Tumor', 'Metastasis',
                'Hernia', 'Obstruction', 'Dilatation', 'Distension', 'Volvulus',
                'Calculus', 'Stone', 'Lithiasis', 'Calcification',
                'Thickening', 'Inflammation', 'Infection', 'Abscess', 'Fluid Collection',
                'Hemorrhage', 'Bleeding', 'Hematoma',
                'Thrombosis', 'Embolism', 'Infarct', 'Ischemia',
                'Perforation', 'Rupture',
                'Fracture', 'Dislocation',
                'Lymphadenopathy', 'Enlarged Node',
                'Anomaly', 'Deformity',
                'Fatty Liver', 'Steatosis', 'Cirrhosis',
                'Hydronephrosis', 'Hydroureter',
                'Cholecystitis', 'Cholelithiasis',
                'Appendicitis', 'Diverticulitis', 'Colitis',
                'Pancreatitis',
                'Splenomegaly', 'Hepatomegaly', 'No Finding'
            ]
        self.num_classes = len(self.condition_names)
        
        # Add classifier head (parallel path)
        self.classifier = nn.Sequential(
            nn.Linear(self.llama_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes)
        )
                
        self.classification_criterion = nn.BCEWithLogitsLoss()  # Added for multi-label classification
                
        self.val_classification_threshold = 0.5  # Default threshold for binary classification
        
        self.confidence_levels = {
            args.confidence_4: "highly likely",
            args.confidence_3: "likely",
            args.confidence_2: "possible",
            args.confidence_1: "low possibility of"
        }
        
        self.base_prompt = 'Generate a comprehensive and detailed diagnosis report for this abdomen CT scan image.'
        
        # Apply LoRA to vision encoder (from parent class)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                r=args.vis_r,
                lora_alpha=args.vis_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                modules_to_save=["classifier"],  # Save classifier parameters along with LoRA
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
            
        # Apply LoRA to classifier
        if args.classifier_use_lora:
            peft_config_classifier = LoraConfig(
                r=args.classifier_r,
                lora_alpha=args.classifier_alpha,
                target_modules=["0", "2"],  # Target both linear layers in classifier
                lora_dropout=args.lora_dropout,
                bias="none",
            )
            self.classifier = get_peft_model(self.classifier, peft_config_classifier)
            self.classifier.print_trainable_parameters()
            print('Loading classifier with LoRA -- Done')
        

        # Load pretrained weights if provided
        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    def forward(self, samples):
        image = samples["image"]
        
        # Get image embeddings
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        
        # Path A: Original report generation
        report_outputs = super().forward(samples)
        
        # Path B: Classification
        visual_cls = img_embeds.mean(dim=1)
        classification_logits = self.classifier(visual_cls)
        classification_probs = torch.sigmoid(classification_logits)
        
        # Format classification results
        batch_classifications = [
            self.format_classification_output(probs) 
            for probs in classification_probs
        ]
        
        return {
            "generation_loss": report_outputs["loss"],
            "classification_logits": classification_logits,
            "classification_output": batch_classifications
        }

    def training_step(self, batch, batch_idx):
        # Debug info
        if batch_idx == 0:  # Only print for first batch
            print("\nBatch contents:")
            print("Keys:", batch.keys())
            if 'disease_labels' in batch:
                print("Labels shape:", batch['disease_labels'].shape)
                print("Labels sample:", batch['disease_labels'][0])
                print("Non-zero labels:", (batch['disease_labels'] != 0).sum().item())
                
        # Get both generation and classification outputs
        outputs = self(batch)
        
        # Extract losses
        generation_loss = outputs["generation_loss"]
        classification_logits = outputs["classification_logits"]
        
        # Calculate classification loss
        if 'disease_labels' in batch:
            classification_loss = self.classification_criterion(
                classification_logits, 
                batch["disease_labels"].to(classification_logits.device)
            ) * 5
            # Debug classification prediction
            if batch_idx == 0:
                print("Classification logits:", classification_logits[0])
                print("Classification predictions:", torch.sigmoid(classification_logits[0]))
                print("Classification loss:", classification_loss.item())
        else:
            classification_loss = torch.tensor(0.0, device=self.device)
            if batch_idx == 0:
                print("No disease labels found in batch!")
        
        # Combine losses with weighting
        total_loss = generation_loss + classification_loss
        
        # Log losses
        self.log("train_generation_loss", generation_loss, prog_bar=True)
        self.log("train_classification_loss", classification_loss, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)
        
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        # Get report generation results
        report_hypo, report_ref = super().validation_step(batch, batch_idx)
        
        # Get classification predictions
        img_embeds, _ = self.encode_img(batch["image"])
        visual_cls = img_embeds.mean(dim=1)
        classification_logits = self.classifier(visual_cls)
        classification_probs = torch.sigmoid(classification_logits)
        
        # Calculate classification metrics if labels are available
        if "disease_labels" in batch:
            classification_loss = self.classification_criterion(
                classification_logits, 
                batch["disease_labels"].to(classification_logits.device)
            )
            self.log("val_classification_loss", classification_loss, sync_dist=True)
        
        # Format classification results
        classifications = [
            self.format_classification_output(probs) 
            for probs in classification_probs
        ]
        
        # Store both outputs
        self.val_step_outputs[-1].update({
            "classifications": classifications,
            "classification_probs": classification_probs.detach(),  # Keep on GPU
            "disease_labels": batch.get("disease_labels", None)
        })
        
        return report_hypo, report_ref, classifications


    def on_validation_epoch_end(self):
        # Process report generation metrics first
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        # Calculate generation metrics
        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        generation_metrics = self.score(ref=ref, hypo=hypo)
        
        # Initialize metrics dictionary with generation metrics
        metrics_to_save = {**generation_metrics}  # Start with generation metrics
        
        # Try to calculate classification metrics if available
        all_probs = []
        all_labels = []
        
        for output in self.val_step_outputs:
            if "classification_probs" in output and output["classification_probs"] is not None:
                # Move to the same device as the model
                probs = output["classification_probs"].to(self.device)
                all_probs.append(probs)
            if "disease_labels" in output and output["disease_labels"] is not None:
                # Move to the same device as the model
                labels = output["disease_labels"].to(self.device)
                all_labels.append(labels)
        
        # If we have both predictions and labels, calculate classification metrics
        if all_probs and all_labels:
            try:
                all_probs = torch.cat(all_probs, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                classification_metrics = self.calculate_classification_metrics(all_probs, all_labels)
                metrics_to_save.update(classification_metrics)  # Add classification metrics
            except Exception as e:
                print(f"Warning: Failed to calculate classification metrics: {e}")
        
        # Save results
        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        
        # Save metrics
        self.log_dict(metrics_to_save, sync_dist=True)
        
        # Calculate combined score using available metrics
        val_score = sum(generation_metrics[score_type] * weight 
                        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights))
        
        # Save checkpoint always
        if self.trainer.local_rank == 0:
            self.save_checkpoint(metrics_to_save)
            if val_score > self.val_score:
                self.val_score = val_score
        
        # Clear validation outputs
        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        # Get report generation results
        report_hypo, report_ref = super().test_step(samples, batch_idx)
        
        # Get image embeddings and classification results
        img_embeds, _ = self.encode_img(samples["image"])
        img_embeds = self.layer_norm(img_embeds)
        visual_cls = img_embeds.mean(dim=1)
        classification_logits = self.classifier(visual_cls)
        classification_probs = torch.sigmoid(classification_logits)
        
        # Format classification results for findings
        classification_outputs = [
            self.format_classification_output(probs) 
            for probs in classification_probs
        ]
        
        # Store both generation and classification outputs
        self.test_step_outputs[-1].update({
            "classification_output": classification_outputs,
            "classification_probs": classification_probs,
            "image_ids": samples["id"]  # changed from study_ids to image_ids
        })
        
        return report_hypo, report_ref, classification_outputs
    
    def on_test_epoch_end(self):
        ref, hypo, ids = [], [], []
        all_classification_outputs = []
        all_probs = []
        all_image_ids = []
        
        for output in self.test_step_outputs:
            ref.extend(output['ref'])
            hypo.extend(output['hypo'])
            ids.extend(output['id'])
            if 'classification_output' in output:
                all_classification_outputs.extend(output['classification_output'])
            if 'classification_probs' in output:
                all_probs.append(output['classification_probs'])
            if 'image_ids' in output:
                all_image_ids.extend(output['image_ids'])

        # Process report generation results
        ref_dict = {k:[v] for k, v in zip(ids, ref)}
        hypo_dict = {k:[v] for k, v in zip(ids, hypo)}
        generation_metrics = self.score(ref=ref_dict, hypo=hypo_dict)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)

        # Save results
        json.dump(hypo_dict, open(os.path.join(result_folder, f"test_result.json"), 'w', encoding='utf-8'), ensure_ascii=False)
        json.dump(ref_dict, open(os.path.join(result_folder, 'test_refs.json'), 'w', encoding='utf-8'), ensure_ascii=False)

        classification_results = {
            'results': [{'id': id_, 'findings': findings} 
                    for id_, findings in zip(ids, all_classification_outputs)]
        }
        
        json.dump(classification_results,
                open(os.path.join(result_folder, 'test_classification_findings.json'), 'w', encoding='utf-8'),
                indent=2, ensure_ascii=False)

        # Modified tensor conversion sequence
        if all_probs:
            all_probs = torch.cat(all_probs, dim=0).to(torch.float32).cpu().numpy()
            classification_df_data = []
            
            for image_id, probs in zip(all_image_ids, all_probs):
                row_dict = {'image_id': image_id}
                for condition, prob in zip(self.condition_names, probs):
                    label = 1.0 if prob >= self.val_classification_threshold else 0.0
                    row_dict[condition] = label
                classification_df_data.append(row_dict)

            import pandas as pd
            df = pd.DataFrame(classification_df_data)
            df.to_csv(os.path.join(result_folder, 'test_classification_labels.csv'), index=False)

        self.print(f"\nTest Results for {self.hparams.delta_file}:")
        self.print("\nGeneration Metrics:")
        for metric, value in generation_metrics.items():
            self.print(f"{metric}: {value:.4f}")
        
        self.test_step_outputs.clear()

    def format_classification_output(self, probs):
        """Convert classification probabilities to human-readable format"""
        findings = []
        probs = probs.float().cpu().detach().numpy()
        
        # Check No Finding first
        no_finding_idx = self.condition_names.index('No Finding')
        no_finding_prob = probs[no_finding_idx]
        
        if no_finding_prob >= 0.7:
            return [f"Normal study ({no_finding_prob:.1%} confidence)"]
        
        # If not a normal study, process other findings
        for i, (prob, condition) in enumerate(zip(probs, self.condition_names)):
            if i == no_finding_idx:  # Skip No Finding in detailed findings
                continue
                
            if prob >= 0.3:  # Only include findings above 30% confidence
                for threshold, description in sorted(self.confidence_levels.items(), reverse=True):
                    if prob >= threshold:
                        findings.append(f"{description} {condition} ({prob:.1%})")
                        break
        
        return findings if findings else ["No significant findings detected with high confidence"]
    
    def calculate_classification_metrics(self, probs, labels):
        """Calculate precision, recall, F1 score for classification"""
        predictions = (probs >= self.val_classification_threshold).float()
        
        # Calculate metrics per class
        metrics = {}
        for i, condition in enumerate(self.condition_names):
            true_positives = ((predictions[:, i] == 1) & (labels[:, i] == 1)).sum()
            false_positives = ((predictions[:, i] == 1) & (labels[:, i] == 0)).sum()
            false_negatives = ((predictions[:, i] == 0) & (labels[:, i] == 1)).sum()
            
            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            metrics[f"{condition}_precision"] = precision
            metrics[f"{condition}_recall"] = recall
            metrics[f"{condition}_f1"] = f1
        
        # Calculate macro averages
        metrics["macro_precision"] = sum([metrics[f"{c}_precision"] for c in self.condition_names]) / self.num_classes
        metrics["macro_recall"] = sum([metrics[f"{c}_recall"] for c in self.condition_names]) / self.num_classes
        metrics["macro_f1"] = sum([metrics[f"{c}_f1"] for c in self.condition_names]) / self.num_classes
        
        return metrics

