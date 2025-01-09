from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import os
from .database import TSDB, TS_item
import torch
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class TSRouter:
    def __init__(self):
        self.DB = TSDB()
        self.tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

        if torch.cuda.is_available():

            self.model.to('cuda')


        self.gen_kwargs = {
            "max_length": 256,
            "length_penalty": 0,
            "num_beams": 3,
            "num_return_sequences": 3,
        }

    def extract_triplets(self, text):
        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
        return triplets


    def post_edit_search(self, text):
        model_inputs = self.tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors='pt')
        generated_tokens = self.model.generate(
            model_inputs["input_ids"].to(self.model.device),
            attention_mask=model_inputs["attention_mask"].to(self.model.device),
            **self.gen_kwargs,
        )
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        for idx, sentence in enumerate(decoded_preds):
            s_triplets = self.extract_triplets(sentence)
            retrieve_t = self.DB.TS_search(s_triplets['head'],s_triplets['tail'])
            if retrieve_t is not None:
                return self.DB[retrieve_t].index
        return -100