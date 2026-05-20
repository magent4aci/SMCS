import torch
from torch import Tensor
import numpy as np
from transformers import AutoTokenizer, AutoModel
from global_utils.runtime_config import get_section, ensure_configured_path


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


class LinqEmbedMistral:
    def __init__(self, model_path, name, device):
        self.em_tokenizer = None
        self.em = None
        self.model_path = model_path
        self.name = name
        self.device = device
        self.load_model()

    def load_model(self):
        self.em = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.em_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_path)

    def obtain_embedding(self, sentences, tasks=None, batch_size=4, max_length=8192):
        # obtain the embedding of each sentence
        # if using batch mode, question is list[str], response is list[str], return list[float]
        if tasks is not None:
            assert len(sentences) == len(tasks)
            sentences = [get_detailed_instruct(t, s) if t != '' else s for s, t in zip(sentences, tasks)]
        ins_num = len(sentences)
        batch_index = list(range(0, ins_num, batch_size)) + [ins_num]
        embedding_list = []
        for i in range(len(batch_index) - 1):
            st_index, end_index = batch_index[i], batch_index[i + 1]
            input_texts = np.array(sentences)[st_index:end_index].tolist()
            # Tokenize the input texts
            batch_dict = self.em_tokenizer(input_texts, max_length=max_length, padding=True, truncation=True,
                                   return_tensors="pt").to(self.em.device)
            with torch.no_grad():
                outputs = self.em(**batch_dict)
            embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embedding_list.extend(embedding.detach().cpu().tolist())
        return embedding_list

model_dict = {
    'Linq-Embed-Mistral': LinqEmbedMistral,
}

em_path_dict = {
    "Linq-Embed-Mistral": "your path to Linq-Embed-Mistral",
}
em_path_dict.update(get_section("embedding_model_paths"))


def get_em_model_path(model_name):
    return ensure_configured_path(em_path_dict[model_name], model_name, "Embedding model path")


def auto_get_em(model_name):
    return model_dict[model_name]

if __name__ == '__main__':
    # question = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    # response_right = "Let's think step by step.\n1. Janet's ducks lay 16 eggs per day.\n2. She eats 3 eggs for breakfast every morning.\n3. She uses 4 eggs to bake muffins for her friends every day.\n4. The total number of eggs used for her personal consumption and baking is \\(3 + 4 = 7\\) eggs.\n5. The number of eggs remaining for her to sell at the farmers' market is \\(16 - 7 = 9\\) eggs.\n6. She sells each egg for $2.\n7. Therefore, the amount she makes from selling the eggs is \\(9 \\times 2 = 18\\) dollars.\n\nThe answer is 18."
    # response_wrong = "Let's think step by step.\n1. Janet's ducks lay 16 eggs per day.\n2. She eats 3 eggs for breakfast every morning.\n3. She uses 4 eggs to bake muffins for her friends every day.\n4. The total number of eggs used for her personal consumption and baking is \\(3 + 4 = 24\\) eggs.\n5. The number of eggs remaining for her to sell at the farmers' market is \\(16 - 7 = 9\\) eggs.\n6. She sells each egg for $156.\n7. Therefore, the amount she makes from selling the eggs is \\(9 \\times 2 = 88\\) dollars.\n\nThe answer is 25."

    # test auto_get_em
    model_name = "Linq-Embed-Mistral"
    model = auto_get_em(model_name)(get_em_model_path(model_name), model_name, 'auto')
    passages = [
        "I love you very much",
        "I like you.",
        "I hate you very much",
        "I LOVE you.",
        "I hate you.",
    ]
    tasks = ['Sentences with similar sentiment to this sentence']
    while 1:
        a = model.obtain_embedding(passages, tasks*5, batch_size=2)
