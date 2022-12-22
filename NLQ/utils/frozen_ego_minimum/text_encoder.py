import os
import json
import torch
import inspect
import transformers
from pprint import pprint

import utils.frozen_ego_minimum.model.model as module_arch
from utils.frozen_ego_minimum.utils import state_dict_data_parallel_fix

# import model.model as module_arch
# from utils import state_dict_data_parallel_fix

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EgoVLP_TextEncoder:
    def __init__(self, checkpoint):
        
        super(EgoVLP_TextEncoder, self).__init__()
        # build model architecture
        config = json.load(open(f'./utils/frozen_ego_minimum/configs/ego4d-nlq-l_{checkpoint}.json','r'))
        self.model = self.initialize(config, 'arch', module_arch) 
        checkpoint = f'./utils/frozen_ego_minimum/pretrained/{checkpoint}.pth'
        if os.path.exists(checkpoint):
            checkpoint = torch.load(checkpoint)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.model.state_dict())
            self.model.load_state_dict(new_state_dict, strict=True)
        else:
            raise ValueError('Checkpoint not found')
            
        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()

        # Build tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
    def initialize(self, config, name, module,  *args, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_name = config[name]['type']
        module_args = dict(config[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)

        return getattr(module, module_name)(*args, **module_args)


    def encode(self, text):
        with torch.no_grad():
            assert len(text)==1
            text = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            text = {key: val.cuda() for key, val in text.items()}
            text['input_ids'] = self.model.compute_text_tokens(text)[0]
            text = {k:v.detach().cpu() for k,v in text.items()}
            return text


if __name__ == "__main__":
    text_encoder = EgoVLP_TextEncoder()
    text_embed   = text_encoder.encode(['I have 4 apples'])
    print(text_embed)


