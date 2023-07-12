
import os
import re
from .utils import count_unique_tokens
import logging
logger = logging.getLogger(__name__)
try:
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
except ImportError:
    logger.warning("Could not import sentencepiece. Pruning embeddings of sentencepiece-based model is not available.")


class T5SentencepieceTokenizer:
    additional_special_token_ids = []

    

    @classmethod
    def get_token_ids(cls, tokenizer, dataiter=None, additional_tokens=None, additional_token_ids=None, min_count=1):
        token_ids = []
        #special_token_ids = list(set(tokenizer.all_special_ids) - set(tokenizer.additional_special_tokens_ids))
        special_token_ids = list(tokenizer.all_special_ids)
        cls.additional_special_token_ids = tokenizer.additional_special_tokens_ids


        normal_token_ids = []
        if dataiter is not None:
            token_ids_counter = count_unique_tokens(dataiter, tokenizer)
            normal_token_ids += [k for k,v in token_ids_counter.items() if v >= min_count]
        if additional_tokens is not None and len(additional_tokens) > 0:
            normal_token_ids += list(
                tokenizer.convert_tokens_to_ids(additional_tokens))
        if additional_token_ids is not None and len(additional_token_ids) > 0:
            normal_token_ids += list(additional_token_ids)
        normal_token_ids = list(set(normal_token_ids)-set(special_token_ids))
        token_ids = sorted(special_token_ids + normal_token_ids)
        
        return token_ids
        
    @classmethod
    def save_vocab(cls, tokenizer, token_ids, outdir):

        
        spm_token_ids = list(set(token_ids) - set(cls.additional_special_token_ids))
        m = sp_pb2_model.ModelProto()
        m.ParseFromString(tokenizer.sp_model.serialized_model_proto())

        spm_tokens = set([m.pieces[i].piece for i in spm_token_ids])
        new_pieces = [p for p in m.pieces if p.piece in spm_tokens]

        # delete all
        del m.pieces[:]
        m.pieces.extend(new_pieces)

        pruned_vocab_file = os.path.join(outdir, 'spiece.model')
        with open(pruned_vocab_file, 'wb') as f:
            f.write(m.SerializeToString())
        print(f"New embedding  pruned vocab file has been saved to {pruned_vocab_file}. Reintialize the tokenizer!")