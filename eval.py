from model import *
import heapq

model.load_state_dict(torch.load('model_alt.pth'))

# A function to generate the output sequence autoregressively using the greedy decoder algorithm
# Usually, we would utilize something like beam search
def greedy_decode(model, src, max_len, start_symbol):
    src = src.to(DEVICE)
    memory = model.encode(src.view(1,-1), None)
    memory = memory.to(DEVICE)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        out = model.decode(ys.view(1,-1), memory, None)
        prob = nn.functional.softmax(model.generator(out[:, -1]),dim=1)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def beam_search_decode(model, src, max_len, start_symbol, beam_width):
    src = src.to(DEVICE)
    memory = model.encode(src.view(1,-1), None)
    memory = memory.to(DEVICE)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    completed_sequences = []
    sequences = [(0, ys)]
    
    for _ in range(max_len-1):
        all_candidates = []
        for score, sequence in sequences:
            out = model.decode(sequence.view(1,-1), memory, None)
            prob = nn.functional.softmax(model.generator(out[:, -1]),dim=1)
            top_scores, top_words = torch.topk(prob, beam_width, dim=1)
            
            for i in range(beam_width):
                candidate_score = score + top_scores[0][i].item()
                candidate_sequence = torch.cat([sequence, top_words[0][i].unsqueeze(0).unsqueeze(0)], dim=1)
                all_candidates.append((candidate_score, candidate_sequence))
        
        sequences = heapq.nlargest(beam_width, all_candidates, key=lambda x: x[0])
        
        for score, sequence in sequences:
            if sequence[0][-1].item() == EOS_IDX:
                completed_sequences.append((score, sequence))
        
        sequences = sequences[:beam_width - len(completed_sequences)]
        
        if len(sequences) == 0:
            break
    
    if len(completed_sequences) > 0:
        completed_sequences.sort(key=lambda x: x[0], reverse=True)
        ys = completed_sequences[0][1]
    else:
        ys = sequences[0][1]
    
    return ys

# Function for translation
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    tgt_tokens = beam_search_decode(model,  src, max_len=num_tokens + 5, start_symbol=SOS_IDX, beam_width=20).flatten()
    return " ".join(vocabularies[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<sos>", "").replace("<eos>", "")

# Trying to translate from English to Python
print(translate(model, "write a python program to add two numbers "))